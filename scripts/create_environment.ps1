#requires -Version 5.1
<#
.SYNOPSIS
Create the project Conda environment and retain installation evidence.
.DESCRIPTION
Uses config/environment_ec.yaml as the only dependency specification. An existing
environment is left untouched unless -Update is supplied. -DryRun asks Conda to
solve without installing packages or registering a kernel. No Conda installation,
shell initialization, or profile modification is performed.
#>
[CmdletBinding()]
param(
    [string]$CondaExecutable,
    [switch]$Update,
    [switch]$DryRun,
    [string]$EvidenceDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$EnvironmentName = 'pypsa-earth-ec'
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Specification = Join-Path $RepositoryRoot 'config/environment_ec.yaml'
if ($Update -and $DryRun) {
    throw '-Update and -DryRun cannot be combined: this Conda env update interface has no dry-run option.'
}

function Find-Conda {
    param([string]$ExplicitPath)
    if ($ExplicitPath) {
        if (Test-Path -LiteralPath $ExplicitPath -PathType Leaf) {
            return (Resolve-Path -LiteralPath $ExplicitPath).Path
        }
        $command = Get-Command $ExplicitPath -CommandType Application -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
        throw "Conda executable not found: $ExplicitPath"
    }

    $candidates = New-Object 'System.Collections.Generic.List[string]'
    if ($env:CONDA_EXE) { $candidates.Add($env:CONDA_EXE) }
    foreach ($commandName in @('conda.exe', 'conda.bat', 'conda')) {
        $command = Get-Command $commandName -CommandType Application -ErrorAction SilentlyContinue
        if ($command) { $candidates.Add($command.Source) }
    }
    $roots = @($env:USERPROFILE, $env:LOCALAPPDATA, $env:ProgramData,
        $env:ProgramFiles, ${env:ProgramFiles(x86)}, ($env:SystemDrive + '\'))
    foreach ($root in $roots) {
        if (-not $root) { continue }
        foreach ($distribution in @('miniconda3', 'anaconda3', 'miniforge3', 'mambaforge',
                'Miniconda', 'Anaconda', 'Miniforge', 'Mambaforge')) {
            $basePath = Join-Path $root $distribution
            foreach ($relative in @('Scripts/conda.exe', 'condabin/conda.bat', '_conda.exe')) {
                $candidates.Add((Join-Path $basePath $relative))
            }
        }
    }
    if ($env:USERPROFILE) {
        $registry = Join-Path $env:USERPROFILE '.conda/environments.txt'
        if (Test-Path -LiteralPath $registry -PathType Leaf) {
            foreach ($entry in Get-Content -LiteralPath $registry) {
                $registeredPath = $entry.Trim()
                if (-not $registeredPath -or $registeredPath.StartsWith('#')) { continue }
                $bases = @($registeredPath)
                $parent = Split-Path -Parent $registeredPath
                if ($parent -and (Split-Path -Leaf $parent) -eq 'envs') {
                    $bases += Split-Path -Parent $parent
                }
                foreach ($basePath in $bases) {
                    foreach ($relative in @('Scripts/conda.exe', 'condabin/conda.bat', '_conda.exe')) {
                        $candidates.Add((Join-Path $basePath $relative))
                    }
                }
            }
        }
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw 'Conda was not found. Install Conda separately or pass -CondaExecutable with its executable path.'
}

function Invoke-Conda {
    param([string[]]$Arguments)
    Write-Host ('> conda ' + ($Arguments -join ' '))
    # Windows PowerShell wraps native stderr in ErrorRecord even on success.
    # Preserve it in the transcript, then decide success using the exit code.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $stderrPath = Join-Path $EvidenceDirectory ('conda-stderr-' + [guid]::NewGuid().ToString('N') + '.log')
        $lines = @(& $script:ResolvedConda @Arguments 2> $stderrPath)
        $nativeExitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousPreference }
    $strings = @($lines | ForEach-Object { $_.ToString() })
    foreach ($line in $strings) { Write-Host $line }
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath | ForEach-Object { Write-Host $_ }
    }
    if ($nativeExitCode -ne 0) {
        throw "Conda command failed with exit code ${nativeExitCode}: $($Arguments -join ' ')"
    }
    return $strings
}

if (-not (Test-Path -LiteralPath $Specification -PathType Leaf)) {
    throw "Missing dependency specification: $Specification"
}
$script:ResolvedConda = Find-Conda $CondaExecutable
if (-not $EvidenceDirectory) {
    $EvidenceDirectory = Join-Path $RepositoryRoot ('results/environment_creation/' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
elseif (-not [IO.Path]::IsPathRooted($EvidenceDirectory)) {
    $EvidenceDirectory = Join-Path $RepositoryRoot $EvidenceDirectory
}
$null = New-Item -ItemType Directory -Path $EvidenceDirectory -Force
$EvidenceDirectory = (Resolve-Path -LiteralPath $EvidenceDirectory).Path
$transcriptStarted = $false
try {
    $null = Start-Transcript -LiteralPath (Join-Path $EvidenceDirectory 'creation-transcript.txt') -NoClobber
    $transcriptStarted = $true
    Write-Host "Conda: $script:ResolvedConda"
    Write-Host "Specification: $Specification"
    $version = Invoke-Conda -Arguments @('--version')
    $version | Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'conda-version.txt') -Encoding UTF8
    Copy-Item -LiteralPath $Specification -Destination (Join-Path $EvidenceDirectory 'environment-input.yaml')
    (Get-FileHash -LiteralPath $Specification -Algorithm SHA256).Hash |
        Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'environment-input.sha256') -Encoding UTF8

    $environmentInfo = (Invoke-Conda -Arguments @('env', 'list', '--json') | Out-String) | ConvertFrom-Json
    $existing = @($environmentInfo.envs | Where-Object {
        (Split-Path -Leaf $_.TrimEnd([char[]]'\/')) -eq $EnvironmentName
    })
    if ($existing.Count -gt 0 -and -not $Update) {
        throw "Environment '$EnvironmentName' already exists. Use -Update explicitly to modify it. Evidence: $EvidenceDirectory"
    }
    if ($Update -and $existing.Count -eq 0) {
        throw "Environment '$EnvironmentName' does not exist. Omit -Update to create it."
    }
    if ($existing.Count -gt 1) {
        throw "Multiple environments named '$EnvironmentName' were found. Resolve the ambiguity before updating."
    }
    if ($Update) { $operation = 'update' } else { $operation = 'create' }
    $creationArguments = @('env', $operation, '--name', $EnvironmentName, '--file', $Specification, '--quiet')
    if (-not $Update) { $creationArguments += '--yes' }
    if ($DryRun) { $creationArguments += '--dry-run' }
    $null = Invoke-Conda -Arguments $creationArguments
    if ($DryRun) {
        Write-Host "Dry run completed. No environment installation or kernel registration was requested. Evidence: $EvidenceDirectory"
    }
    else {
        # Keep resolved installation evidence even if a later runtime check fails.
        $export = Invoke-Conda -Arguments @('env', 'export', '--name', $EnvironmentName)
        $export | Where-Object { $_ -notmatch '^prefix\s*:' } |
            Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'environment-resolved.yaml') -Encoding UTF8
        $explicit = Invoke-Conda -Arguments @('list', '--name', $EnvironmentName, '--explicit', '--sha256')
        [IO.File]::WriteAllLines((Join-Path $EvidenceDirectory 'conda-explicit.txt'),
            [string[]]$explicit, (New-Object Text.UTF8Encoding($false)))
        Invoke-Conda -Arguments @('run', '--no-capture-output', '--name', $EnvironmentName, 'python', '-m', 'pip', 'freeze') |
            Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'pip-freeze.txt') -Encoding UTF8
        # Conda-built wheels may freeze to build-machine file:// references.
        # Retain that evidence and also record portable package/version rows.
        Invoke-Conda -Arguments @('run', '--no-capture-output', '--name', $EnvironmentName, 'python', '-m', 'pip', 'list', '--format=freeze') |
            Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'pip-versions.txt') -Encoding UTF8
        Invoke-Conda -Arguments @('run', '--no-capture-output', '--name', $EnvironmentName, 'python', '-m', 'pip', 'check') |
            Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'pip-check.txt') -Encoding UTF8
        $runtimeCheck = @'
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

expected = "pypsa-earth-ec"
if Path(sys.prefix).name.casefold() != expected.casefold():
    raise RuntimeError(f"Expected {expected} interpreter, got {sys.executable} (prefix {sys.prefix})")
modules = {
    "pypsa": "pypsa", "numpy": "numpy", "pandas": "pandas",
    "scipy": "scipy", "xarray": "xarray", "h5netcdf": "h5netcdf",
    "netCDF4": "netCDF4", "yaml": "PyYAML", "requests": "requests",
    "jsonschema": "jsonschema", "nbformat": "nbformat", "nbclient": "nbclient",
    "ipykernel": "ipykernel", "linopy": "linopy", "highspy": "highspy",
    "geopandas": "geopandas", "matplotlib": "matplotlib",
    "seaborn": "seaborn", "networkx": "networkx",
}
versions = {}
for module, distribution in modules.items():
    importlib.import_module(module)
    versions[distribution] = importlib.metadata.version(distribution)
import highspy
import linopy
if "highs" not in linopy.available_solvers:
    raise RuntimeError("HiGHS is not available through linopy")
print(json.dumps({"executable": sys.executable, "prefix": sys.prefix,
                  "python": sys.version, "packages": versions,
                  "highs_runtime": highspy.Highs().version(),
                  "available_solvers": list(linopy.available_solvers)}, indent=2))
'@
        $runtimeCheckPath = Join-Path $EvidenceDirectory 'verify-runtime.py'
        $runtimeCheck | Set-Content -LiteralPath $runtimeCheckPath -Encoding UTF8
        $runtime = Invoke-Conda -Arguments @('run', '--no-capture-output', '--name', $EnvironmentName, 'python', $runtimeCheckPath)
        $runtime | Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'runtime-verification.txt') -Encoding UTF8
        $null = Invoke-Conda -Arguments @('run', '--no-capture-output', '--name', $EnvironmentName,
            'python', '-m', 'ipykernel', 'install', '--user', '--name', $EnvironmentName,
            '--display-name', 'Python (pypsa-earth-ec)')

        Write-Host "Environment and user kernel are ready. Evidence: $EvidenceDirectory"
    }
}
finally {
    if ($transcriptStarted) { $null = Stop-Transcript }
}

# Reconstructing the Phase 06H environment

The owner authorized a fresh `pypsa-earth-ec` Conda environment on this laptop
on 2026-10-20, including package installation and kernel registration. This is a
specific exception to the earlier policy of using the other laptop's existing
environment. No original environment freeze was available here.

## Dependency document

`config/environment_ec.yaml` is the installer's single dependency specification.
Its scope is the canonical Phase 06H notebooks 00, 01, 02, 03, 04, 05, 06 and
06a, including imported modules and the Jupyter runtime. It does not attempt to
reconstruct the entire upstream PyPSA-Earth/Snakemake toolchain or legacy notebooks.

Evidence: `docs/ENVIRONMENT.md`, the saved Phase 06H executed preflight outputs,
and the imports reachable from those eight notebooks in `src/`. Historical pins
cover Python 3.11.14, PyPSA 0.30.3, NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.2,
xarray 2025.1.2, h5netcdf 1.7.3, NetCDF4 1.7.3, Matplotlib 3.10.7,
GeoPandas 1.1.1, HiGHS bindings 1.11.0, PyYAML 6.0.3, requests 2.32.5,
jsonschema 4.25.1, nbformat 5.10.4 and nbclient 0.10.2.

Linopy 0.5.8 is an explicit reconstruction choice, not a recovered historical
version. PyPSA 0.30.3 requires Linopy >=0.3.12 and HiGHS; Linopy 0.5.8 supports
the retained xarray version. Other direct dependencies have bounded version
ranges; the resolved export records their actual versions and transitive builds.
See [PyPSA's versioned dependency metadata](https://github.com/PyPSA/PyPSA/blob/v0.30.3/pyproject.toml)
and [Linopy's versioned metadata](https://github.com/PyPSA/linopy/blob/v0.5.8/pyproject.toml).

GeoPandas supplies its GIS dependencies through Conda. NetworkX supports topology;
seaborn supports 06a; packaging supports runtime version checks. The notebook
packages explicitly include ipykernel, IPython, JupyterLab, jupyter_client,
nbclient, nbformat and debugpy. VS Code's Python and Jupyter extensions must be
installed in the editor separately; select **Python (pypsa-earth-ec)** as kernel.

Unused direct entries removed from the old broad specification include atlite,
powerplantmatching, earth-osm, Snakemake, commercial Gurobi, GLPK, SCIP,
weather-regeneration/GIS visualization extras, Excel readers and currency/download
helpers. The supported pipeline consumes prepared profiles and CSV interfaces,
does not regenerate ERA5, and explicitly selects HiGHS. Transitive dependencies
required by PyPSA are still resolved normally.

Only conda-forge is used; `nodefaults` prevents appending workstation default
channels. This follows the [Conda environment-file specification](https://conda.org/learn/specifications/exchange/environment-yml/).
The YAML is a direct-dependency specification, not a platform-independent lock.
An explicit export is platform-specific; a fresh solve on another platform still
requires acceptance testing.

## Create the environment

From the repository root in Windows PowerShell 5.1 or newer:

```powershell
.\scripts\create_environment.ps1
```

If Conda is not automatically discovered:

```powershell
.\scripts\create_environment.ps1 -CondaExecutable '<Conda installation>\Scripts\conda.exe'
```

Conda itself must already be installed. The script does not install a distribution,
change shell profiles, or open Anaconda Navigator. It uses Conda's command line,
creates the named environment outside the repository, verifies imports and HiGHS,
and registers a user Jupyter kernel pointing to that environment. It refuses an
existing environment unless `-Update` is explicitly supplied. `-DryRun` solves
without installing or registering a kernel. Native command failures stop the run.
`-Update -DryRun` is rejected because the supported Conda update interface does
not offer a dry-run option.

Each build retains its transcript, dependency-document checksum, resolved YAML,
explicit Conda package URLs, `pip freeze`, `pip check` and runtime versions under
`results/environment_creation/`. The exports capture the new environment; they
must not be described as the missing freeze from the other laptop.
Conda-built distributions can make `pip freeze` contain build-machine `file://`
references; `pip-versions.txt` also records plain package/version rows. Use the
SHA-256-pinned `config/locks/pypsa-earth-ec-win-64.explicit.txt` for exact Conda
build replay on Windows, rather than installing those pip file references:

```powershell
conda create -n pypsa-earth-ec --file config/locks/pypsa-earth-ec-win-64.explicit.txt
conda run -n pypsa-earth-ec python -m ipykernel install --user --name pypsa-earth-ec --display-name 'Python (pypsa-earth-ec)'
```

This replay command creates an absent environment; it does not replace an
existing one. Installation through the main PowerShell script additionally runs
the import, solver and dependency-health checks.

## Execute the unchanged notebook pipeline

The owner subsequently authorized a full-year solve because unchanged notebook
06a requires an 8,760-hour solved artifact. The original one-week limit was
superseded. The acceptance helper requires `--allow-full-year` explicitly:

```powershell
conda run --no-capture-output -n pypsa-earth-ec python scripts/verify_environment_pipeline.py --checkout '<isolated Phase 06H checkout>' --evidence '<evidence directory>' --restore-pinned-inputs --allow-full-year
```

The target checkout must contain the Phase 06H workflow; historical `main` alone
does not. Prepare a separate checkout of the intended revision, preserving the
exact Git blob bytes (disable automatic newline conversion for that checkout).
The manifest contains mixed newline conventions: some hashes require LF, some
require CRLF, and some require CRLF record terminators with LF inside quoted
provenance fields. Neither a uniformly LF nor a uniformly CRLF checkout is
sufficient. `--restore-pinned-inputs` tries those representations of Git source
blobs in the isolated checkout and installs a candidate only when its byte count
and SHA-256 exactly match the existing manifest. It never repairs a hash by
changing the manifest, and leaves generated products to their notebooks. The
four small recovery tests cover exact preservation of embedded provenance,
rejection of unmatched checksums, generated-product exclusion and path containment.

The helper obtains topology inputs via the existing
manifest preflight before the offline topology notebook; notebook 00 obtains its
own published inputs. The GADM input is about 99 MB. No full ERA5 cutout is needed.

The helper executes the existing `src.notebook_runner` sequentially, captures logs
and executed copies, stops at the first error, and hashes the notebooks, source
modules and scientific configuration before and after. Generated outputs remain
in the isolated checkout. Each invocation uses a new evidence subdirectory so
retries preserve earlier failure logs. It does not edit notebooks, bypass preflight, alter
checksums, relax scientific criteria or substitute datasets.

Successful execution demonstrates environment compatibility. It does not change
Phase 06H's scientific status: the baseline remains provisional and its failed
nonconventional carrier screen and unresolved statistical boundary remain reported.

## Acceptance run

See `reports/ENVIRONMENT_CREATION_REPORT.md` for the tested revision, commands,
outcomes and retained evidence paths.

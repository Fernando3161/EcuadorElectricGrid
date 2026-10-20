$global:LASTEXITCODE = 0
if ($args[0] -eq '--version') { 'conda mock for installer checks'; return }
if ($args[0] -eq 'env' -and $args[1] -eq 'list') {
    Write-Error 'simulated Conda stderr warning'
    '{"envs": []}'
    return
}
if ($args[0] -eq 'env' -and $args[1] -eq 'create') {
    if ($args -notcontains '--dry-run' -or $args -notcontains '--quiet') { exit 7 }
    'mock solve completed; no packages installed'
    return
}
exit 9

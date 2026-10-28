param([string]$RepositoryUrl="https://github.com/Fernando3161/EcuadorElectricGrid.git",[string]$RepositoryRef="refactor/v1-phase-10-runtime-prune",[string]$RunId="")
$env:REPOSITORY_URL=$RepositoryUrl
$env:REPOSITORY_REF=$RepositoryRef
if ($RunId) { $env:RUN_ID=$RunId } else { Remove-Item Env:RUN_ID -ErrorAction SilentlyContinue }
docker compose build
docker compose run --rm reproducibility

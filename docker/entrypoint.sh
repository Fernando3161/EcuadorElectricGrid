#!/usr/bin/env bash
set -Eeuo pipefail
repo_url="${REPOSITORY_URL:?REPOSITORY_URL is required}"
repo_ref="${REPOSITORY_REF:?REPOSITORY_REF is required}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}}"
export_root="/export/${run_id}"
work_root="/tmp/ecuador-${run_id}"
clone_root="${work_root}/repository"
mkdir -p "$export_root" "$clone_root"
exec > >(tee "$export_root/run.log") 2>&1
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Phase 11 Docker run ${run_id} started at ${started_at}"
git config --global core.autocrlf false
git config --global core.eol lf
git clone --no-tags --branch "$repo_ref" --single-branch "$repo_url" "$clone_root"
cd "$clone_root"
resolved_sha="$(git rev-parse HEAD)"
micromamba create -y -n pypsa-earth-ec -f config/environment_ec.yaml
eval "$(micromamba shell hook --shell bash)"
micromamba activate pypsa-earth-ec
python -m pip check
python -c "import sys, importlib.metadata as m; print(sys.version); [print(x, m.version(x)) for x in ('pypsa','numpy','pandas','scipy','linopy','highspy','nbclient','nbformat')]"
# Notebook 04 intentionally validates offline. Hydrate its external dependency
# first through the manifest-driven downloader so the notebook can remain
# deterministic and still enforce the recorded size/SHA-256.
python -m src.preflight --stage 04_topology
sequence=(
  notebooks/00_preflight_and_data_inventory.ipynb
  notebooks/01_prepare_2022_demand.ipynb
  notebooks/02_prepare_2022_generation.ipynb
  notebooks/03_prepare_renewable_and_hydro_profiles.ipynb
  notebooks/04_build_and_validate_2022_topology.ipynb
  notebooks/05_assemble_and_smoke_test_2022_network.ipynb
  notebooks/06_run_and_validate_2022_full_year.ipynb
  notebooks/06a_diagnose_2022_validation.ipynb
  notebooks/07_prepare_master_plan_expansion.ipynb
  notebooks/08_future_masterplan_screening.ipynb
  notebooks/08a_hydro_shortage_2026.ipynb
)
mkdir -p "$export_root/executed_notebooks" "$export_root/notebook_logs"
status=success
for notebook in "${sequence[@]}"; do
  name="$(basename "$notebook" .ipynb)"
  echo "=== RUN ${name} ==="
  if python -m src.notebook_runner "$notebook" --output "$export_root/executed_notebooks/${name}.ipynb" --timeout "${RUN_TIMEOUT:-600}" >"$export_root/notebook_logs/${name}.log" 2>&1; then
    echo "${name},success" >> "$export_root/notebook_status.csv"
  else
    rc=$?
    echo "${name},failure,${rc}" >> "$export_root/notebook_status.csv"
    status=failure
    break
  fi
done
for tree in data/processed data/interim results reports/figures; do
  if [[ -d "$tree" ]]; then mkdir -p "$export_root/$tree"; tar -C "$clone_root" --exclude='*/__pycache__' --exclude='*/.ipynb_checkpoints' -cf - "$tree" | tar -C "$export_root" -xf -; fi
done
python - "$export_root" "$repo_url" "$repo_ref" "$resolved_sha" "$started_at" "$status" <<'PY'
import hashlib, json, platform, sys
from pathlib import Path
out, url, ref, sha, started, status = sys.argv[1:]
out=Path(out)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
spec=Path('/tmp') / ('ecuador-'+out.name) / 'repository/config/environment_ec.yaml'
meta={'repository_url':url,'requested_ref':ref,'resolved_commit':sha,'clone_timestamp_utc':started,'status':status,'base_image':'mambaorg/micromamba:2.3.0@sha256:d008b2611d604c4e0c8e3652c7f9dca7cd65a103d68a2db30112702078a45c7a','environment_spec':'config/environment_ec.yaml','environment_spec_sha256':digest(spec) if spec.exists() else None,'python':platform.python_version(),'notebook_sequence':(out/'notebook_status.csv').read_text().splitlines()}
(out/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
files=[]
for p in sorted(out.rglob('*')):
    if p.is_file() and p.name != 'artifact_inventory.json': files.append({'path':p.relative_to(out).as_posix(),'size':p.stat().st_size,'sha256':digest(p)})
(out/'artifact_inventory.json').write_text(json.dumps(files,indent=2)+'\n')
PY
echo "Run status: ${status}"
echo "Export: ${export_root}"
test "$status" = success

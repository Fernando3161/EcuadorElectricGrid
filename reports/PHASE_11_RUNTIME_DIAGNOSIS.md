# Phase 11 runtime dependency and kernel failure diagnosis

Investigated on 2026-10-27. This is a failure diagnosis, not Docker or scientific
acceptance. Run `phase11-20261027-173233` cloned commit
`be3d33548f4e60dede49990bac3d591c72644a15` and completed notebooks 00 through 05.

## Notebook 06 failure

The preserved notebook log ends with `nbclient.exceptions.DeadKernelError` during
HiGHS postsolve (the last reported solver time is 251 seconds). Docker events for
container `7a46ae31a75159008eee537c8290c132fc8574cce75bf5d5786ef0f4755bda05`
record `oom` at Unix timestamp 1793119307, followed by container exit at
1793119314. The entire container ran for 560 seconds, including environment setup.
This was an out-of-memory failure, not the 600-second cell timeout.

Docker reported 7,904,346,112 bytes of total memory (about 7.36 GiB); the Windows
host reported 16,325,677,056 bytes (about 15.20 GiB). No successful solver
termination or solved-network export was established. A retry requires review
of Docker/WSL memory resources; removing the timeout does not resolve exhaustion.
No host resource settings or solver options were changed by this diagnosis.

## Documentary PDFs versus runtime inputs

The three `cap_3_demanda.pdf`, `cap_4_generacion.pdf`, and
`cap_5_transmision.pdf` files are local, ignored by `.gitignore`'s `*.pdf` rule,
and absent from the committed tree. Their manifest entries previously required
them for stage 07 despite providing no download URLs.

Searching all supported notebooks and `src/` found no runtime extraction from
these chapters. Only manifest preflight reads their bytes for integrity checks;
the notebook introduction mentions them as older source evidence. Their source
entries, sizes, hashes, and local files are retained. They now belong to the
explicit `planning_reference_audit` stage, outside the simulation dependency
closure. Its offline integrity audit is available as:

```powershell
conda run -n pypsa-earth-ec python -m src.preflight --stage planning_reference_audit --offline
```

`AJUSTE-PME-2026.pdf` remains required: `src/planning_products.py` reads its bytes
to record SHA-256 provenance in the planning products. Numeric planning values
come from the retained CSV transcriptions, not live PDF parsing. Its configured
download and checksum checks remain unchanged.

## Execution timeout

The notebook runner and Docker defaults previously set 600 seconds per cell.
Both now default to `-1`, the nbclient unlimited-execution setting. An explicitly
supplied positive `--timeout` or `RUN_TIMEOUT` still requests a limit. Network
download timeouts are separate and unchanged.

## Verification and handoff

Focused checks cover runtime dependency closure, strict missing-file behavior
for the explicit reference audit, and real execution of a tiny notebook with the
unlimited default in `pypsa-earth-ec`. Compose configuration must resolve the
default `RUN_TIMEOUT` to `-1`.

Completed checks:

- `conda run -n pypsa-earth-ec python -m unittest discover -s tests -p test_runtime_preflight.py -v`: three tests passed, including a real tiny kernel run.
- `conda run -n pypsa-earth-ec python -m src.preflight --stage planning_reference_audit --offline`: all three local chapter PDFs passed exact integrity checks.
- `docker compose config --format json`: resolved `RUN_TIMEOUT` is `-1`.
- `git diff --check`: passed.

The full annual solve was not repeated during diagnosis. Repository changes
must be reviewed and published by the owner before a runtime clone can use them;
Docker infrastructure must also be rebuilt. No phase completion is claimed.

# Environment creation and acceptance

Date: 2026-10-20. Pipeline under test: Phase 06H consolidation,
revision `342d8de` (`v1-pahe-06h-consolidation`).

## Scope and authorization

The owner requested a branch from `main`, reconstruction of the named
`pypsa-earth-ec` environment, a PowerShell installer driven by a dependency
document, execution of notebooks 00 through 06a, and merge into Phase 06H only
after successful notebook execution. Git forbids spaces in branch names, so the
branch is `environment-creation`, based on `main` revision `f4b119a`.

The owner initially limited notebook 06 to one week, then explicitly authorized
a full-year simulation when informed that unchanged 06a requires 8,760 solved
snapshots. No weekly adaptation or weakening of annual checks was implemented.

## Files and assumptions

- `config/environment_ec.yaml`: active-pipeline direct dependencies and known
  historical pins, including HiGHS and VS Code/Jupyter kernel support.
- `scripts/create_environment.ps1`: Conda discovery, create/update safeguards,
  dependency-file installation, kernel registration and resolved exports.
- `scripts/verify_environment_pipeline.py`: sequential execution of the existing
  notebook runner in an isolated checkout, logs and pipeline fingerprints.
- `docs/ENVIRONMENT_CREATION.md`: dependency evidence, reconstruction choices,
  commands, scope and limits.
- `config/locks/pypsa-earth-ec-win-64.explicit.txt`: exact resolved Windows Conda
  package URLs and SHA-256 values; replay was verified with an offline dry run.
- `tests/test_environment_creation.py`: four tiny integrity/recovery tests.

No original notebook, source module, artifact manifest, validation contract or
scientific input was edited. Exact Git blob bytes were used for the isolated checkout:
the system Git configuration converts LF CSVs to CRLF, which otherwise breaks
manifest size and SHA-256 checks. The two committed NLR CSVs were verified
against their existing manifest digests after extraction with
`git -c core.autocrlf=false archive`. Other manifest entries retain Windows CSV
record endings, including LF within quoted multiline fields. The acceptance
helper recovered those representations from Git only when both size and the
unchanged manifest SHA-256 matched. Five source interfaces required recovery:
demand template, carrier metadata, generation source, plant-location evidence
and hydro assignments. These were input provisioning operations in the isolated
checkout, with no changes to field values or the owner working tree.

Dependencies were inferred from the active import graph and historical runtime
evidence, not from a recovered freeze of the old laptop. Linopy 0.5.8 is a
reconstruction choice. No scientific calibration, validation tolerance, source
edition, weather year or carrier treatment was changed.

## Execution evidence

Local installation evidence is retained in
`results/environment_creation/build/`; notebook logs and executed copies are
retained in `results/environment_creation/pipeline/`. Regenerated data, figures
and networks remain in the separate validation worktree. Large products are
excluded from this environment branch.

Commands used (paths shown as placeholders for portability):

```powershell
git worktree add -b environment-creation '<environment worktree>' main
git worktree add --detach '<validation worktree>' 342d8de
git -c core.autocrlf=false archive --format=tar --output='<snapshot.tar>' 342d8de
tar -xf '<snapshot.tar>' -C '<validation worktree>'
.\scripts\create_environment.ps1 -EvidenceDirectory '<installation evidence>'
```

PowerShell 5.1 syntax parsing, explicit native argument binding, and the initial
Conda dependency dry run passed. The installer created the new environment;
its runtime imports and HiGHS discovery passed, `pip check` reported no broken
requirements, and the user kernel was registered with debugger support. Exact
Conda builds, a raw pip freeze and portable pip version rows were retained.
The final installer was also exercised with a simulated Conda warning on stderr:
JSON parsing and the creation dry-run command succeeded without mixing the
warning into machine-readable stdout. Existing-environment refusal was tested
and correctly stopped without changing packages. The unsupported combination
`-Update -DryRun` was identified and now receives an explicit early rejection.

The committed Windows lock passed `conda create --file ... --dry-run --offline`
with exit 0; this check did not create another environment. Four new deterministic
input-recovery tests passed. The original repository suite initially had missing
input errors before data preparation. After preparation, 90 tests ran with
87 passes and three existing governance failures: stale file classifications,
incomplete module inventory, and a workstation path in the old Phase 06D report.
No pipeline or governance files were changed to hide these failures. Logs:
`unit-tests.log`, `unit-tests-after-data.log`, `environment-tool-tests.log`,
`installer-mock-dryrun.log`, `existing-environment-refusal.log` and
`explicit-lock-dryrun.log`, under the evidence directory above.

The original failed notebook attempts are retained: the first stopped at the
demand-template newline mismatch; the next stopped at multiline generation CSV
record endings. Exact manifest-byte recovery resolved both without changing
the source fields or validation requirements. The subsequent full sequence is
recorded under `pipeline/20261020-102228/`.

## Acceptance result: PASS for environment compatibility

The new named environment executed all eight unchanged notebooks sequentially:

| Notebook | Exit code | Seconds |
|---|---:|---:|
| 00 preflight | 0 | 5.39 |
| 01 demand | 0 | 15.39 |
| 02 generation | 0 | 8.05 |
| 03 profiles | 0 | 11.69 |
| 04 topology | 0 | 26.84 |
| 05 assembly | 0 | 22.16 |
| 06 full year | 0 | 614.55 |
| 06a diagnostics | 0 | 49.38 |

The full-year run recorded solver status `ok`, termination `optimal`, and
8,760 hourly snapshots. HiGHS reported 467.92 seconds of solver time. The
solved network contains 94,056,790 bytes with SHA-256
`7d118f8001bc2596b2d326b0addc8bc34efee8bc76777b8efeaa36093824ef2a`.
All 17 diagnostic figures were generated, with PNG and SVG evidence, and 06a
recorded the same network checksum and `solver_invoked: false`.

The full-sequence execution report records `status: PASS` and
`pipeline_files_unchanged: true`. The final acceptance helper also checks solver
success, annual diagnostic coverage, absence of skipped figures, and agreement
between the actual solved-network digest, artifact metadata and every diagnostic
manifest row. Its final verification reruns only the read-only 06a notebook;
no second annual optimization is requested.
That verification completed successfully under `final-verification/`, with
`status: PASS`, all 17 figures, matching digests, and unchanged pipeline hashes.
The installed VS Code extensions include both `ms-python.python` and
`ms-toolsai.jupyter` (with its renderer and keymap extensions).

The statistical result retains the existing `FAIL` screen and `PARTIAL`
comparability status: nonconventional share remains outside the configured
screen and the 931.43 GWh metering-boundary residual remains unresolved. The
rerun reports zero load shedding and a numerical energy-balance residual of
approximately 2.15e-11 GWh. These are recorded execution observations; the
environment test neither changes nor establishes the scientific acceptance gate.

Checks intentionally not performed: legacy/upstream weather-regeneration
workflows, Linux/macOS environment reproduction, scenario batches and a second
annual solve. They are outside the requested active-pipeline environment scope.
No packages in other existing environments were upgraded. No notebook, source,
manifest, validation threshold or source edition was changed to make execution
pass. The three existing governance test failures remain owner follow-up items.

The owner explicitly requested integration after successful execution. Only
the seven environment specification, lock, installer, acceptance helper,
test and documentation files are included in that integration. No push is
requested or performed.

## Specialist assistance

The read-only `research` specialist audited active dependencies and historical
version evidence. The `developer` specialist implemented and parser-checked the
PowerShell installer. The parent reviewed the implementation and owns environment
installation, execution evidence, testing and Git integration. Specialists did not
install packages, run scientific calculations or make commits.

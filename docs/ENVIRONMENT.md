# Approved execution environment

## Policy

The project uses the existing Conda environment `pypsa-earth-ec`. Do not create a
repository-local environment, `.venv`, temporary replacement environment, or a
second Conda environment. Do not install or upgrade packages without owner
approval.

`config/environment_ec.yaml` is an auditable environment specification, not a
guarantee that a newly solved environment will reproduce the accepted runtime.
Phase 2 corrected its invalid entries and reconciled the packages needed for
preflight against the working environment without installing anything.

## Tested Phase 2 runtime

| Package | Observed version | Phase 2 requirement |
|---|---:|---|
| Python | 3.11.14 | `>=3.10,<3.12` |
| PyPSA | 0.30.3 | `>=0.25,<=0.30.3` |
| NumPy | 1.26.4 | `<2` |
| pandas | 2.3.3 | `>=2` |
| xarray | 2025.1.2 | `>=2023.11.0,<=2025.1.2` |
| h5netcdf | 1.7.3 | `>=1` |
| PyYAML | 6.0.3 | `>=6` |
| requests | 2.32.5 | `>=2` |
| jsonschema | 4.25.1 | `>=4` |
| nbformat | 5.10.4 | `>=5` |
| nbclient | 0.10.2 | `>=0.10` |

The environment also contains atlite 0.4.1, GeoPandas 1.1.1, HiGHS bindings
1.11.0, matplotlib 3.10.7, NetCDF4 1.7.3, OpenPyXL 3.1.5,
powerplantmatching 0.7.1, SciPy 1.15.2, and Plotly 6.3.1. These versions are
observations for reproducibility, not permission to update the environment.

## Specification corrections

Phase 2 changed only the YAML specification:

- corrected `ploty=6.3.1` to a valid Plotly constraint;
- removed the empty dependency entry;
- made the preflight/runtime dependencies explicit: h5netcdf, PyYAML, requests,
  jsonschema, nbclient, and nbformat.

No package was installed, removed, or upgraded.

## Verification

Run:

```powershell
python -m src.preflight --stage 00_preflight --offline
```

The command fails before artifact retrieval if the interpreter is not from
`pypsa-earth-ec`, the Python version is outside the approved range, or a required
preflight package is missing/incompatible.

## Notebook execution

The reference workstation does not have a persistent `pypsa-earth-ec` Jupyter
kernelspec. A generic `python3` kernelspec may resolve to a different interpreter.
Therefore command-line notebook execution uses:

```powershell
python -m src.notebook_runner <notebook-path>
```

The runner first validates the environment, creates an isolated temporary
kernelspec pointing to the current interpreter, executes the notebook from the
repository root, and removes the temporary kernel information. An optional
`--output` writes an executed copy without modifying the source notebook.

GIS/weather imports previously emitted GDAL/ecCodes discovery warnings. Phase 2
NetCDF structural validation uses h5netcdf directly and does not trigger those
plugins. Their configuration remains relevant for later GIS or weather-profile
processing and must be addressed in the phase that exercises those operations.

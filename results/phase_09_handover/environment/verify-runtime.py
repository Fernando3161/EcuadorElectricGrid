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

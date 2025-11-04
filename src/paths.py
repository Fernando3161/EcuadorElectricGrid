import os
from typing import Dict, List


# Resolve repository root as the parent of this file's directory (which is `src`)
ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Top-level directories
CONFIG_DIR: str = os.path.join(ROOT, "config")
DATA_DIR: str = os.path.join(ROOT, "data")
LITERATURE_DIR: str = os.path.join(ROOT, "literature")
NOTEBOOKS_DIR: str = os.path.join(ROOT, "notebooks")
RESULTS_DIR: str = os.path.join(ROOT, "results")
GRAPHS_DIR: str = os.path.join(RESULTS_DIR, "graphs")
SRC_DIR: str = os.path.join(ROOT, "src")

# Known subdirectories under data
DATA_PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")
DATA_RAW_DIR: str = os.path.join(DATA_DIR, "raw")

# Known subdirectories under data/raw
RAW_CUTOUTS_DIR: str = os.path.join(DATA_RAW_DIR, "cutouts")
RAW_NETWORKS_DIR: str = os.path.join(DATA_RAW_DIR, "networks")
RAW_DEMANDS_DIR: str = os.path.join(DATA_RAW_DIR, "demand")
RAW_GADM_DIR : str = os.path.join(DATA_RAW_DIR, "gadm")
RAW_GENERATION_DIR : str = os.path.join(DATA_RAW_DIR, "generation")


# Known subdirectories under data/processed
PROC_GENERATION_DIR : str = os.path.join(DATA_PROCESSED_DIR, "generation")
PROC_LOAD_DIR : str = os.path.join(DATA_PROCESSED_DIR, "scaled_loads")

# Common directory names to ignore during traversal
IGNORE_DIRS = {".git", "__pycache__"}


def list_subdirs(base_dir: str) -> List[str]:
    """Return absolute paths of immediate subdirectories in base_dir."""
    try:
        return [
            os.path.join(base_dir, name)
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name)) and name not in IGNORE_DIRS
        ]
    except FileNotFoundError:
        return []


def all_dirs() -> Dict[str, str]:
    """
    Return a mapping of directory keys to absolute paths for all folders
    (and subfolders) under ROOT. Keys are repository-relative paths using
    forward slashes, with the empty string mapped to ROOT itself.
    """
    mapping: Dict[str, str] = {"": ROOT}
    for dirpath, dirnames, _ in os.walk(ROOT, topdown=True):
        # prune ignored directories from traversal
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        rel = os.path.relpath(dirpath, ROOT)
        if rel != "." and any(part in IGNORE_DIRS for part in rel.split(os.sep)):
            # Skip any path inside ignored dirs (defensive)
            continue

        key = "" if rel == "." else rel.replace(os.sep, "/")
        mapping[key] = dirpath
        # Ensure subdirectories are captured as well during traversal
        for d in dirnames:
            if d in IGNORE_DIRS:
                continue
            sub_abs = os.path.join(dirpath, d)
            sub_rel = os.path.relpath(sub_abs, ROOT).replace(os.sep, "/")
            mapping[sub_rel] = sub_abs
    return mapping


def data_dirs() -> Dict[str, str]:
    """
    Return a mapping of all directories under DATA_DIR.
    Keys are paths relative to `data` (e.g., "raw/cutouts"),
    with the empty string mapped to DATA_DIR itself.
    Excludes entries under ignored directories like __pycache__.
    """
    mapping: Dict[str, str] = {"": DATA_DIR}
    for dirpath, dirnames, _ in os.walk(DATA_DIR, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        rel = os.path.relpath(dirpath, DATA_DIR)
        if rel != "." and any(part in IGNORE_DIRS for part in rel.split(os.sep)):
            continue
        key = "" if rel == "." else rel.replace(os.sep, "/")
        mapping[key] = dirpath
        for d in dirnames:
            if d in IGNORE_DIRS:
                continue
            sub_abs = os.path.join(dirpath, d)
            sub_rel = os.path.relpath(sub_abs, DATA_DIR).replace(os.sep, "/")
            mapping[sub_rel] = sub_abs
    return mapping


__all__ = [
    "ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "LITERATURE_DIR",
    "NOTEBOOKS_DIR",
    "RESULTS_DIR",
    "GRAPHS_DIR",
    "SRC_DIR",
    "DATA_PROCESSED_DIR",
    "DATA_RAW_DIR",
    "RAW_CUTOUTS_DIR",
    "RAW_NETWORKS_DIR",
    "RAW_DEMANDS_DIR",
    "RAW_GADM_DIR",
    "RAW_GENERATION_DIR",
    "PROC_GENERATION_DIR",
    "PROC_LOAD_DIR",
    "list_subdirs",
    "all_dirs",
    "data_dirs",
]

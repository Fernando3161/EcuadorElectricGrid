"""Execute notebooks with the current approved interpreter, without installation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from .artifacts import ROOT
from .environment import check_environment, format_environment_report


TEMP_KERNEL_NAME = "ecuador-grid-active-environment"


def _write_temporary_kernelspec(root: Path) -> None:
    kernel_dir = root / "kernels" / TEMP_KERNEL_NAME
    kernel_dir.mkdir(parents=True)
    specification = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "EcuadorElectricGrid active environment",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(
        json.dumps(specification, indent=2), encoding="utf-8"
    )


def execute_notebook(
    notebook_path: Path | str,
    *,
    output_path: Path | str | None = None,
    working_directory: Path | str = ROOT,
    timeout: int = -1,
) -> nbformat.NotebookNode:
    """Execute through ``sys.executable``; -1 disables the cell execution limit."""

    environment = check_environment()
    if not environment.ok:
        raise RuntimeError(format_environment_report(environment))

    source = Path(notebook_path).resolve()
    workdir = Path(working_directory).resolve()
    notebook = nbformat.read(source, as_version=4)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    previous_jupyter_path = os.environ.get("JUPYTER_PATH")
    with tempfile.TemporaryDirectory(prefix="ecuador-grid-kernel-") as temporary:
        kernel_root = Path(temporary)
        _write_temporary_kernelspec(kernel_root)
        path_parts = [str(kernel_root)]
        if previous_jupyter_path:
            path_parts.append(previous_jupyter_path)
        os.environ["JUPYTER_PATH"] = os.pathsep.join(path_parts)
        try:
            client = NotebookClient(
                notebook,
                timeout=timeout,
                kernel_name=TEMP_KERNEL_NAME,
                resources={"metadata": {"path": str(workdir)}},
                allow_errors=False,
            )
            executed = client.execute()
        finally:
            if previous_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous_jupyter_path

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(executed, destination)
    return executed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute a notebook using the currently active approved environment."
    )
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--working-directory", type=Path, default=ROOT)
    parser.add_argument(
        "--timeout", type=int, default=-1,
        help="Optional per-cell time limit in seconds; default -1 means unlimited.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execute_notebook(
        args.notebook,
        output_path=args.output,
        working_directory=args.working_directory,
        timeout=args.timeout,
    )
    print(f"Notebook executed successfully: {args.notebook}")
    if args.output:
        print(f"Executed copy: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

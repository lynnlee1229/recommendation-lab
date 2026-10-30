"""Execute saved tutorials top-to-bottom using clean, project-scoped kernels."""

from pathlib import Path
import argparse
import json
import os
import time
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="+", help="Exact notebook stems to execute; defaults to all"
    )
    args = parser.parse_args()
    for key, relative in {
        "IPYTHONDIR": "logs/ipython",
        "JUPYTER_DATA_DIR": ".venv/share/jupyter",
        "JUPYTER_RUNTIME_DIR": "logs/jupyter-runtime",
        "MPLCONFIGDIR": "logs/matplotlib",
    }.items():
        p = ROOT / relative
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    output = ROOT / "artifacts/notebook-previews"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if args.only:
        selected = set(args.only)
        if selected - set(p.stem for p in paths):
            parser.error("Unknown notebook stem")
        paths = [p for p in paths if p.stem in selected]
    receipt_path = output / "execution.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for path in paths:
        started = time.monotonic()
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            kernel_name="reclab",
            timeout=180,
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.validate(nb)
        nbformat.write(nb, path)
        html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
        (output / (path.stem + ".html")).write_text(html)
        cells = [cell for cell in nb.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in cells)
        assert not any(
            out.output_type == "error" for cell in cells for out in cell.outputs
        )
        print(
            f"PASS {path.name}: {len(cells)} code cells; {time.monotonic() - started:.1f}s",
            flush=True,
        )
        receipt[path.name] = {
            "state": "executed",
            "code_cells": len(cells),
            "seconds": round(time.monotonic() - started, 2),
        }
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        curriculum_path = ROOT / "docs/curriculum.json"
        curriculum = json.loads(curriculum_path.read_text())
        for lesson in curriculum["lessons"]:
            if lesson.get("notebook", {}).get("file") == path.name:
                lesson["notebook"]["state"] = "executed"
        curriculum_path.write_text(json.dumps(curriculum, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

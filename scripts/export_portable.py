"""Build a checksummed, credential-free study release; never uploads or pushes Git."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts.restore_portable import digest, safe_path, verify_packets

BUDGET = 96 * 1024**2


def descendants(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and not p.is_symlink()
                  and "__pycache__" not in p.parts and ".ipynb_checkpoints" not in p.parts)


def select_payload(root=ROOT):
    """Explicit roots, never a recursive copy of the user's machine or runtime."""
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    payload = {name: root / name for name in tracked if name}
    for name in payload:
        safe_path(name)
        if name.split('/')[0] in {".venv", ".git", "logs", "data", "artifacts", ".codex"}:
            raise ValueError(f"Tracked runtime/config file requires explicit review: {name}")
    for relative in ("data/processed", "data/xmrec/raw", "artifacts/baselines", "artifacts/configs", "artifacts/traces"):
        for path in descendants(root / relative):
            payload[path.relative_to(root).as_posix()] = path
    payload["data/raw/retailrocket.zip"] = root / "data/raw/retailrocket.zip"
    for folder in sorted((root / "artifacts/runs").iterdir()):
        if (folder / "report.json").is_file() and (folder / "model.pt").is_file():
            for path in descendants(folder):
                if path.suffix in {".pt", ".json", ".npy", ".npz"}:
                    payload[path.relative_to(root).as_posix()] = path
    for name in ("final-test.json", "acceptance.json"):
        if (root / "artifacts" / name).is_file():
            payload["artifacts/" + name] = root / "artifacts" / name
    latest = json.loads((root / "artifacts/xmrec/latest.json").read_text())["run_id"]
    run = root / "artifacts/xmrec" / latest
    if Path(latest).name != latest or json.loads((run / "receipt.json").read_text()).get("status") != "PASS":
        raise ValueError("Latest XMRec run is not a completed, successful run")
    for folder in (run, root / "artifacts/xmrec/experiments"):
        for path in descendants(folder):
            payload[path.relative_to(root).as_posix()] = path
    payload["artifacts/xmrec/latest.json"] = root / "artifacts/xmrec/latest.json"
    sources = json.loads((root / "docs/xmrec_sources.json").read_text())
    for relative in sources["source_hashes"]:
        path = root / "data/xmrec/upstream/winner" / relative
        if digest(path) != sources["source_hashes"][relative]:
            raise ValueError("Original competition reference source changed")
        payload[path.relative_to(root).as_posix()] = path
    original_readme = root / "data/xmrec/upstream/winner/README.md"
    if original_readme.is_file():
        payload[original_readme.relative_to(root).as_posix()] = original_readme
    curriculum = json.loads((root / "docs/curriculum.json").read_text())
    slides = Path(curriculum["local_source"])
    if not slides.is_absolute():
        slides = root / slides
    for lesson in curriculum["lessons"]:
        name = lesson["file"]
        if Path(name).name != name:
            raise ValueError("Unexpected lecture filename")
        payload["resources/slides/" + name] = slides / name
    curriculum["local_source"] = "resources/slides"
    overrides = {"docs/curriculum.json": json.dumps(curriculum, ensure_ascii=False, indent=2).encode()}
    for name, path in payload.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular payload: {name}")
    return payload, overrides


def build_release(payload, overrides, output, revision, budget=BUDGET):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new release directory; releases are never overwritten")
    output.mkdir(parents=True)
    files = []
    for name, path in sorted(payload.items()):
        safe_path(name)
        if name in overrides:
            content = overrides[name]
            size, sha = len(content), hashlib.sha256(content).hexdigest()
        else:
            size, sha = path.stat().st_size, digest(path)
        files.append({"path": name, "bytes": size, "sha256": sha,
                      "mode": 0o755 if name.endswith(".sh") else 0o644})
    packets = []
    normal = [row for row in files if row["bytes"] <= budget]
    oversized = [row for row in files if row["bytes"] > budget]
    chunks = []
    current, total = [], 0
    for row in normal:
        if current and total + row["bytes"] > budget:
            chunks.append(current)
            current, total = [], 0
        current.append(row)
        total += row["bytes"]
    if current:
        chunks.append(current)
    for index, chunk in enumerate(chunks, 1):
        name = f"content-{index:02}.zip"
        path = output / name
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for row in chunk:
                relative = row["path"]
                if relative in overrides:
                    archive.writestr(relative, overrides[relative])
                else:
                    archive.write(payload[relative], relative)
        packets.append({"name": name, "kind": "zip", "bytes": path.stat().st_size, "sha256": digest(path)})
    for index, row in enumerate(oversized, 1):
        # Original Retailrocket is already a ZIP; transfer as a file, not nested archive entries.
        name = f"input-{index:02}{payload[row['path']].suffix}"
        path = output / name
        shutil.copyfile(payload[row["path"]], path)
        packets.append({"name": name, "kind": "file", "destination": row["path"],
                        "bytes": path.stat().st_size, "sha256": digest(path)})
    spec = {"format": "local-study-release-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": revision, "target_profile": "macOS14plus-arm64-16GiB-or-more",
            "portable_adjustments": ["docs/curriculum.json local_source=resources/slides"],
            "excluded": ["virtual environments", "runtime logs", "Jupyter sessions and credentials",
                         "failed XMRec runs", "local Git configuration", "unrelated lab"],
            "files": files, "packets": packets}
    (output / "manifest.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    verify_packets(output, spec)
    return spec


def export(output):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise RuntimeError("Commit/review current in-scope changes before freezing a release")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload, overrides = select_payload()
    spec = build_release(payload, overrides, output, revision)
    output = Path(output)
    shutil.copyfile(ROOT / "scripts/restore_portable.py", output / "restore.py")
    shutil.copyfile(ROOT / "docs/RESTORE_ON_ANOTHER_MAC.md", output / "START_HERE.md")
    shutil.copyfile(ROOT / "docs/RESTORE_PROMPT.md", output / "RESTORE_PROMPT.md")
    paths = [output / p["name"] for p in spec["packets"]]
    paths += [output / name for name in ["manifest.json", "restore.py", "START_HERE.md", "RESTORE_PROMPT.md"]]
    (output / "SHA256SUMS").write_text(''.join(f'{digest(p)}  {p.name}\n' for p in paths))
    print(json.dumps({"state": "PASS", "release": str(output.resolve()), "packets": len(spec["packets"]),
                      "project_files": len(spec["files"]), "total_packet_mib": sum(p["bytes"] for p in spec["packets"])/1024**2,
                      "uploaded": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.output)

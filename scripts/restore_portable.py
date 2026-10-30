"""Standalone recovery entry. Python standard library only; no external Git required."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import zipfile

UV_VERSION = "0.12.5"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(name):
    value = PurePosixPath(name)
    if not name or not value.parts or value.is_absolute() or ".." in value.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    if name != value.as_posix() or name.endswith("/"):
        raise ValueError(f"Noncanonical archive path: {name}")
    return value


def load_manifest(folder):
    spec = json.loads((Path(folder) / "manifest.json").read_text())
    if spec.get("format") != "local-study-release-v1":
        raise ValueError("Unknown recovery format")
    paths = [row["path"] for row in spec["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate project paths in manifest")
    for name in paths:
        safe_path(name)
    packet_names = [p["name"] for p in spec["packets"]]
    if len(packet_names) != len(set(packet_names)):
        raise ValueError("Duplicate packet names")
    for name in packet_names:
        if len(safe_path(name).parts) != 1:
            raise ValueError("Packets must be in the release folder")
    return spec


def verify_packets(folder, spec=None):
    folder = Path(folder)
    spec = spec or load_manifest(folder)
    declared = {row["path"]: row for row in spec["files"]}
    seen = set()
    for packet in spec["packets"]:
        path = folder / packet["name"]
        if path.stat().st_size != packet["bytes"] or digest(path) != packet["sha256"]:
            raise ValueError(f"Packet checksum mismatch: {path.name}")
        if packet["kind"] == "file":
            name = packet["destination"]
            if name not in declared or name in seen:
                raise ValueError("Undeclared or duplicate raw payload")
            row = declared[name]
            if path.stat().st_size != row["bytes"] or packet["sha256"] != row["sha256"]:
                raise ValueError("Raw payload manifest mismatch")
            seen.add(name)
        elif packet["kind"] == "zip":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    name = info.filename
                    safe_path(name)
                    mode = info.external_attr >> 16
                    if info.is_dir() or stat.S_ISLNK(mode):
                        raise ValueError("Archive directories and symlinks are not permitted")
                    if name not in declared or name in seen:
                        raise ValueError(f"Undeclared or duplicate archive member: {name}")
                    if info.file_size != declared[name]["bytes"]:
                        raise ValueError(f"Archive member size mismatch: {name}")
                    # Validate content before any project directory or executable is created.
                    h = hashlib.sha256()
                    with archive.open(info) as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            h.update(chunk)
                    if h.hexdigest() != declared[name]["sha256"]:
                        raise ValueError(f"Archive member checksum mismatch: {name}")
                    seen.add(name)
        else:
            raise ValueError("Unknown packet kind")
    if seen != set(declared):
        raise ValueError("Release is missing project files")
    return spec


def verify_project(destination, spec):
    destination = Path(destination)
    for row in spec["files"]:
        p = destination.joinpath(*safe_path(row["path"]).parts)
        if p.is_symlink() or not p.is_file() or digest(p) != row["sha256"]:
            raise ValueError(f"Restored file changed or missing: {row['path']}")
    return {"state": "PASS", "verified_project_files": len(spec["files"])}


def restore(folder, destination):
    folder, destination = Path(folder).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a NEW destination; existing folders are never overwritten")
    spec = verify_packets(folder)
    destination.mkdir(parents=True)
    for packet in spec["packets"]:
        if packet["kind"] == "file":
            out = destination.joinpath(*safe_path(packet["destination"]).parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / packet["name"], out)
        else:
            with zipfile.ZipFile(folder / packet["name"]) as archive:
                for info in archive.infolist():
                    out = destination.joinpath(*safe_path(info.filename).parts)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as inp, out.open("xb") as stream:
                        shutil.copyfileobj(inp, stream)
    for row in spec["files"]:
        destination.joinpath(*safe_path(row["path"]).parts).chmod(row.get("mode", 0o644))
    result = verify_project(destination, spec)
    result.update({"source_revision": spec["source_revision"], "target": str(destination),
                   "environment": "NOT_INSTALLED", "target_profile": spec["target_profile"]})
    (destination / ".portable-receipt.json").write_text(json.dumps(result, indent=2))
    return result


def require_managed_target(destination):
    destination = Path(destination).resolve()
    if not (destination / ".portable-receipt.json").is_file():
        raise ValueError("Setup is allowed only in a newly restored project")
    if (destination / ".venv").is_symlink() or (destination / ".restore-tools").is_symlink():
        raise ValueError("Refusing redirected environment folders")
    return destination


def check_platform():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("This release targets Apple Silicon macOS; do not force it onto another platform")
    if int(platform.mac_ver()[0].split(".")[0]) < 14:
        raise RuntimeError("The locked PyTorch wheel requires macOS 14 or newer; no OS upgrade was attempted")
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    if memory < 16 * 1024**3:
        raise RuntimeError("This release is validated for at least 16 GiB RAM")
    return {"os": platform.platform(), "architecture": platform.machine(),
            "physical_memory_gib": memory / 1024**3}


def setup(destination):
    destination = require_managed_target(destination)
    machine = check_platform()
    omp = Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")
    if not omp.is_file():
        raise RuntimeError("Missing libomp. With your existing Homebrew run: brew install libomp; then retry setup")
    uv = shutil.which("uv")
    if uv is None:
        bootstrap = destination / ".restore-tools"
        subprocess.run([sys.executable, "-m", "venv", str(bootstrap)], check=True)
        bootstrap_python = str(bootstrap / "bin/python")
        subprocess.run([bootstrap_python, "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
                        f"uv=={UV_VERSION}"], check=True)
        uv = str(bootstrap / "bin/uv")
    env = dict(os.environ)
    env.update({"UV_PROJECT_ENVIRONMENT": str(destination / ".venv"),
                "UV_PYTHON_DOWNLOADS": "automatic", "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    subprocess.run([uv, "sync", "--frozen", "--group", "dev", "--group", "notebooks"],
                   cwd=destination, env=env, check=True)
    python = str(destination / ".venv/bin/python")
    subprocess.run([python, "-m", "ipykernel", "install", "--prefix", str(destination / ".venv"),
                    "--name", "reclab", "--display-name", "Recommendation Lab (Python 3.12)"],
                   cwd=destination, env=env, check=True)
    receipt_path = destination / ".portable-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update({"environment": "INSTALLED", "machine": machine,
                    "python": subprocess.check_output([python, "--version"], text=True).strip(),
                    "uv": subprocess.check_output([uv, "--version"], text=True).strip(),
                    "validation": "NOT_RUN"})
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return receipt


def validate(destination, notebooks=False):
    destination = require_managed_target(destination)
    python = str(destination / ".venv/bin/python")
    env = dict(os.environ)
    for key, folder in {"JUPYTER_DATA_DIR": ".venv/share/jupyter", "JUPYTER_RUNTIME_DIR": "logs/jupyter-runtime",
                        "IPYTHONDIR": "logs/ipython", "MPLCONFIGDIR": "logs/matplotlib"}.items():
        env[key] = str(destination / folder)
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    subprocess.run([python, "-m", "compileall", "-q", "reclab", "scripts", "tests"],
                   cwd=destination, env=env, check=True)
    subprocess.run([python, "-m", "pytest", "-q"], cwd=destination, env=env, check=True)
    if notebooks:
        subprocess.run([python, "scripts/execute_notebooks.py", "--only", "11_XMRec_Start",
                        "12_XMRec_Features", "13_XMRec_Graph", "14_XMRec_Ranking", "15_XMRec_Evaluation"],
                       cwd=destination, env=env, check=True)
    p = destination / ".portable-receipt.json"
    receipt = json.loads(p.read_text())
    receipt.update({"validation": "PASS", "competition_notebooks": "PASS" if notebooks else "NOT_RUN",
                    "services_started": False})
    p.write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "restore", "setup", "validate"])
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--notebooks", action="store_true")
    args = parser.parse_args()
    if args.action == "verify":
        spec = verify_packets(args.bundle)
        result = {"state": "PASS", "packets": len(spec["packets"]), "project_files": len(spec["files"])}
    else:
        if args.destination is None:
            parser.error("--destination is required")
        if args.action == "restore":
            result = restore(args.bundle, args.destination)
        elif args.action == "setup":
            result = setup(args.destination)
        else:
            result = validate(args.destination, args.notebooks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

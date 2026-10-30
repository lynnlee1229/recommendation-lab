"""Idempotent local bootstrap; never starts or alters the lakehouse lab."""

from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reclab.data import SOURCE, digest
from reclab.paths import read_json


def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "retailrocket.zip"
    if not archive.exists():
        partial = raw / "retailrocket.zip.part"
        print("Downloading original Retailrocket archive (~291 MiB).", flush=True)
        request = urllib.request.Request(
            SOURCE, headers={"User-Agent": "search-recommendation-ads-lab/0.1"}
        )
        with (
            urllib.request.urlopen(request, timeout=60) as source,
            partial.open("wb") as out,
        ):
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
        if not zipfile.is_zipfile(partial):
            raise ValueError(
                "Download was not a ZIP; no credentials or browser fallback attempted."
            )
        partial.replace(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError("Invalid existing archive; preserved for inspection.")
    manifest = ROOT / "data/processed/manifest.json"
    if manifest.exists() and read_json(manifest)["archive_sha256"] != digest(archive):
        raise ValueError(
            "Archive differs from the frozen dataset manifest; preserve both and prepare a separate snapshot."
        )

    def run(*args):
        subprocess.run(
            [sys.executable, "-m", "reclab.cli", *args], cwd=ROOT, check=True
        )

    if not (ROOT / "data/processed/manifest.json").exists():
        run("prepare", str(archive))
    if not (ROOT / "artifacts/baselines/metrics.json").exists():
        run("baseline")
    for config, run_id in [
        ("history32", "baseline-history32"),
        ("no-history", "ablation-no-history"),
        ("logq", "ablation-logq"),
        ("history64", "ablation-history64"),
    ]:
        folder = ROOT / "artifacts/runs" / run_id
        if (folder / "report.json").exists():
            continue
        if folder.exists():
            raise RuntimeError(
                f"Incomplete run preserved at {folder}; choose a new run ID rather than overwriting it."
            )
        run(
            "train",
            "--config",
            str(ROOT / "configs" / (config + ".json")),
            "--run-id",
            run_id,
        )
    print("Ready. Run ./lab.sh up then open http://127.0.0.1:18580")


if __name__ == "__main__":
    main()

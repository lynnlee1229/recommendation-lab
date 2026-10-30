from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:18580"
PID = ROOT / "logs/server.pid"


def health():
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=2) as r:
            return json.load(r).get("project") == str(ROOT)
    except Exception:
        return False


def owned_pid():
    try:
        pid = int(PID.read_text())
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
        )
        expected = str(ROOT / ".venv/bin/python") + " -m reclab.cli serve"
        return pid if expected in result.stdout else None
    except (OSError, ValueError):
        return None


def main():
    action = sys.argv[1]
    if action == "status":
        print(json.dumps({"ready": health(), "owned_pid": owned_pid(), "url": URL}))
        return
    if action == "stop":
        pid = owned_pid()
        if not pid:
            print("No owned server process to stop.")
            return
        try:
            with urllib.request.urlopen(URL + "/api/job", timeout=2) as r:
                job = json.load(r)
            if job.get("state") == "running":
                raise RuntimeError(
                    "Training is active. Wait for completion before stopping the workbench."
                )
        except (OSError, ValueError):
            pass
        os.kill(pid, signal.SIGTERM)
        for _ in range(80):
            if owned_pid() is None:
                PID.unlink(missing_ok=True)
                print("Stopped this workbench. Data and model versions preserved.")
                return
            time.sleep(0.1)
        raise RuntimeError(
            "Shutdown is still pending; PID file preserved for inspection."
        )
    if health():
        print(URL)
        return
    if owned_pid():
        raise RuntimeError(
            "Owned process exists but is unhealthy; inspect logs/server.log before restarting."
        )
    if not (ROOT / "data/processed/manifest.json").exists():
        raise RuntimeError("Run ./lab.sh setup first.")
    PID.parent.mkdir(exist_ok=True)
    with (ROOT / "logs/server.log").open("a") as log:
        proc = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), "-m", "reclab.cli", "serve"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID.write_text(str(proc.pid))
    for _ in range(80):
        if health():
            print(URL)
            return
        if proc.poll() is not None:
            raise RuntimeError("Server exited; inspect logs/server.log")
        time.sleep(0.25)
    raise RuntimeError(
        "Startup timed out; inspect logs/server.log. Process left for inspection."
    )


if __name__ == "__main__":
    main()

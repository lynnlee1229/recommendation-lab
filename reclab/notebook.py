"""Project-scoped, token-authenticated JupyterLab. Never print authentication URLs."""

from pathlib import Path
from urllib.parse import urlencode
import importlib.util
import json
import os
import secrets
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from .paths import ROOT

PORT = 18581
BASE_URL = f"http://127.0.0.1:{PORT}"
RUNTIME = ROOT / "logs/jupyter-runtime"
PID_FILE = ROOT / "logs/notebook.pid"
PYTHON = ROOT / ".venv/bin/python"
TARGETS = {
    "ads_curriculum_code": "reclab/ads_curriculum.py",
    "ads_47": "notebooks/47_Ads_Products.ipynb",
    "ads_48": "notebooks/48_Ads_Billing.ipynb",
    "ads_49": "notebooks/49_Ads_Contracts.ipynb",
    "ads_50": "notebooks/50_Ads_SponsoredSearch.ipynb",
    "ads_51": "notebooks/51_Ads_MultiSlot.ipynb",
    "ads_52": "notebooks/52_Ads_RTB.ipynb",
    "ads_53": "notebooks/53_Ads_Targeting.ipynb",
    "ads_54": "notebooks/54_Ads_CandidateIndex.ipynb",
    "ads_55": "notebooks/55_Ads_FM.ipynb",
    "ads_56": "notebooks/56_Ads_Conversion.ipynb",
    "ads_57": "notebooks/57_Ads_Bidding.ipynb",
    "ads_58": "notebooks/58_Ads_FeedbackPacing.ipynb",
    "ads_59": "notebooks/59_Ads_Attribution.ipynb",
    "ads_60": "notebooks/60_Ads_ReservePrices.ipynb",
    "ads_61": "notebooks/61_Ads_ExperimentIntegrity.ipynb",
    "ads_62": "notebooks/62_Ads_Ledger.ipynb",

    "search_35": "notebooks/35_Search_Needs_Judgments.ipynb",
    "search_36": "notebooks/36_Search_Chinese_Tokenization.ipynb",
    "search_37": "notebooks/37_Search_Term_Fields.ipynb",
    "search_38": "notebooks/38_Search_Intent.ipynb",
    "search_39": "notebooks/39_Search_Rewrite.ipynb",
    "search_40": "notebooks/40_Search_Positional_Index.ipynb",
    "search_41": "notebooks/41_Search_Candidate_Cache.ipynb",
    "search_42": "notebooks/42_Search_BERT_Finetune.ipynb",
    "search_43": "notebooks/43_Search_Ranking_Objectives.ipynb",
    "search_44": "notebooks/44_Search_Quality_Diversity.ipynb",
    "search_45": "notebooks/45_Search_Query_Suggestions.ipynb",
    "search_46": "notebooks/46_Search_Trace_Review.ipynb",

    "common_34": "notebooks/34_Common_Scores_Evaluation.ipynb",
    "common_33": "notebooks/33_Common_Optimization.ipynb",
    "common_32": "notebooks/32_Common_Gradients.ipynb",
    "common_31": "notebooks/31_Common_Tensors.ipynb",
    "search_lessons_code": "reclab/search_lessons.py",
    "search_task_qrels": "notebooks/25_Search_Task_Qrels.ipynb",
    "search_bm25": "notebooks/26_Search_BM25.ipynb",
    "search_semantic": "notebooks/27_Search_Semantic.ipynb",
    "search_ann": "notebooks/28_Search_ANN.ipynb",
    "search_fusion_ranking": "notebooks/29_Search_Fusion_Ranking.ipynb",
    "search_errors_service": "notebooks/30_Search_Errors_Service.ipynb",

    "ads_19": "notebooks/19_Ads_Data_Splits.ipynb",
    "ads_20": "notebooks/20_Ads_Linear_Features.ipynb",
    "ads_21": "notebooks/21_Ads_Nonlinearity.ipynb",
    "ads_22": "notebooks/22_Ads_Calibration.ipynb",
    "ads_23": "notebooks/23_Ads_Auction_Bids.ipynb",
    "ads_24": "notebooks/24_Ads_Pacing_Integration.ipynb",

    "ads": "notebooks/18_Ads_Experiments.ipynb",
    "ads_code": "reclab/ads.py",
    "search": "notebooks/17_Search_Experiments.ipynb",
    "search_code": "reclab/search.py",
    "deep_recommendation": "notebooks/16_Recommendation_Experiments.ipynb",
    "course": "notebooks/00_Course_Map.ipynb",
    "xmrec_start": "notebooks/11_XMRec_Start.ipynb",
    "xmrec_features": "notebooks/12_XMRec_Features.ipynb",
    "xmrec_graph": "notebooks/13_XMRec_Graph.ipynb",
    "xmrec_ranking": "notebooks/14_XMRec_Ranking.ipynb",
    "xmrec_evaluation": "notebooks/15_XMRec_Evaluation.ipynb",
    "xmrec_code_data": "reclab/competition/data.py",
    "xmrec_code_memory": "reclab/competition/memory.py",
    "xmrec_code_graph": "reclab/competition/graph.py",
    "xmrec_code_ranking": "reclab/competition/ranking.py",
    "xmrec_code_experiment": "reclab/competition/experiment.py",
    "xmrec_code_metrics": "reclab/competition/metrics.py",

    "basics": "notebooks/03_Foundations_AB.ipynb",
    "retrieval_course": "notebooks/04_Retrieval_Methods.ipynb",
    "ranking": "notebooks/05_Ranking_Multitask.ipynb",
    "cross": "notebooks/06_Feature_Crossing.ipynb",
    "sequence": "notebooks/07_Sequence_DIN_SIM.ipynb",
    "diversity": "notebooks/08_Diversity_DPP.ipynb",
    "cold": "notebooks/09_ColdStart_Traffic.ipynb",
    "engineering": "notebooks/10_Engineering_Improvement.ipynb",
    "data": "notebooks/01_Data_Features_ItemCF.ipynb",
    "tower": "notebooks/02_TwoTower_HandsOn.ipynb",
    "models": "reclab/models.py",
    "training": "reclab/train.py",
    "sampling": "reclab/data.py",
    "retrieval": "reclab/retrieval.py",
    "lab": "",
}
_start_lock = threading.Lock()


def owned_pid():
    try:
        pid = int(PID_FILE.read_text())
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True
        )
        expected = f"{PYTHON} -m jupyterlab"
        return pid if expected in result.stdout and str(ROOT) in result.stdout else None
    except (OSError, ValueError):
        return None


def server_info():
    pid = owned_pid()
    if not pid:
        return None
    info_file = RUNTIME / f"jpserver-{pid}.json"
    try:
        info = json.loads(info_file.read_text())
        if info.get("pid") != pid or info.get("port") != PORT:
            return None
        if Path(info.get("root_dir", "")).resolve() != ROOT or not info.get("token"):
            return None
        req = urllib.request.Request(
            BASE_URL + "/api/status",
            headers={"Authorization": "token " + info["token"]},
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                return info
    except (OSError, ValueError):
        return None
    return None


def ensure_server():
    with _start_lock:
        info = server_info()
        if info:
            return info
        if owned_pid():
            raise RuntimeError(
                "JupyterLab process exists but is not ready; inspect the project notebook log."
            )
        if importlib.util.find_spec("jupyterlab") is None:
            raise RuntimeError(
                "Notebook dependencies missing. Run uv sync --group dev --group notebooks in this project."
            )
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", PORT)) == 0:
                raise RuntimeError(
                    "Port 18581 is occupied by another service; it was left untouched."
                )
        for folder in (RUNTIME, ROOT / "logs/jupyter-config", ROOT / "logs/ipython"):
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            folder.chmod(0o700)
        # The kernel belongs to this venv, not the user's global Jupyter installation.
        subprocess.run(
            [
                str(PYTHON),
                "-m",
                "ipykernel",
                "install",
                "--prefix",
                str(ROOT / ".venv"),
                "--name",
                "reclab",
                "--display-name",
                "Search Recommendation Ads Lab (Python 3.12)",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        env = dict(os.environ)
        env["JUPYTER_RUNTIME_DIR"] = str(RUNTIME)
        env["JUPYTER_CONFIG_DIR"] = str(ROOT / "logs/jupyter-config")
        env["JUPYTER_DATA_DIR"] = str(ROOT / ".venv/share/jupyter")
        env["IPYTHONDIR"] = str(ROOT / "logs/ipython")
        env["JUPYTER_TOKEN"] = secrets.token_urlsafe(32)
        log_path = ROOT / "logs/notebook.log"
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.chmod(log_path, 0o600)
        with os.fdopen(fd, "a") as log:
            proc = subprocess.Popen(
                [
                    str(PYTHON),
                    "-m",
                    "jupyterlab",
                    "--no-browser",
                    "--ServerApp.ip=127.0.0.1",
                    f"--ServerApp.port={PORT}",
                    "--ServerApp.port_retries=0",
                    f"--ServerApp.root_dir={ROOT}",
                    "--ServerApp.allow_remote_access=False",
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        PID_FILE.write_text(str(proc.pid))
        for _ in range(120):
            info = server_info()
            if info:
                return info
            if proc.poll() is not None:
                raise RuntimeError(
                    "JupyterLab exited; inspect logs/notebook.log locally."
                )
            time.sleep(0.25)
        raise RuntimeError(
            "JupyterLab startup timed out. The owned process was preserved for inspection."
        )


def login_url(target="course"):
    if target not in TARGETS:
        raise ValueError("Unknown notebook/editor target")
    info = ensure_server()
    next_path = "/lab/tree/" + TARGETS[target] if TARGETS[target] else "/lab"
    # /login consumes the token and redirects to a clean URL with a signed cookie.
    return BASE_URL + "/login?" + urlencode({"token": info["token"], "next": next_path})


def status():
    return {
        "ready": server_info() is not None,
        "owned_pid": owned_pid(),
        "entry": "http://127.0.0.1:18580/notebooks",
        "authentication": "token + signed session cookie",
        "project_root": str(ROOT),
    }


def stop():
    pid = owned_pid()
    if not pid:
        return {"state": "stopped"}
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if owned_pid() is None:
            PID_FILE.unlink(missing_ok=True)
            return {"state": "stopped", "data_preserved": True}
        time.sleep(0.1)
    raise RuntimeError("Notebook shutdown is pending; PID retained.")

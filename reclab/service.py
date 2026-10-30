from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import threading
import uuid
import numpy as np
from scipy import sparse
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from .data import Dataset
from .paths import ROOT, ARTIFACTS, PROCESSED, read_json, atomic_json
from .train import TrainConfig, load_model, model_scorer
from .retrieval import cf_scores, topk, mmr

app = FastAPI(title="Recommendation Lab", docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
SOURCE_FILES = {
    "search_lessons": ("reclab/search_lessons.py", "深入搜索：复用真实语料、向量和查询级训练练习"),
    "ads_teaching": ("reclab/ads_teaching.py", "真实点击小预算单变量教学训练"),
    "ads": ("reclab/ads.py", "真实点击预测、独立校准与分段审计"),
    "probability": ("reclab/probability.py", "共享概率评估：AUC、logloss、Brier与可靠性"),
    "auction": ("reclab/auction.py", "独立模拟市场：二价拍卖与预算pacing"),
    "search": ("reclab/search.py", "真实查询与文档：检索、融合和重排"),
    "experiments": ("reclab/experiments.py", "跨搜索推荐广告共用实验记录"),
    "deep_recommendation": ("reclab/deep_recommendation.py", "真实推荐：负采样、序列与索引对照"),
    "xmrec_data": ("reclab/competition/data.py", "竞赛复现：官方数据与标签边界"),
    "xmrec_memory": ("reclab/competition/memory.py", "竞赛复现：冠军 ItemCF 公式移植"),
    "xmrec_graph": ("reclab/competition/graph.py", "竞赛复现：负采样、LightGCN 与 BPR"),
    "xmrec_ranking": ("reclab/competition/ranking.py", "竞赛复现：LightGBM 候选评分"),
    "xmrec_experiment": ("reclab/competition/experiment.py", "竞赛复现：两阶段完整实验"),
    "xmrec_metrics": ("reclab/competition/metrics.py", "竞赛复现：官方 nDCG@10"),

    "models": ("reclab/models.py", "双塔网络：Embedding、MLP、归一化与损失"),
    "data": ("reclab/data.py", "真实数据、时间切分与因果训练样本"),
    "retrieval": ("reclab/retrieval.py", "ItemCF、过滤、排名指标和 MMR"),
    "train": ("reclab/train.py", "训练循环、优化器、checkpoint 与验证"),
    "service": ("reclab/service.py", "Python 推荐服务与工作台 API"),
}
job_lock = threading.Lock()
job_state = {"state": "idle"}
torch.set_num_threads(4)


@app.middleware("http")
async def local_only(request: Request, call_next):
    # Browser-origin protection for local training writes; no CORS or arbitrary commands.
    if request.method not in ("GET", "HEAD"):
        from urllib.parse import urlsplit

        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {"detail": "Cross-origin write rejected"}, status_code=403
            )
        if request.headers.get("sec-fetch-site") == "cross-site":
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {"detail": "Cross-site write rejected"}, status_code=403
            )
    return await call_next(request)


@lru_cache(maxsize=1)
def data():
    return Dataset()


@lru_cache(maxsize=1)
def similarity():
    return sparse.load_npz(ARTIFACTS / "baselines/itemcf.npz")


def reports():
    return [read_json(p) for p in sorted((ARTIFACTS / "runs").glob("*/report.json"))]


@lru_cache(maxsize=3)
def loaded(run_id):
    if not any(r["run_id"] == run_id for r in reports()):
        raise HTTPException(404, "Unknown completed run")
    model, report = load_model(data(), run_id)
    _, vectors = model_scorer(data(), model, report["config"]["history_len"])
    return model, report, vectors


@app.get("/")
def home():
    return FileResponse(ROOT / "web/index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "project": str(ROOT),
        "data_ready": (PROCESSED / "manifest.json").is_file(),
    }


@app.get("/api/code")
def code_catalog():
    return {
        "files": [
            {"id": key, "path": value[0], "description": value[1]}
            for key, value in SOURCE_FILES.items()
        ]
    }


@app.get("/api/notebooks")
def notebook_catalog():
    from .notebook import TARGETS

    curriculum = read_json(ROOT / "docs/curriculum.json")
    metadata = {row["file"]: row for row in curriculum.get("notebook_chapters", [])}
    basics = {
        "00_Course_Map.ipynb": ("课程导航", "从这里选择章节与学习顺序"),
        "01_Data_Features_ItemCF.ipynb": (
            "数据、特征与 ItemCF",
            "真实数据、SQL 与基线推荐",
        ),
        "02_TwoTower_HandsOn.ipynb": ("双塔与梯度实操", "真实数据、张量、损失与训练"),
    }
    rows = []
    for target, relative in TARGETS.items():
        path = Path(relative)
        if path.parent != Path("notebooks") or path.suffix != ".ipynb":
            continue
        chapter = metadata.get(path.name, {})
        title, mode = basics.get(
            path.name,
            (chapter.get("title", path.stem), chapter.get("mode", "Notebook 实验")),
        )
        rows.append(
            {
                "target": target,
                "file": path.name,
                "title": f"{path.name[:2]} · {title}",
                "mode": mode,
                "available": (ROOT / path).is_file(),
            }
        )
    return {"notebooks": sorted(rows, key=lambda row: row["file"])}


@app.get("/api/code/{module}")
def source_code(module: str):
    if module not in SOURCE_FILES:
        raise HTTPException(404, "Unknown source module")
    relative, description = SOURCE_FILES[module]
    source = (ROOT / relative).read_text()
    return {
        "path": relative,
        "description": description,
        "source": source,
        "lines": len(source.splitlines()),
        "editable": False,
    }


@app.get("/notebooks")
def open_notebook(request: Request, target: str = "course"):
    from .notebook import login_url, TARGETS
    from urllib.parse import urlsplit

    origin = request.headers.get("origin")
    if request.headers.get("sec-fetch-site") == "cross-site" or (
        origin and urlsplit(origin).netloc != request.headers.get("host")
    ):
        raise HTTPException(403, "Cross-site notebook launch rejected")
    if target not in TARGETS:
        raise HTTPException(404, "Unknown notebook/editor target")
    try:
        response = RedirectResponse(login_url(target), status_code=303)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/summary")
def summary():
    ds = data()
    runs = reports()
    users, _ = ds.evaluation("valid", 300)
    return {
        "manifest": ds.manifest,
        "baselines": read_json(ARTIFACTS / "baselines/metrics.json"),
        "runs": [{k: v for k, v in r.items() if k not in ("samples",)} for r in runs],
        "users": [
            {
                "user_idx": u,
                "user_id": ds.users[u],
                "train_events": len(ds.histories[u]),
                "distinct_items": len(ds.seen[u]),
            }
            for u in users
        ],
        "job": job_state,
        "final_test": read_json(ARTIFACTS / "final-test.json")
        if (ARTIFACTS / "final-test.json").exists()
        else None,
    }


@app.get("/api/users/{user}/features")
def features(user: int, run_id: str | None = None):
    ds = data()
    if user not in ds.histories:
        raise HTTPException(404, "Unknown training user index")
    events = ds.train[ds.train.user_idx == user].sort_values("timestamp", kind="stable")
    history = ds.histories[user]
    cats = Counter(ds.items[i - 1]["category"] for i in history)
    vec = []
    actual = history[-20:]
    if run_id:
        model, report, _ = loaded(run_id)
        actual = history[-report["config"]["history_len"] :]
        with torch.no_grad():
            vec = (
                model.encode_user(
                    torch.tensor([user]),
                    torch.tensor(
                        ds.history_array(user, report["config"]["history_len"])[None, :]
                    ),
                )
                .numpy()[0]
                .tolist()
            )
    return {
        "user_idx": user,
        "user_id": ds.users[user],
        "snapshot_ms": ds.manifest["train_end_ms"],
        "feature_version": "history-id-v1",
        "model_inputs": {
            "user_idx": user,
            "history_item_indices": actual,
            "padding_idx": 0,
            "history_pooling": "Masked mean; order-invariant within the retained window.",
        },
        "metadata_not_model_inputs": {
            "training_event_count": len(history),
            "distinct_items": len(ds.seen[user]),
            "category_counts": cats.most_common(12),
            "last_event_ms": int(events.timestamp.max()),
        },
        "history": [{**ds.items[i - 1], "item_idx": i} for i in actual],
        "events": [
            {
                "timestamp": int(r.timestamp),
                "event": str(r.event),
                "item_id": int(r.itemid),
                "item_idx": int(r.item_idx),
            }
            for r in events.tail(80).itertuples()
        ],
        "user_vector": vec,
    }


class RecommendationRequest(BaseModel):
    user_idx: int = Field(ge=0)
    run_id: str
    channel: str = "hybrid"
    limit: int = Field(default=10, ge=1, le=30)
    candidate_budget: int = Field(default=100, ge=20, le=500)
    mmr_lambda: float = Field(default=1.0, ge=0.0, le=1.0)
    history_override: list[int] | None = Field(default=None, max_length=100)


@app.post("/api/recommend")
def recommend(body: RecommendationRequest):
    import time

    started = time.perf_counter()
    ds = data()
    if body.user_idx not in ds.histories:
        raise HTTPException(404, "Unknown user")
    if body.channel not in ("popular", "itemcf", "two_tower", "hybrid"):
        raise HTTPException(422, "Unknown channel")
    if body.history_override is not None and any(
        i < 1 or i >= ds.n_items for i in body.history_override
    ):
        raise HTTPException(422, "History IDs must be valid item indices")
    model, report, vectors = loaded(body.run_id)
    history = (
        ds.histories[body.user_idx]
        if body.history_override is None
        else body.history_override
    )
    # Counterfactual changes model context, not the actual already-consumed exclusion set.
    excluded = ds.seen[body.user_idx]
    with torch.no_grad():
        user_vec = model.encode_user(
            torch.tensor([body.user_idx]),
            torch.tensor(
                ds.history_array(
                    body.user_idx, report["config"]["history_len"], history
                )[None, :]
            ),
        ).numpy()[0]
    scores = {
        "popular": ds.popularity,
        "itemcf": cf_scores(similarity(), history, ds.n_items),
        "two_tower": vectors @ user_vec,
    }
    candidates = {
        name: topk(values, body.candidate_budget, excluded)
        for name, values in scores.items()
    }
    ranks = {
        name: {item: rank + 1 for rank, item in enumerate(items)}
        for name, items in candidates.items()
    }
    if body.channel == "hybrid":
        merged = np.full(ds.n_items, -np.inf)
        for ids in candidates.values():
            for rank, item in enumerate(ids):
                if not np.isfinite(merged[item]):
                    merged[item] = 0.0
                merged[item] += 1 / (60 + rank + 1)
        pool = topk(merged, body.candidate_budget, excluded)
    else:
        merged = scores[body.channel]
        pool = candidates[body.channel]
    chosen, steps = mmr(pool, merged, vectors, body.limit, body.mmr_lambda)
    step_map = {s["item_idx"]: s for s in steps}
    output = []
    for pos, item in enumerate(chosen, 1):
        weights = np.geomspace(0.2, 1, len(history[-50:]))
        evidence = []
        for h, w in zip(history[-50:], weights):
            score = float(similarity()[h, item]) * float(w)
            if score > 0:
                evidence.append(
                    {
                        "history_item_idx": h,
                        "history_item_id": ds.items[h - 1]["item_id"],
                        "contribution": score,
                    }
                )
        output.append(
            {
                **ds.items[item - 1],
                "position": pos,
                "channels": {name: r[item] for name, r in ranks.items() if item in r},
                "scores": {name: float(v[item]) for name, v in scores.items()},
                "fusion_score": float(merged[item]),
                "rerank": step_map[item],
                "itemcf_evidence": sorted(evidence, key=lambda e: -e["contribution"])[
                    :5
                ],
                "vector": vectors[item].tolist(),
            }
        )
    trace = {
        "request_id": uuid.uuid4().hex,
        "user_id": ds.users[body.user_idx],
        "run_id": body.run_id,
        "dataset_version": ds.manifest["dataset_version"],
        "feature_version": "history-id-v1",
        "snapshot_ms": ds.manifest["train_end_ms"],
        "counterfactual": body.history_override is not None,
        "channel": body.channel,
        "catalog": ds.n_items - 1,
        "seen_excluded": len(excluded),
        "available_after_filter": ds.n_items - 1 - len(excluded),
        "per_channel_candidates": {k: len(v) for k, v in candidates.items()},
        "merged_pool": len(pool),
        "returned": len(output),
        "mmr_lambda": body.mmr_lambda,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "stage_status": {
            "retrieval": "executed",
            "fusion": "RRF(k=60)" if body.channel == "hybrid" else "single channel",
            "coarse_rank": "not implemented",
            "fine_rank": "not implemented",
            "rerank": "greedy MMR on two-tower cosine vectors",
        },
        "evaluation_note": "Interactive counterfactual output is not an offline metric or observed CTR.",
    }
    result = {
        "trace": trace,
        "items": output,
        "channel_candidates": {
            name: [
                {
                    "item_idx": i,
                    "item_id": ds.items[i - 1]["item_id"],
                    "rank": n + 1,
                    "score": float(scores[name][i]),
                }
                for n, i in enumerate(ids)
            ]
            for name, ids in candidates.items()
        },
        "user_vector": user_vec.tolist(),
        "effective_history": history[-report["config"]["history_len"] :],
    }
    atomic_json(ARTIFACTS / "traces" / (trace["request_id"] + ".json"), result)
    return result


@app.get("/api/runs/{run_id}/samples")
def samples(run_id: str):
    _, report, _ = loaded(run_id)
    return {
        "samples": report["samples"],
        "negative_policy": "Training uses shuffled in-batch target IDs; same target IDs are masked. These are sampled negatives, not observed non-clicks.",
    }


@app.get("/api/curriculum")
def curriculum():
    return read_json(ROOT / "docs/curriculum.json")


@app.get("/api/runs/{run_id}/diagnostics")
def diagnostics(run_id: str):
    model, report, _ = loaded(run_id)
    samples = report["samples"][:8]
    config = report["config"]
    length = config["history_len"]
    users = torch.tensor([s["user_idx"] for s in samples])
    targets = torch.tensor([s["positive_item_idx"] for s in samples])
    history = torch.tensor(
        [[0] * (length - len(s["history"])) + s["history"] for s in samples]
    )
    with torch.no_grad():
        cosine = model.encode_user(users, history) @ model.encode_item(targets).T
    logits = cosine / config["temperature"]
    if config["logq_correction"]:
        logq = torch.from_numpy(np.load(ARTIFACTS / "runs" / run_id / "logq.npy"))
        logits = logits - logq[targets][None, :]
    duplicate = targets[:, None].eq(targets[None, :])
    duplicate.fill_diagonal_(False)
    logits = logits.masked_fill(duplicate, -1e9).detach().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(samples)))
    gradients = torch.autograd.grad(loss, logits)[0]
    return {
        "users": users.tolist(),
        "targets": targets.tolist(),
        "cosine": cosine.tolist(),
        "logits": logits.detach().tolist(),
        "probabilities": logits.detach().softmax(1).tolist(),
        "gradients": gradients.tolist(),
        "loss": float(loss.detach()),
        "temperature": config["temperature"],
        "logq_correction": config["logq_correction"],
        "note": "Gradient is with respect to this reconstructed batch's logits only; no parameters were updated. Same-target accidental hits are masked. Unobserved preferences can still produce false negatives.",
    }


@app.get("/api/lectures/{name}")
def lecture(name: str):
    curriculum = read_json(ROOT / "docs/curriculum.json")
    row = next((x for x in curriculum["lessons"] if x["file"] == name), None)
    if not row:
        raise HTTPException(404)
    p = Path(curriculum["local_source"]) / name
    if not p.is_file():
        raise HTTPException(404, "Local lecture unavailable")
    return FileResponse(p, media_type="application/pdf")


@app.get("/api/job")
def job():
    state = dict(job_state)
    path = ARTIFACTS / "runs" / str(state.get("run_id", "_")) / "status.json"
    if path.exists():
        state["progress"] = read_json(path)
    return state


@app.post("/api/train", status_code=202)
def start_training(config: TrainConfig):
    if config.device not in ("cpu", "mps"):
        raise HTTPException(422, "device must be cpu or mps")
    if not job_lock.acquire(blocking=False):
        raise HTTPException(409, "A training job is already running")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    config_path = ARTIFACTS / "configs" / (run_id + ".json")
    atomic_json(config_path, config.model_dump())
    job_state.clear()
    job_state.update({"state": "running", "run_id": run_id})

    def worker():
        try:
            log = ARTIFACTS / "logs" / (run_id + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as out:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "reclab.cli",
                        "train",
                        "--config",
                        str(config_path),
                        "--run-id",
                        run_id,
                    ],
                    cwd=ROOT,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
            job_state["state"] = "complete" if result.returncode == 0 else "failed"
            job_state["returncode"] = result.returncode
        except Exception as exc:
            job_state.update({"state": "failed", "error": str(exc)})
        finally:
            job_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return dict(job_state)


@app.get('/api/experiments')
def experiment_catalog():
    from .experiments import catalog
    return {'experiments': catalog()}


@app.get('/api/experiments/{domain}/{run_id}')
def experiment_result(domain: str, run_id: str):
    from .experiments import result
    try:
        return result(domain, run_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, 'Unknown experiment')


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    method: str = 'hybrid_rrf'
    limit: int = Field(default=10, ge=1, le=20)


@lru_cache(maxsize=1)
def search_index(run_id):
    from .search import SearchIndex
    return SearchIndex(read_json(ARTIFACTS / 'experiments/search' / run_id / 'receipt.json'))


@app.post('/api/search')
def search_documents(request: SearchRequest):
    from .search import latest
    if request.method not in ('bm25','dense','hybrid_rrf','linear_rerank','tinybert_rerank') or not request.query.strip():
        raise HTTPException(422, 'Choose a supported method and nonempty query')
    try:
        receipt, _ = latest()
        return search_index(receipt['run_id']).search(request.query,request.method,request.limit)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, 'Search inputs unavailable; run python -m reclab.search --download') from exc


class AdsPredictRequest(BaseModel):
    numeric: list[float | None] = Field(min_length=13, max_length=13)
    categorical: list[str | None] = Field(min_length=26, max_length=26)


@lru_cache(maxsize=1)
def ads_predictor(run_id):
    from .ads import Predictor
    return Predictor(run_id)


@app.get('/api/ads/example')
def ads_example():
    from .ads import latest, NUM, CAT
    try:
        receipt, _ = latest()
        examples = read_json(ARTIFACTS / 'experiments/ads' / receipt['run_id'] / 'example_inputs.json')
        row = examples[0]
        return {'numeric': [row[k] for k in NUM], 'categorical': [row[k] for k in CAT]}
    except FileNotFoundError as exc:
        raise HTTPException(503, 'Run python -m reclab.ads --download') from exc


@app.post('/api/ads/predict')
def ads_predict(request: AdsPredictRequest):
    import math
    from .ads import latest
    if any(v is not None and (not math.isfinite(v) or abs(v)>1e12) for v in request.numeric) or any(v is not None and len(v)>64 for v in request.categorical):
        raise HTTPException(422, 'Use finite numeric features and categorical strings up to 64 characters')
    try:
        receipt, _ = latest()
        return ads_predictor(receipt['run_id']).predict(request.numeric, request.categorical)
    except FileNotFoundError as exc:
        raise HTTPException(503, 'Run python -m reclab.ads --download') from exc


class AuctionRequest(BaseModel):
    strategy: str = 'value'
    paced: bool = True
    budget: float = Field(default=60., gt=0, le=10000, allow_inf_nan=False)
    multiplier: float = Field(default=1., gt=0, le=10, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=1000000)


@app.post('/api/ads/simulate')
def ads_simulate(request: AuctionRequest):
    from .auction import market, simulate
    if request.strategy not in ('flat', 'value'):
        raise HTTPException(422, 'Choose flat or value')
    return {'scope': 'SYNTHETIC market only; revenue and currency units are simulated. Independent of Criteo predictions.',
            **simulate(market(request.seed), request.strategy, request.budget, request.multiplier, request.paced)}

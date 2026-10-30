from datetime import datetime, timezone
import random
import time
import numpy as np
from scipy import sparse
import torch
from pydantic import BaseModel, Field
from .data import Dataset
from .models import TwoTower
from .paths import ARTIFACTS, atomic_json, read_json
from .retrieval import itemcf, cf_scores, topk, metrics


class TrainConfig(BaseModel):
    name: str = Field(default="two-tower", min_length=1, max_length=60)
    dim: int = Field(default=32, ge=8, le=128)
    history_len: int = Field(default=20, ge=2, le=100)
    use_history: bool = True
    logq_correction: bool = False
    epochs: int = Field(default=8, ge=1, le=30)
    batch_size: int = Field(default=256, ge=32, le=512)
    learning_rate: float = Field(default=0.003, ge=0.00001, le=0.03)
    temperature: float = Field(default=0.1, ge=0.02, le=1.0)
    dropout: float = Field(default=0.1, ge=0.0, le=0.6)
    seed: int = Field(default=42, ge=0, le=100000)
    max_samples: int = Field(default=120000, ge=1000, le=250000)
    eval_users: int = Field(default=1000, ge=50, le=2000)
    device: str = "cpu"


def evaluate(dataset, score_fn, split="valid", max_users=1000):
    targets, audit = dataset.evaluation(split, max_users)
    if not targets:
        raise ValueError(
            f"No eligible targets in {split}; an empty evaluation is not a zero score."
        )
    predictions = {u: topk(score_fn(u), 100, dataset.seen[u]) for u in targets}
    result = {
        "split": split,
        "audit": audit,
        "candidate_policy": "Every training-catalog item except training-seen items; deterministic item-index tie break.",
        "catalog_items": dataset.n_items - 1,
        "metrics": {},
    }
    for k in (10, 20, 50, 100):
        result["metrics"].update(metrics(predictions, targets, k))
    result["metrics"]["catalog_coverage@20"] = len(
        {i for pred in predictions.values() for i in pred[:20]}
    ) / (dataset.n_items - 1)
    return result


def ensure_baselines(dataset):
    folder = ARTIFACTS / "baselines"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "itemcf.npz"
    if path.exists():
        sim = sparse.load_npz(path)
    else:
        sim = itemcf(dataset)
        sparse.save_npz(path, sim)
    results = {}
    for name, fn in {
        "popular": lambda u: dataset.popularity,
        "itemcf": lambda u: cf_scores(sim, dataset.histories[u], dataset.n_items),
    }.items():
        results[name] = evaluate(dataset, fn)
    atomic_json(
        folder / "metrics.json",
        {
            "dataset_version": dataset.manifest["dataset_version"],
            "models": results,
            "itemcf_variant": "Binary user-item cosine with inverse log user activity and top-100 neighbors; last-50 recency weights.",
        },
    )
    return results


def load_model(dataset, run_id):
    folder = ARTIFACTS / "runs" / run_id
    report = read_json(folder / "report.json")
    if report["dataset_version"] != dataset.manifest["dataset_version"]:
        raise ValueError("Dataset version mismatch")
    config = TrainConfig(**report["config"])
    model = TwoTower(
        dataset.n_users, dataset.n_items, config.dim, config.use_history, config.dropout
    )
    model.load_state_dict(
        torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, report


def model_scorer(dataset, model, history_len=20):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        item_vectors = (
            model.encode_item(torch.arange(dataset.n_items, device=device))
            .cpu()
            .numpy()
        )

    def score(user):
        with torch.no_grad():
            vec = (
                model.encode_user(
                    torch.tensor([user], device=device),
                    torch.tensor(
                        dataset.history_array(user, history_len)[None, :], device=device
                    ),
                )
                .cpu()
                .numpy()[0]
            )
        return item_vectors @ vec

    return score, item_vectors


def train(config: TrainConfig, run_id=None):
    if config.device not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; select cpu")
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    dataset = Dataset()
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    folder = ARTIFACTS / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    atomic_json(
        folder / "status.json",
        {"state": "running", "epoch": 0, "config": config.model_dump()},
    )
    try:
        users, history, targets, timestamps = dataset.examples(config.history_len)
        if not len(targets):
            raise ValueError("No causal training samples")
        if len(targets) > config.max_samples:
            indices = np.sort(
                np.random.default_rng(config.seed).choice(
                    len(targets), config.max_samples, replace=False
                )
            )
            users, history, targets, timestamps = [
                x[indices] for x in (users, history, targets, timestamps)
            ]
        model = TwoTower(
            dataset.n_users,
            dataset.n_items,
            config.dim,
            config.use_history,
            config.dropout,
        ).to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )
        tensors = [torch.from_numpy(x) for x in (users, history, targets)]
        frequency = np.bincount(targets, minlength=dataset.n_items).astype(np.float32)
        logq = (
            torch.tensor(
                np.log(np.maximum(frequency, 1) / max(float(frequency.sum()), 1)),
                device=config.device,
            )
            if config.logq_correction
            else None
        )
        if logq is not None:
            np.save(folder / "logq.npy", logq.cpu().numpy())
        rng = torch.Generator().manual_seed(config.seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors),
            batch_size=config.batch_size,
            shuffle=True,
            generator=rng,
        )
        curve = []
        best = -1.0
        best_epoch = 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            total = 0.0
            count = 0
            grad_total = 0.0
            batches = 0
            for u, h, y in loader:
                if len(u) < 2:
                    continue
                u, h, y = [x.to(config.device) for x in (u, h, y)]
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(u, h, y, config.temperature, logq)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss")
                loss.backward()
                grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                optimizer.step()
                total += float(loss.detach()) * len(u)
                count += len(u)
                grad_total += grad
                batches += 1
            scorer, _ = model_scorer(dataset, model, config.history_len)
            val = evaluate(dataset, scorer, max_users=config.eval_users)
            score = val["metrics"]["ndcg@20"]
            if score > best:
                best = score
                best_epoch = epoch
                torch.save(model.state_dict(), folder / "model.pt")
            row = {
                "epoch": epoch,
                "loss": total / max(count, 1),
                "gradient_norm_before_clip": grad_total / max(batches, 1),
                "validation_ndcg@20": score,
                "validation_recall@20": val["metrics"]["recall@20"],
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            curve.append(row)
            atomic_json(
                folder / "status.json",
                {
                    "state": "running",
                    "epoch": epoch,
                    "config": config.model_dump(),
                    "curve": curve,
                },
            )
            print(f"{run_id} {row}", flush=True)
        model.load_state_dict(
            torch.load(
                folder / "model.pt", map_location=config.device, weights_only=True
            )
        )
        scorer, vectors = model_scorer(dataset, model, config.history_len)
        np.save(folder / "item_vectors.npy", vectors)
        val = evaluate(dataset, scorer, max_users=config.eval_users)
        # Actual examples from the frozen training arrays; no manufactured labels.
        samples = []
        for i in np.linspace(0, len(targets) - 1, min(60, len(targets)), dtype=int):
            samples.append(
                {
                    "user_idx": int(users[i]),
                    "user_id": dataset.users[int(users[i])],
                    "history": [int(x) for x in history[i] if x],
                    "positive_item_idx": int(targets[i]),
                    "event_time_ms": int(timestamps[i]),
                    "negative_policy": "Other target IDs in the shuffled minibatch; equal positive IDs masked.",
                }
            )
        report = {
            "run_id": run_id,
            "state": "complete",
            "config": config.model_dump(),
            "dataset_version": dataset.manifest["dataset_version"],
            "feature_version": "history-id-v1",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_samples": len(targets),
            "best_epoch": best_epoch,
            "selection": "Highest validation NDCG@20; test not read during training or selection.",
            "curve": curve,
            "validation": val,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "samples": samples,
            "reproduction_scope": "Educational two-tower retrieval + in-batch sampled softmax. Optional LogQ uses exact empirical target frequency in this training subset, not the paper's streaming estimator. Not an original-paper dataset/scale/results reproduction.",
            "architecture": "user ID embedding + masked mean recent item-ID embeddings -> MLP -> L2; item ID embedding -> MLP -> L2; dot similarity.",
        }
        atomic_json(folder / "report.json", report)
        atomic_json(
            folder / "status.json", {"state": "complete", "epoch": config.epochs}
        )
        return report
    except Exception as exc:
        atomic_json(folder / "status.json", {"state": "failed", "error": str(exc)})
        raise


def final_test(run_id):
    dataset = Dataset()
    model, report = load_model(dataset, run_id)
    scorer, _ = model_scorer(dataset, model, report["config"]["history_len"])
    sim = sparse.load_npz(ARTIFACTS / "baselines/itemcf.npz")
    result = {
        "selected_run": run_id,
        "dataset_version": dataset.manifest["dataset_version"],
        "selection_policy": "Select by validation before this command; do not tune against test.",
        "models": {},
    }
    for name, fn in {
        "popular": lambda u: dataset.popularity,
        "itemcf": lambda u: cf_scores(sim, dataset.histories[u], dataset.n_items),
        run_id: scorer,
    }.items():
        result["models"][name] = evaluate(
            dataset, fn, "test", report["config"]["eval_users"]
        )
    atomic_json(ARTIFACTS / "final-test.json", result)
    return result

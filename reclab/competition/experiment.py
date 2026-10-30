"""End-to-end fixed-protocol competition experiment; append-only run directories."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import resource
import time
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import torch
import lightgbm
from threadpoolctl import threadpool_limits
from .data import OUTPUT, SPEC, MARKETS, interactions, corpus, candidates, partition_users, sha256
from .memory import ItemCF
from .graph import GraphConfig, train_graph, graph_score
from .ranking import train_ranker, score_partitions
from .metrics import evaluate


def save_memory(model, path):
    path.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path / "similarity.npz", model.similarity)
    sparse.save_npz(path / "history.npz", model.history)
    (path / "index.json").write_text(json.dumps({"users": model.users.tolist(),
        "items": model.items.tolist(), "kind": model.kind, "top_k": model.top_k}))


def load_memory(path):
    path = Path(path)
    meta = json.loads((path / "index.json").read_text())
    model = ItemCF(meta["kind"], meta["top_k"])
    model.users, model.items = pd.Index(meta["users"]), pd.Index(meta["items"])
    model.history = sparse.load_npz(path / "history.npz")
    model.similarity = sparse.load_npz(path / "similarity.npz")
    return model


def latest_run():
    return OUTPUT / json.loads((OUTPUT / "latest.json").read_text())["run_id"]


def add_statistics(rows):
    """Counts/rating means, separated by market and file, matching champion feature families."""
    result = pd.DataFrame(index=rows.index)
    for market in MARKETS:
        for part in ("train", "train_5core"):
            frame = interactions(market, part)
            counts = frame.groupby("itemId").size()
            result[f"{market}_{part}_item_count"] = rows.itemId.map(counts).fillna(0).astype("float32")
        if market.startswith("t"):
            mask = rows.market.eq(market)
            frame = interactions(market)
            for key, prefix in (("userId", "user"), ("itemId", "item")):
                for metric in ("count", "mean"):
                    values = frame.groupby(key).rating.agg(metric)
                    col = f"target_{prefix}_rating_{metric}"
                    if col not in result:
                        result[col] = np.float32(0)
                    result.loc[mask, col] = rows.loc[mask, key].map(values).fillna(0).astype("float32")
    return result.astype("float32")


def generate_features(run, graph_config):
    # Candidate identity is public. Relevance labels are joined only AFTER all features exist.
    valid = pd.concat([candidates(m, labels=False) for m in ("t1", "t2")], ignore_index=True)
    test = pd.concat([candidates(m, "test", labels=False) for m in ("t1", "t2")], ignore_index=True)
    rows = pd.concat([valid, test], ignore_index=True)
    features = add_statistics(rows)
    frames = {m: corpus([m]) for m in MARKETS}
    timings = []
    with threadpool_limits(limits=4):
        # Target-only and target+s1 preserve market-specific signals; all5 shares item evidence.
        for target in ("t1", "t2"):
            for suffix, mkts in (("local", [target]), ("s1", [target, "s1"]),
                                 ("s3", [target, "s3"])):
                start = time.monotonic()
                frame = pd.concat([frames[m] for m in mkts], ignore_index=True)
                model = ItemCF("iuf").fit(frame)
                mask = rows.market.eq(target)
                col = f"icf_{suffix}"
                if col not in features:
                    features[col] = np.float32(0)
                features.loc[mask, col] = model.score(rows.loc[mask])
                save_memory(model, run / "memory" / f"{target}_{suffix}")
                timings.append({"model": f"{target}_{suffix}", "seconds": time.monotonic() - start,
                                "interactions": len(frame), "similarity_nnz": model.similarity.nnz})
                print(f'PASS memory {target}_{suffix}: {timings[-1]["seconds"]:.1f}s', flush=True)
                del model
        all_frame = pd.concat(list(frames.values()), ignore_index=True)
        for kind in ("iuf", "rating"):
            start = time.monotonic()
            model = ItemCF(kind).fit(all_frame)
            features[f"icf_all_{kind}"] = model.score(rows)
            save_memory(model, run / "memory" / f"all_{kind}")
            timings.append({"model": f"all_{kind}", "seconds": time.monotonic() - start,
                            "interactions": len(all_frame), "similarity_nnz": model.similarity.nnz})
            print(f'PASS memory all_{kind}: {timings[-1]["seconds"]:.1f}s', flush=True)
            del model
        graph_trace = {}
        for layers in (0, graph_config.layers):
            cfg = GraphConfig(**{**asdict(graph_config), "layers": layers})
            fitted = train_graph(all_frame, cfg)
            features[f"graph_layers{layers}"] = graph_score(fitted, rows)
            graph_trace[str(layers)] = fitted["trace"]
            torch.save({"state_dict": fitted["model"].state_dict(), "config": fitted["config"],
                        "users": fitted["users"].tolist(), "items": fitted["items"].tolist()},
                       run / f"graph_layers{layers}.pt")
            np.savez_compressed(run / f"graph_vectors{layers}.npz",
                                users=fitted["user_vectors"], items=fitted["item_vectors"])
            del fitted
    (run / "graph_trace.json").write_text(json.dumps(graph_trace, indent=2))
    (run / "timings.json").write_text(json.dumps(timings, indent=2))
    labels = pd.concat([candidates(m) for m in ("t1", "t2")], ignore_index=True)
    assert labels[["market", "userId", "itemId"]].equals(valid[["market", "userId", "itemId"]])
    labels["partition"] = partition_users(labels)
    labels.to_parquet(run / "valid_rows.parquet", index=False)
    test.to_parquet(run / "test_rows.parquet", index=False)
    features.iloc[:len(valid)].to_parquet(run / "valid_features.parquet", index=False)
    features.iloc[len(valid):].reset_index(drop=True).to_parquet(run / "test_features.parquet", index=False)
    return labels, features.iloc[:len(valid)].reset_index(drop=True), test, features.iloc[len(valid):].reset_index(drop=True)


def run_experiment(graph_config=GraphConfig(), tag="champion-core"):
    if graph_config.layers == 0:
        raise ValueError("Main experiment needs a graph arm; use graph.train_graph for MF-only")
    start = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "-" + tag
    if not tag.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Use a simple run tag")
    run = OUTPUT / run_id
    run.mkdir(parents=True, exist_ok=False)
    config = {"run_id": run_id, "graph": asdict(graph_config),
              "scope": "champion core port; full data; reduced model/feature/ensemble recipe",
              "target_qrels_in_feature_corpus": False, "audit_fraction": 0.2,
              "source_spec": json.loads(SPEC.read_text()),
              "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "python": platform.python_version(),
              "torch": torch.__version__, "lightgbm": lightgbm.__version__,
              "numpy": np.__version__, "scipy": scipy.__version__}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    rows, features, test_rows, test_features = generate_features(run, graph_config)
    results = []
    predictions = {}
    for col in ("target_item_rating_count", "icf_local", "icf_s1", "icf_s3", "icf_all_iuf",
                "icf_all_rating", "graph_layers0", f"graph_layers{graph_config.layers}"):
        predictions[col] = features[col].to_numpy()
    stats = [c for c in features if not c.startswith(("icf_", "graph_"))]
    memory = [c for c in features if c.startswith("icf_")]
    arms = {"lgb_stats": (stats, 99), "lgb_memory": (stats + memory, 99),
            "lgb_all": (list(features), 99), "lgb_neg9": (list(features), 9)}
    selected_scores = {}
    models = {}
    for name, (columns, keep) in arms.items():
        pred = np.zeros(len(rows))
        test_pred = np.zeros(len(test_rows))
        for market in ("t1", "t2"):
            mask = rows.market.eq(market).to_numpy()
            tm = test_rows.market.eq(market).to_numpy()
            # Reset index before negative sampling so retained row IDs are local.
            model, p, receipt = train_ranker(rows.loc[mask].reset_index(drop=True),
                                             features.loc[mask, columns].reset_index(drop=True),
                                             negative_keep=keep)
            pred[mask] = p
            test_pred[tm] = model.predict_proba(test_features.loc[tm, columns])[:, 1]
            model.booster_.save_model(str(run / f"{name}_{market}.txt"))
            models[f"{name}_{market}"] = {"columns": columns, **receipt}
        predictions[name] = pred
        select = rows.partition.eq("select").to_numpy()
        selected_scores[name] = float(evaluate(rows.loc[select], pred[select]).iloc[-1].ndcg10)
        print(f'PASS ranker {name}: selection nDCG10={selected_scores[name]:.6f}', flush=True)
        test_rows.assign(score=test_pred).to_parquet(run / f"test_{name}.parquet", index=False)
    # Freeze the choice using selection users before inspecting ANY audit labels/metrics.
    chosen = max(selected_scores, key=selected_scores.get)
    (run / "selection.json").write_text(json.dumps({"selected": chosen, "selection_scores": selected_scores,
        "criterion": "combined select nDCG@10, user weighted; audit metrics not consulted"}, indent=2))
    for name, pred in predictions.items():
        table = score_partitions(rows, pred)
        table["model"] = name
        results.append(table)
    pd.DataFrame(predictions).to_parquet(run / "valid_predictions.parquet", index=False)
    report = pd.concat(results, ignore_index=True)
    report.to_csv(run / "metrics.csv", index=False)
    (run / "rankers.json").write_text(json.dumps(models, indent=2))
    receipt = {"run_id": run_id, "selected": chosen, "seconds": time.monotonic() - start,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
               "official_valid_users": int(rows.userId.nunique()), "features": len(features.columns),
               "status": "PASS", "hidden_test_score": None, "lightgbm_threads": 1}
    (run / "receipt.json").write_text(json.dumps(receipt, indent=2))
    (OUTPUT / "latest.json").write_text(json.dumps({"run_id": run_id}))
    print(json.dumps(receipt, indent=2), flush=True)
    return run

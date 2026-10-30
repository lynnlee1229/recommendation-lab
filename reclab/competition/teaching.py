"""Small notebook helpers; all experiment logic remains in editable Python modules."""
from datetime import datetime, timezone
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import OUTPUT
from .experiment import latest_run
from .metrics import evaluate, per_user
from .ranking import train_ranker


def load_run():
    run = latest_run()
    return run, pd.read_parquet(run / "valid_rows.parquet"), pd.read_parquet(run / "valid_features.parquet")


def style():
    plt.rcParams.update({"figure.figsize": (9, 4), "figure.dpi": 110, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#182b2b", "text.color": "#182b2b",
                         "axes.prop_cycle": plt.cycler(color=["#006e60", "#8c6832", "#596b6a"]),
                         "axes.axisbelow": True, "savefig.bbox": "tight"})


def bar_scores(table, title):
    fig, ax = plt.subplots(figsize=(9, max(3, .4 * len(table))))
    table = table.sort_values("ndcg10")
    ax.barh(table.model, table.ndcg10, color="#006e60")
    for idx, value in enumerate(table.ndcg10):
        ax.text(value + .008, idx, f"{value:.4f}", va="center")
    ax.set(xlim=(0, min(1, float(table.ndcg10.max()) + .1)), xlabel="nDCG@10 (official 100 candidates)", title=title)
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    plt.show()


def sandbox_ranker(columns=None, negative_keep=99, leaves=15, tag="notebook"):
    """Actual full t1 ranker retraining; expose selection only while experimenting."""
    run, rows, features = load_run()
    mask = rows.market.eq("t1")
    rows = rows.loc[mask].reset_index(drop=True)
    features = features.loc[mask].reset_index(drop=True)
    columns = list(features) if columns is None else list(columns)
    if not set(columns) <= set(features) or not columns:
        raise ValueError("Choose existing nonempty feature columns")
    model, scores, receipt = train_ranker(rows, features[columns], negative_keep=negative_keep, leaves=leaves)
    select = rows.partition.eq("select").to_numpy()
    metrics = evaluate(rows.loc[select], scores[select])
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    folder = OUTPUT / "experiments" / name
    folder.mkdir(parents=True, exist_ok=False)
    model.booster_.save_model(str(folder / "ranker.txt"))
    (folder / "receipt.json").write_text(json.dumps({"parent_run": run.name, "tag": tag,
        "columns": columns, "leaves": leaves, **receipt, "selection": metrics.to_dict("records")}, indent=2))
    return model, metrics, folder


def paired_bootstrap(rows, scores_a, scores_b, repeats=1000, seed=2022):
    """Stratified user bootstrap for a prespecified paired comparison; no causal claim."""
    a, b = per_user(rows, scores_a), per_user(rows, scores_b)
    joined = a.merge(b, on=["market", "userId"], suffixes=("_a", "_b"), validate="one_to_one")
    groups = [g.ndcg10_b.to_numpy() - g.ndcg10_a.to_numpy() for _, g in joined.groupby("market")]
    rng = np.random.default_rng(seed)
    estimates = np.zeros(repeats)
    for delta in groups:
        estimates += rng.choice(delta, (repeats, len(delta)), replace=True).sum(1)
    estimates /= len(joined)
    return {"users": len(joined), "delta_b_minus_a": float(np.concatenate(groups).mean()),
            "bootstrap_95_interval": np.quantile(estimates, [.025, .975]).tolist(),
            "scope": "fixed published audit users; descriptive after repeated inspection"}

"""Official single-positive nDCG@10; deterministic item-ID ties for local runs."""
import numpy as np
import pandas as pd


def per_user(rows, scores, k=10):
    values = np.asarray(scores)
    if len(values) != len(rows) or not np.isfinite(values).all():
        raise ValueError("Scores must be finite and aligned with all candidate rows")
    if not rows.groupby(["market", "userId"]).label.sum().eq(1).all():
        raise ValueError("Single-positive protocol required")
    ranked = rows[["market", "userId", "itemId", "label"]].copy()
    ranked["score"] = values
    ranked = ranked.sort_values(["market", "userId", "score", "itemId"],
                                ascending=[True, True, False, True], kind="stable")
    ranked["rank"] = ranked.groupby(["market", "userId"]).cumcount() + 1
    pos = ranked[ranked.label == 1].copy()
    pos["ndcg10"] = np.where(pos["rank"] <= k, 1 / np.log2(pos["rank"] + 1), 0)
    pos["hr10"] = (pos["rank"] <= k).astype(float)
    return pos[["market", "userId", "rank", "ndcg10", "hr10"]].reset_index(drop=True)


def evaluate(rows, scores):
    users = per_user(rows, scores)
    result = [{"market": m, "users": len(g), "ndcg10": float(g.ndcg10.mean()),
               "hr10": float(g.hr10.mean())} for m, g in users.groupby("market")]
    result.append({"market": "combined", "users": len(users),
                   "ndcg10": float(users.ndcg10.mean()), "hr10": float(users.hr10.mean())})
    return pd.DataFrame(result)

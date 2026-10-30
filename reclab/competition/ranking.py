"""Champion uses binary LightGBM scoring; local user-separated selection/audit."""
import numpy as np
import lightgbm as lgb
from .metrics import evaluate


def train_ranker(rows, features, negative_keep=99, seed=2022, rounds=250, leaves=15):
    if not (1 <= negative_keep <= 99 and 1 <= rounds <= 1000 and 2 <= leaves <= 63):
        raise ValueError("Ranker configuration exceeds local experiment bounds")
    fit = rows.partition.eq("fit").to_numpy()
    select = rows.partition.eq("select").to_numpy()
    if negative_keep < 99:
        # Retain all positives and exactly K negatives per fit user, never sample evaluation.
        rng = np.random.default_rng(seed)
        retained = []
        for _, group in rows[fit].groupby(["market", "userId"], sort=False):
            positives = group.index[group.label.eq(1)].to_numpy()
            neg = group.index[group.label.eq(0)].to_numpy()
            retained.extend(positives)
            retained.extend(rng.choice(neg, min(negative_keep, len(neg)), replace=False))
        fit = np.zeros(len(rows), bool)
        fit[retained] = True
    # macOS: avoid OpenMP barrier deadlock with the PyTorch runtime in one kernel.
    model = lgb.LGBMClassifier(n_estimators=rounds, num_leaves=leaves, learning_rate=0.05,
                               min_child_samples=40, reg_lambda=1.0, verbosity=-1,
                               n_jobs=1, random_state=seed, deterministic=True,
                               force_col_wise=True)
    model.fit(features.loc[fit], rows.loc[fit, "label"],
              eval_X=features.loc[select], eval_y=rows.loc[select, "label"],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    predictions = model.predict_proba(features)[:, 1]
    return model, predictions, {"fit_rows": int(fit.sum()), "negative_keep": negative_keep,
                                 "best_iteration": model.best_iteration_}


def score_partitions(rows, predictions):
    output = []
    for part in ("select", "audit", "all_official_valid"):
        mask = np.ones(len(rows), bool) if part == "all_official_valid" else rows.partition.eq(part).to_numpy()
        table = evaluate(rows.loc[mask], np.asarray(predictions)[mask])
        table["partition"] = part
        output.append(table)
    import pandas as pd
    return pd.concat(output, ignore_index=True)

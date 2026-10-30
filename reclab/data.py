"""Retailrocket preparation. The vocabulary and cohort use training data only."""

from hashlib import sha256
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
from .paths import PROCESSED, atomic_json, read_json

SOURCE = (
    "https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset"
)
WEIGHTS = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def prepare(
    archive: Path, max_items=5000, max_users=12000, train_ratio=0.70, valid_ratio=0.85
):
    """Global time cutoffs; evaluate novel, warm items using the frozen train snapshot."""
    if (PROCESSED / "manifest.json").exists():
        raise FileExistsError(
            "Prepared dataset exists. Keep this snapshot; use a separate project/data copy for another dataset."
        )
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if Path(n).name == "events.csv")
        events = pd.read_csv(
            z.open(name),
            dtype={
                "timestamp": "int64",
                "visitorid": "int64",
                "itemid": "int64",
                "event": "string",
            },
        )
        required = {"timestamp", "visitorid", "itemid", "event", "transactionid"}
        if not required <= set(events):
            raise ValueError("Unexpected Retailrocket schema")
        if events[list(required - {"transactionid"})].isna().any().any():
            raise ValueError("Null identity/time/behavior field")
        if not set(events.event.unique()) <= set(WEIGHTS):
            raise ValueError("Unknown behavior")
        raw_count = len(events)
        duplicates = int(events.duplicated().sum())
        events = (
            events.drop_duplicates()
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )
        # Wall-clock cutoffs, not per-user leave-one-out: no future users/items enter training.
        start, end = int(events.timestamp.min()), int(events.timestamp.max())
        train_end = int(start + (end - start) * train_ratio)
        valid_end = int(start + (end - start) * valid_ratio)
        train = events[events.timestamp < train_end]
        counts = (
            train.groupby("itemid").size().sort_values(ascending=False, kind="stable")
        )
        item_ids = counts[counts >= 5].head(max_items).index.sort_values().tolist()
        filtered = train[train.itemid.isin(item_ids)]
        activity = (
            filtered.groupby("visitorid")
            .itemid.nunique()
            .sort_values(ascending=False, kind="stable")
        )
        user_ids = activity[activity >= 5].head(max_users).index.sort_values().tolist()
        frame = events[events.visitorid.isin(user_ids)].copy()
        frame["split"] = np.select(
            [frame.timestamp < train_end, frame.timestamp < valid_end],
            ["train", "valid"],
            default="test",
        )
        imap = {int(v): i + 1 for i, v in enumerate(item_ids)}
        umap = {int(v): i for i, v in enumerate(user_ids)}
        frame["item_idx"] = frame.itemid.map(imap).fillna(0).astype("int32")
        frame["user_idx"] = frame.visitorid.map(umap).astype("int32")
        frame["weight"] = frame.event.map(WEIGHTS).astype("float32")
        frame["event_id"] = frame.index.astype(str)
        # Category snapshot is metadata at train cutoff, never a model input in v1.
        category = {}
        category_ts = {}
        for name in sorted(z.namelist()):
            if "item_properties" not in name or not name.endswith(".csv"):
                continue
            for chunk in pd.read_csv(
                z.open(name),
                chunksize=300000,
                dtype={"property": "string", "value": "string"},
            ):
                rows = chunk[
                    (chunk.property == "categoryid")
                    & (chunk.timestamp < train_end)
                    & chunk.itemid.isin(item_ids)
                ]
                for row in rows.sort_values("timestamp").itertuples():
                    key = imap[int(row.itemid)]
                    if int(row.timestamp) >= category_ts.get(key, -1):
                        category[key], category_ts[key] = (
                            str(row.value),
                            int(row.timestamp),
                        )
        PROCESSED.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(PROCESSED / "events.parquet", index=False)
        item_rows = [
            {
                "item_idx": imap[int(item)],
                "item_id": int(item),
                "category": category.get(imap[int(item)], "unknown"),
                "category_asof_ms": category_ts.get(imap[int(item)]),
                "train_events": int(counts.loc[item]),
            }
            for item in item_ids
        ]
        atomic_json(PROCESSED / "items.json", item_rows)
        atomic_json(PROCESSED / "users.json", [int(v) for v in user_ids])
        manifest = {
            "dataset": "Retailrocket",
            "source_url": SOURCE,
            "source_card": "https://www.kaggle.com/retailrocket/ecommerce-dataset/home",
            "archive_sha256": digest(archive),
            "raw_events": raw_count,
            "exact_duplicate_rows_removed": duplicates,
            "raw_unique_users": int(events.visitorid.nunique()),
            "raw_unique_items": int(events.itemid.nunique()),
            "start_ms": start,
            "end_ms": end,
            "train_end_ms": train_end,
            "valid_end_ms": valid_end,
            "cohort_users": len(user_ids),
            "catalog_items": len(item_ids),
            "cohort_events": len(frame),
            "warm_events": int((frame.item_idx > 0).sum()),
            "split_rows": {k: int(v) for k, v in frame.groupby("split").size().items()},
            "selection": "Top training-window items (min 5 events), then users with >=5 distinct training catalog items; cohort is not population-representative.",
            "task": "Novel warm-item future interaction retrieval. No exposure log and no CTR claim.",
            "metadata": "Latest category strictly before train cutoff; metadata only, not training inputs.",
            "feature_version": "history-id-v1",
            "policy_version": "global-time-novel-warm-v1",
        }
        manifest["dataset_version"] = sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()[:16]
        atomic_json(PROCESSED / "manifest.json", manifest)
    return manifest


class Dataset:
    def __init__(self):
        self.manifest = read_json(PROCESSED / "manifest.json")
        self.events = pd.read_parquet(PROCESSED / "events.parquet")
        self.items = read_json(PROCESSED / "items.json")
        self.users = read_json(PROCESSED / "users.json")
        self.n_items, self.n_users = len(self.items) + 1, len(self.users)
        self.train = self.events[
            (self.events.split == "train") & (self.events.item_idx > 0)
        ]
        self.histories = {
            int(u): group.sort_values("timestamp", kind="stable")
            .item_idx.astype(int)
            .tolist()
            for u, group in self.train.groupby("user_idx")
        }
        self.seen = {u: set(h) for u, h in self.histories.items()}
        self.popularity = np.zeros(self.n_items, dtype=np.float32)
        for item, count in self.train.groupby("item_idx").size().items():
            self.popularity[int(item)] = count

    def examples(self, history_len=20):
        users, histories, targets, times = [], [], [], []
        for user, group in self.train.groupby("user_idx", sort=True):
            past, seen, last_time = [], set(), -1
            for row in group.sort_values("timestamp", kind="stable").itertuples():
                item = int(row.item_idx)
                if (
                    len(seen) >= 2
                    and item not in seen
                    and last_time < int(row.timestamp)
                ):
                    h = past[-history_len:]
                    users.append(int(user))
                    histories.append([0] * (history_len - len(h)) + h)
                    targets.append(item)
                    times.append(int(row.timestamp))
                past.append(item)
                seen.add(item)
                last_time = int(row.timestamp)
        return (
            np.array(users, dtype=np.int64),
            np.array(histories, dtype=np.int64),
            np.array(targets, dtype=np.int64),
            np.array(times, dtype=np.int64),
        )

    def evaluation(self, split="valid", max_users=1000):
        future = self.events[self.events.split == split]
        targets = {}
        raw_targets, cold_targets, repeat_targets = 0, 0, 0
        for user, group in future.groupby("user_idx", sort=True):
            u = int(user)
            raw_targets += len(set(group.itemid.astype(int)))
            cold_targets += int(group[group.item_idx == 0].itemid.nunique())
            warm = set(group.item_idx.astype(int)) - {0}
            repeat_targets += len(warm & self.seen[u])
            novel = warm - self.seen[u]
            if novel:
                targets[u] = novel
        # Deterministic hash sample, not lowest IDs or most active future users.
        selected = sorted(
            targets, key=lambda u: sha256(str(self.users[u]).encode()).hexdigest()
        )[:max_users]
        return {u: targets[u] for u in selected}, {
            "eligible_users": len(targets),
            "evaluated_users": len(selected),
            "distinct_user_item_targets_before_filter": raw_targets,
            "cold_oov_targets_excluded": cold_targets,
            "previously_seen_targets_excluded": repeat_targets,
            "context": "Frozen train-cutoff history for both validation and test; validation events never update the test history.",
        }

    def history_array(self, user, length=20, override=None):
        h = (self.histories[user] if override is None else override)[-length:]
        return np.array([0] * (length - len(h)) + h, dtype=np.int64)

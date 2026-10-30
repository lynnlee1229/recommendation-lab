"""Pinned official data, fixed candidate lists, and label-free feature corpora."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import sparse
from reclab.paths import ROOT

BASE = ROOT / "data/xmrec"
RAW = BASE / "raw"
OUTPUT = ROOT / "artifacts/xmrec"
SPEC = ROOT / "docs/xmrec_sources.json"
MARKETS = ("s1", "s2", "s3", "t1", "t2")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(source=None):
    """Copy a verified checkout, or fetch the pinned public starter archive."""
    spec = json.loads(SPEC.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    if source is None:
        def download(name):
            dest = RAW / name
            if dest.exists():
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_suffix(".part")
            url = f'https://raw.githubusercontent.com/hamedrab/wsdm22_cup_xmrec/{spec["official_revision"]}/DATA/{name}'
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            if sha256(partial) != spec["files"][name]:
                raise ValueError(f"Download checksum mismatch: {name}")
            partial.replace(dest)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, spec["files"]))
    for name, digest in spec["files"].items():
        dest = RAW / name
        if source is not None:
            src = Path(source) / name
            if sha256(src) != digest:
                raise ValueError(f"Source checksum mismatch: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(src, dest)
        if sha256(dest) != digest:
            raise ValueError(f"Data checksum mismatch: {name}")
    original = BASE / "upstream/winner"
    for name, digest in spec["source_hashes"].items():
        dest = original / name
        if dest.exists() and sha256(dest) != digest:
            # Interrupted transfers from the original downloader were final-named.
            # Preserve them, then accept a replacement only after hash validation.
            backup = dest.with_suffix(dest.suffix + '.incomplete')
            if backup.exists():
                raise ValueError(f"Inspect preserved incomplete source: {backup}")
            dest.rename(backup)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://raw.githubusercontent.com/opdai/wsdm2022-xmrec-top1-solution/{spec["winner_revision"]}/{name}'
            partial = dest.with_suffix(dest.suffix + '.part')
            for attempt in range(5):
                offset = partial.stat().st_size if partial.exists() else 0
                request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request, timeout=60) as response:
                    resume = response.status == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
                    if response.status == 206 and not resume:
                        raise ValueError('Unexpected reference partial response')
                    with partial.open('ab' if resume else 'wb') as out:
                        shutil.copyfileobj(response, out)
                if sha256(partial) == digest:
                    partial.replace(dest)
                    break
                if attempt == 4:
                    raise ValueError(f"Pinned reference source download mismatch: {name}")
                time.sleep(attempt + 1)
        if sha256(dest) != digest:
            raise ValueError(f"Pinned reference source changed: {name}")
    return {"verified_files": len(spec["files"]), "root": str(RAW)}


def interactions(market, part="train"):
    if market not in MARKETS or part not in ("train", "train_5core", "valid_qrel"):
        raise ValueError("Unknown market or interaction partition")
    frame = pd.read_csv(RAW / market / f"{part}.tsv", sep="\t")
    if frame.isna().any().any():
        raise ValueError("Invalid interaction rows")
    return frame


def corpus(markets, include_source_valid=True):
    """Never reads t1/t2 qrels. 5core=5 and first duplicate wins, as in champion."""
    frames = []
    for market in markets:
        frames.append(interactions(market))
        core = interactions(market, "train_5core").copy()
        core["rating"] = 5.0
        frames.append(core)
        if include_source_valid and market.startswith("s"):
            frames.append(interactions(market, "valid_qrel"))
    return pd.concat(frames, ignore_index=True).drop_duplicates(["userId", "itemId"])


def candidates(market, part="valid", labels=True):
    if market not in ("t1", "t2") or part not in ("valid", "test"):
        raise ValueError("Unknown target partition")
    run = pd.read_csv(RAW / market / f"{part}_run.tsv", sep="\t", header=None,
                      names=["userId", "items"])
    lists = run["items"].str.split(",")
    if run.userId.duplicated().any() or not lists.map(lambda x: len(x) == len(set(x)) == 100).all():
        raise ValueError("Expected exactly 100 unique candidates for each user")
    run["itemId"] = lists
    rows = run[["userId", "itemId"]].explode("itemId", ignore_index=True)
    rows["market"] = market
    if part == "valid" and labels:
        qrel = interactions(market, "valid_qrel")
        if qrel.userId.duplicated().any() or set(qrel.userId) != set(run.userId):
            raise ValueError("Expected one relevance label per candidate user")
        positives = pd.MultiIndex.from_frame(qrel[["userId", "itemId"]])
        rows["label"] = pd.MultiIndex.from_frame(rows[["userId", "itemId"]]).isin(positives).astype("int8")
        if not rows.groupby("userId").label.sum().eq(1).all():
            raise ValueError("Every positive must occur once in official candidate list")
    return rows


def encode(frame, items=None):
    users = pd.Index(sorted(frame.userId.unique()))
    items = pd.Index(sorted(frame.itemId.unique())) if items is None else items
    u = users.get_indexer(frame.userId)
    i = items.get_indexer(frame.itemId)
    if (i < 0).any():
        raise ValueError("Missing item in vocabulary")
    matrix = sparse.csr_matrix((frame.rating.to_numpy(np.float32), (u, i)),
                               shape=(len(users), len(items)), dtype=np.float32)
    return users, items, matrix


def partition_users(rows, seed=2022):
    """Stable per-market user groups: 60% ranker fit,20% selection,20% audit."""
    keys = rows.market.astype(str) + ":" + rows.userId.astype(str)
    buckets = keys.map(lambda key: int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % 10)
    return np.select([buckets < 6, buckets < 8], ["fit", "select"], default="audit")

"""Execute only named classes from pinned, locally preserved original source for parity.

No original entry point/import/setup script is executed. Never load arbitrary files.
"""
import ast
from collections import defaultdict
import json
import math
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import BASE, SPEC, sha256, corpus, encode
from .memory import ItemCF
from .graph import LightGCN, normalized_graph


def original_class(relative, name, extra=None):
    spec = json.loads(SPEC.read_text())
    if relative not in spec["source_hashes"]:
        raise ValueError("Unregistered original source")
    path = BASE / "upstream/winner" / relative
    if sha256(path) != spec["source_hashes"][relative]:
        raise ValueError("Original source changed; parity reference must stay pinned")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"np": np, "pd": pd, "math": math, "defaultdict": defaultdict,
                 "nn": nn, "torch": torch, "BasicModel": nn.Module,
                 "config": SimpleNamespace(cprint=lambda *args: None)}
    namespace.update(extra or {})
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def check_memory(frame):
    checks = {}
    for kind, file in (("iuf", "icf_v2.py"), ("rating", "icf_v3.py")):
        cls = original_class(f"src/retrieval/{file}", "itemcf_iuf")
        original_frame = frame.copy()
        if kind == "rating":
            original_frame["itemId"] = frame.itemId + "__" + frame.rating.astype(str)
        original = cls(original_frame)
        original.build_i2i_similarity_map()
        port = ItemCF(kind).fit(frame)
        rows = pd.MultiIndex.from_product([port.users, port.items],
                                         names=["userId", "itemId"]).to_frame(index=False)
        expected = []
        scores = {u: dict(original.recommend(u)) for u in port.users}
        for u, i in rows.itertuples(index=False, name=None):
            expected.append(scores[u].get(i, 0))
        error = float(np.max(np.abs(port.score(rows) - expected)))
        assert error < 2e-5, (kind, error)
        checks[kind] = {"max_score_error": error, "pairs": len(rows)}
    return checks


def check_graph(frame):
    _, _, history = encode(frame)
    history.data[:] = 1
    graph = normalized_graph(history)
    original_cls = original_class("src/retrieval/lgcn_v2/model.py", "LightGCN")
    cfg = {"latent_dim_rec": 8, "lightGCN_n_layers": 2, "keep_prob": 1.0,
           "A_split": False, "pretrain": 0, "dropout": False}
    dataset = SimpleNamespace(n_users=history.shape[0], m_items=history.shape[1],
                              getSparseGraph=lambda: graph)
    original = original_cls(cfg, dataset)
    port = LightGCN(*history.shape, dim=8, layers=2)
    with torch.no_grad():
        port.user_embedding.weight.copy_(original.embedding_user.weight)
        port.item_embedding.weight.copy_(original.embedding_item.weight)
    u = torch.tensor([0, 1]); p = torch.tensor([0, 1]); n = torch.tensor([2, 2])
    a, ar = original.bpr_loss(u, p, n)
    b, br = port.bpr_loss(graph, u, p, n)
    assert torch.allclose(a, b, atol=1e-7) and torch.allclose(ar, br, atol=1e-7)
    (a + 1e-4 * ar).backward()
    (b + 1e-4 * br).backward()
    error = float((original.embedding_item.weight.grad - port.item_embedding.weight.grad).abs().max())
    assert error < 1e-7
    return {"loss_error": abs(float(a.detach() - b.detach())), "gradient_max_error": error}


def verify_references():
    frame = corpus(["t1"])
    # A complete set of histories for a small real-data user subset, not invented interactions.
    selected = sorted(frame.userId.unique())[:8]
    small = frame[frame.userId.isin(selected)]
    result = {"memory": check_memory(small), "graph": check_graph(small),
              "data": "t1 real training histories", "users": len(selected)}
    return result


def original_ndcg(rank_list, positive):
    relative = "src/xmrec_utils/evaluation.py"
    path = BASE / "upstream/winner" / relative
    expected = json.loads(SPEC.read_text())["source_hashes"][relative]
    if sha256(path) != expected:
        raise ValueError("Pinned metric reference changed")
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"getDCG", "getNDCG"}]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return float(namespace["getNDCG"](rank_list[:10], [positive]))

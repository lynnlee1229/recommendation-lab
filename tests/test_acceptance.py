import numpy as np
import pytest
from fastapi.testclient import TestClient
from reclab.paths import ARTIFACTS, read_json
from reclab.data import Dataset
from reclab.train import load_model, model_scorer
from reclab.retrieval import topk

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "runs/baseline-history32/report.json").exists(),
    reason="Requires real-data bootstrap; run ./lab.sh setup first",
)


@pytest.fixture(scope="module")
def ds():
    return Dataset()


@pytest.fixture(scope="module")
def client():
    from reclab.service import app

    return TestClient(app, base_url="http://localhost")


def test_real_data_provenance_and_temporal_split(ds):
    m = ds.manifest
    assert m["raw_events"] == 2756101
    assert len(m["archive_sha256"]) == 64
    assert ds.train.timestamp.max() < m["train_end_ms"]
    valid = ds.events[ds.events.split == "valid"]
    assert valid.timestamp.min() >= m["train_end_ms"]
    assert valid.timestamp.max() < m["valid_end_ms"]
    assert all(
        i["category_asof_ms"] is None or i["category_asof_ms"] < m["train_end_ms"]
        for i in ds.items
    )


def test_training_samples_match_prior_real_events(ds):
    report = read_json(ARTIFACTS / "runs/baseline-history32/report.json")
    for s in report["samples"]:
        prior = ds.train[
            (ds.train.user_idx == s["user_idx"])
            & (ds.train.timestamp < s["event_time_ms"])
        ].sort_values("timestamp", kind="stable")
        assert (
            prior.item_idx.astype(int).tail(report["config"]["history_len"]).tolist()
            == s["history"]
        )
        assert s["positive_item_idx"] not in set(prior.item_idx)


def test_eval_targets_are_novel_and_in_catalog(ds):
    targets, audit = ds.evaluation()
    assert audit["evaluated_users"] > 0
    for user, items in targets.items():
        assert not items & ds.seen[user]
        assert all(0 < i < ds.n_items for i in items)


def test_live_inference_equals_offline_scorer(client, ds):
    users, _ = ds.evaluation()
    user = next(iter(users))
    model, report = load_model(ds, "baseline-history32")
    score, _ = model_scorer(ds, model, report["config"]["history_len"])
    expected = topk(score(user), 10, ds.seen[user])
    response = client.post(
        "/api/recommend",
        json={
            "user_idx": user,
            "run_id": "baseline-history32",
            "channel": "two_tower",
            "mmr_lambda": 1,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert [i["item_idx"] for i in payload["items"]] == expected
    assert payload["trace"]["stage_status"]["fine_rank"] == "not implemented"
    assert (ARTIFACTS / "traces" / (payload["trace"]["request_id"] + ".json")).is_file()


def test_api_contract_and_diagnostics(client):
    summary = client.get("/api/summary")
    assert summary.status_code == 200
    user = summary.json()["users"][0]["user_idx"]
    features = client.get(f"/api/users/{user}/features?run_id=baseline-history32")
    assert features.status_code == 200
    assert len(features.json()["user_vector"]) == 32
    d = client.get("/api/runs/baseline-history32/diagnostics").json()
    assert np.allclose(np.array(d["probabilities"]).sum(1), 1)
    assert np.allclose(np.array(d["gradients"]).sum(1), 0, atol=1e-6)
    assert d["loss"] >= 0
    assert client.get("/api/runs/missing/samples").status_code == 404
    assert client.post("/api/train", json={"dim": 100000}).status_code == 422


def test_cross_origin_training_and_rebinding_rejected(client):
    assert (
        client.post(
            "/api/train", headers={"Origin": "https://example.com"}, json={}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/train", headers={"Sec-Fetch-Site": "cross-site"}, json={}
        ).status_code
        == 403
    )
    assert (
        client.get("/api/summary", headers={"Host": "example.com"}).status_code == 400
    )


def test_course_all_files_mapped_and_status_honest(client):
    c = client.get("/api/curriculum").json()
    assert len(c["lessons"]) == 45
    assert sum(x["pages"] for x in c["lessons"]) == 902
    assert len({x["file"] for x in c["lessons"]}) == 45
    assert all(
        (__import__("pathlib").Path(c["local_source"]) / x["file"]).exists()
        for x in c["lessons"]
    )
    assert (
        next(x for x in c["lessons"] if x["file"] == "05_LastN_02.pdf")["status"]
        == "planned"
    )
    assert client.get("/api/lectures/missing.pdf").status_code == 404

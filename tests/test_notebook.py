from urllib.parse import urlsplit, parse_qs
from pathlib import Path
import nbformat
import pytest
from fastapi.testclient import TestClient
from reclab.paths import ROOT
from reclab.service import app
from reclab import notebook


@pytest.fixture
def client():
    return TestClient(app, base_url="http://localhost")


def test_source_catalog_is_real_and_whitelisted(client):
    catalog = client.get("/api/code").json()["files"]
    assert {row["id"] for row in catalog} == {
        "models",
        "data",
        "train",
        "retrieval",
        "service",
        "xmrec_data", "xmrec_memory", "xmrec_graph", "xmrec_ranking",
        "xmrec_experiment", "xmrec_metrics", "experiments", "deep_recommendation", "search", "ads", "probability", "auction", "ads_teaching",
    }
    response = client.get("/api/code/models")
    assert response.status_code == 200
    result = response.json()
    assert result["source"] == (ROOT / "reclab/models.py").read_text()
    assert "class TwoTower" in result["source"]
    assert result["editable"] is False
    assert client.get("/api/code/unknown").status_code == 404


def test_invalid_notebook_target_never_launches(client, monkeypatch):
    def fail_launch():
        pytest.fail("Invalid target should not start a process")

    monkeypatch.setattr(notebook, "ensure_server", fail_launch)
    assert (
        client.get(
            "/notebooks?target=../../outside", follow_redirects=False
        ).status_code
        == 404
    )
    with pytest.raises(ValueError):
        notebook.login_url("outside")


def test_notebook_redirect_keeps_authentication_and_fixed_destination(
    client, monkeypatch
):
    monkeypatch.setattr(
        notebook, "ensure_server", lambda: {"token": "dummy-test-token"}
    )
    response = client.get("/notebooks?target=models", follow_redirects=False)
    assert response.status_code == 303
    location = urlsplit(response.headers["location"])
    assert location.netloc == "127.0.0.1:18581"
    assert location.path == "/login"
    query = parse_qs(location.query)
    assert query["token"] == ["dummy-test-token"]
    assert query["next"] == ["/lab/tree/reclab/models.py"]
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_cross_site_cannot_start_notebook(client, monkeypatch):
    monkeypatch.setattr(
        notebook, "ensure_server", lambda: pytest.fail("Must reject before launch")
    )
    assert (
        client.get("/notebooks", headers={"Sec-Fetch-Site": "cross-site"}).status_code
        == 403
    )
    assert (
        client.get("/notebooks", headers={"Origin": "https://example.com"}).status_code
        == 403
    )


def test_notebooks_have_valid_runnable_cells_and_project_kernel():
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) >= 2
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert nb.metadata.kernelspec.name == "reclab"
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell{index}", "exec")


def test_status_does_not_expose_authentication_token(monkeypatch):
    monkeypatch.setattr(notebook, "server_info", lambda: {"token": "dummy-test-token"})
    monkeypatch.setattr(notebook, "owned_pid", lambda: 1234)
    result = notebook.status()
    assert result["ready"] is True
    assert "dummy-test-token" not in str(result)


def test_catalog_includes_every_registered_notebook(client):
    result = client.get("/api/notebooks")
    assert result.status_code == 200
    rows = result.json()["notebooks"]
    expected = {
        Path(value).name
        for value in notebook.TARGETS.values()
        if Path(value).parent == Path("notebooks") and value.endswith(".ipynb")
    }
    assert {row["file"] for row in rows} == expected
    assert len(rows) >= 11
    assert {f"{i:02}" for i in range(11)} <= {row["file"][:2] for row in rows}
    assert [row["file"] for row in rows] == sorted(expected)
    assert all(row["available"] for row in rows)
    assert rows[0]["target"] == "course"
    assert next(row for row in rows if row["target"] == "cross")["title"].startswith(
        "06 · "
    )

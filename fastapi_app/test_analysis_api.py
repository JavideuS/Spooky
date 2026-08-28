"""
Smoke coverage for the read-only /v1/analysis/* endpoints (analysis_api.py).

These need at least one aggregatable sweep under SPOOKY_BENCHMARKS_DIR
(default <repo>/results/sweeps). That tree isn't committed, so every test
skips cleanly when it's absent rather than failing on a fresh checkout / CI.
"""

import pytest
from fastapi.testclient import TestClient

from api import app
from analysis_api import _sweep_dir_map


def _first_sweep_id():
    ids = list(_sweep_dir_map().keys())
    if not ids:
        pytest.skip("no sweeps under SPOOKY_BENCHMARKS_DIR")
    # newest-first, matching the /sweeps ordering
    with TestClient(app) as client:
        return client.get("/v1/analysis/sweeps").json()["sweeps"][0]["sweep_id"]


def test_list_sweeps_shape():
    with TestClient(app) as client:
        body = client.get("/v1/analysis/sweeps").json()
    assert "sweeps" in body and body["sweep_count"] == len(body["sweeps"])
    for entry in body["sweeps"]:
        assert entry["sweep_id"]
        assert set(entry) >= {"solvers", "grid_sizes", "n_completed", "git_commit"}


def test_summary_and_recommend_agree_on_solvers():
    sweep_id = _first_sweep_id()
    with TestClient(app) as client:
        summary = client.get(f"/v1/analysis/sweeps/{sweep_id}/summary")
        assert summary.status_code == 200, summary.text
        assert isinstance(summary.json()["statistical_tests"], list)

        rec = client.get(f"/v1/analysis/sweeps/{sweep_id}/recommend")
        assert rec.status_code == 200, rec.text
        body = rec.json()
        assert body["solvers"], "recommend returned no solvers for the whole sweep"
        # sorted best-first: success rate descending
        rates = [s["success_rate"] for s in body["solvers"]]
        assert rates == sorted(rates, reverse=True)
        assert body["verdict"]


def test_plots_return_plotly_json():
    sweep_id = _first_sweep_id()
    with TestClient(app) as client:
        for name in ("scaling", "success_rate", "variable_reduction"):
            res = client.get(f"/v1/analysis/sweeps/{sweep_id}/plots/{name}")
            assert res.status_code == 200, (name, res.text)
            assert "data" in res.json()
        assert (
            client.get(f"/v1/analysis/sweeps/{sweep_id}/plots/bogus").status_code == 404
        )


def test_unknown_sweep_404():
    with TestClient(app) as client:
        assert (
            client.get("/v1/analysis/sweeps/does_not_exist/summary").status_code == 404
        )

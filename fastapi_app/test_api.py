"""
Regression pin for the redundant-full-grid-build bug: /v1/plan used to call
builder.build() with no BFS-reduced window active (_active_cells still None)
right before solver.solve(), which immediately rebuilds it properly anyway —
free on tiny synthetic maps, but apply_one_hot()'s O(cells²) constraint loop
made it dominate wall time on real-sized maps (see builder/base_qubo.py's
_warn_if_unrestricted_build(), added alongside the fix).

This needs a map above _warn_if_unrestricted_build()'s 200-cell threshold —
no_obs50x50 (2,500 cells) — even though the robot's own path only needs a
tiny window: the bug was building the *whole grid* regardless of how short
the requested path was, so a small map would never actually exercise it.
The fast classical solver keeps this quick with no GPU/real hardware needed.
"""

import pytest
from fastapi.testclient import TestClient

from api import app


def test_plan_does_not_trigger_unrestricted_build(capsys):
    with TestClient(app) as client:
        response = client.post(
            "/v1/plan",
            json={
                "map_id": "no_obs50x50",
                "solver": "dwave.fast",
                "format": "grid",
                "robots": [{"start": [2, 0], "goal": [0, 2]}],
                "penalty_set": "crash",
            },
        )

    assert response.status_code == 200, response.text

    # The actual regression pin: this warning only fires when build() ran
    # over the full grid instead of a BFS-reduced window — i.e. a
    # builder.build() call snuck back in before solver.solve() again.
    captured = capsys.readouterr()
    assert "unrestricted QUBO" not in captured.out


def test_cbs_profile_plans_on_grid_and_graph():
    """classic.cbs must route to the CBS builders in both /v1/plan formats —
    the endpoints used to branch ilp-vs-QUBO only, so a CBS profile fell
    through to a QUBOBuilder the CBSSolver can't consume."""
    with TestClient(app) as client:
        assert "classic.cbs" in client.get("/solvers").json()

        grid = client.post(
            "/v1/plan",
            json={
                "map_id": "obs5x5_hard",
                "solver": "classic.cbs",
                "format": "grid",
                "robots": [
                    {"id": "r0", "start": [0, 0], "goal": [4, 4]},
                    {"id": "r1", "start": [4, 0], "goal": [0, 4]},
                ],
            },
        )
        assert grid.status_code == 200, grid.text
        body = grid.json()
        assert body["solver_used"] == "classic.cbs"
        assert len(body["paths"]) == 2
        assert body["cost"] == body["cost"]  # finite, not NaN/inf

        graph = client.post(
            "/v1/plan",
            json={
                "map_id": "obs5x5_hard",
                "solver": "classic.cbs",
                "format": "graph",
                "robots": [{"start": [0, 0], "goal": [4, 4]}],
            },
        )
        assert graph.status_code == 200, graph.text


def test_v1_map_detail_exposes_geo_metadata():
    """GET /v1/maps/{map_id} forces a lazy load and surfaces the grid's
    real-world frame — resolution (m/cell) + origin pose (x, y, yaw) — which is
    what /v1/plan's "world" coordinate_format converts against. Pins the JSON
    shape; synthetic maps carry the 1.0 / (0, 0, 0) defaults."""
    with TestClient(app) as client:
        r = client.get("/v1/maps/no_obs3x3")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["map_id"] == "no_obs3x3"
        assert body["loaded"] is True
        assert body["has_grid"] is True
        assert isinstance(body["resolution"], (int, float))
        assert isinstance(body["origin"], list) and len(body["origin"]) == 3

        assert client.get("/v1/maps/does-not-exist").status_code == 404


def test_v1_maps_list_carries_resolution_key_once_loaded():
    """The list entry gains resolution/origin after the map is loaded (null
    before) — GET /v1/maps/{map_id} above is what triggers that load."""
    with TestClient(app) as client:
        client.get("/v1/maps/no_obs3x3")  # force load
        entry = client.get("/v1/maps").json()["maps"]["no_obs3x3"]
        assert entry["loaded"] is True
        assert entry["resolution"] == 1.0
        assert entry["origin"] == [0.0, 0.0, 0.0]


def _plan(client, robots, solver="classic.cbs"):
    return client.post(
        "/v1/plan",
        json={"map_id": "obs5x5_hard", "solver": solver, "format": "grid", "robots": robots},
    )


@pytest.mark.parametrize("solver", ["classic.cbs", "classic.ilp"])
def test_fixed_path_keeps_committed_steps_and_the_optimum(solver):
    """Re-planning with r0 committed to (part of) its own plan returns that
    plan for r0 and the same total cost -- what clew sends to re-plan only a
    deadlocked group around robots that keep going."""
    robots = [
        {"id": "r0", "start": [0, 0], "goal": [4, 4]},
        {"id": "r1", "start": [4, 0], "goal": [0, 4]},
    ]
    with TestClient(app) as client:
        free = _plan(client, robots, solver)
        assert free.status_code == 200, free.text
        r0_path = next(p["path"] for p in free.json()["paths"] if p["robot_id"] == "r0")
        for cut in (3, len(r0_path)):
            committed = [dict(robots[0], fixed_path=r0_path[:cut]), robots[1]]
            resp = _plan(client, committed, solver)
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["cost"] == free.json()["cost"]
            got = next(p["path"] for p in body["paths"] if p["robot_id"] == "r0")
            assert got[:cut] == r0_path[:cut]


def test_fixed_path_on_qubo_and_conflicts():
    with TestClient(app) as client:
        robot = {"id": "r0", "start": [0, 0], "goal": [0, 2], "fixed_path": [[0, 0], [0, 1]]}
        resp = _plan(client, [robot], solver="dwave.general")
        assert resp.status_code == 200, resp.text
        assert resp.json()["paths"][0]["path"][:2] == [[0, 0], [0, 1]]

        clash = [
            {"id": "r0", "start": [0, 0], "goal": [0, 2], "fixed_path": [[0, 0], [0, 1]]},
            {"id": "r1", "start": [0, 2], "goal": [0, 0], "fixed_path": [[0, 2], [0, 1]]},
        ]
        resp = _plan(client, clash)
        assert resp.status_code == 400 and "r0/r1 at t=1" in resp.text, resp.text

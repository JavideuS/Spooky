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

"""
The pre-processing mode vocabulary in quantum.utils.preprocess.

These are pure table lookups, so they run instantly and need no solver, no
builder and no map — but getting them wrong mislabels an entire sweep, which
is expensive to discover later.
"""

import pytest

from quantum.utils import preprocess as pm


def test_modes_map_to_the_right_stages():
    """The whole point of the modes is that BFS pruning and numerical fixing
    become independently selectable; `raw` and `full` must still reproduce the
    old False/True exactly."""
    expected = {
        #  mode              windowed  bfs variant   numeric
        pm.RAW:             (False,   None,          False),
        pm.BFS_AGGRESSIVE:  (True,    "aggressive",  False),
        pm.BFS_SAFE:        (True,    "safe",        False),
        pm.FULL:            (True,    "aggressive",  True),
        pm.FULL_SAFE:       (True,    "safe",        True),
    }
    assert set(expected) == set(pm.MODES)
    for mode, (windowed, bfs, numeric) in expected.items():
        assert pm.uses_windowed_pipeline(mode) is windowed, mode
        assert pm.bfs_variant(mode) == bfs, mode
        assert pm.applies_numeric_reduction(mode) is numeric, mode


def test_legacy_booleans_still_resolve():
    """index.json, manifest.json and every pre-existing sweep config record
    booleans. Those sweeps must stay runnable and re-aggregatable."""
    assert pm.normalize(True) == pm.FULL
    assert pm.normalize(False) == pm.RAW
    assert pm.normalize("full") == pm.FULL
    assert pm.normalize(" BFS_Safe ") == pm.BFS_SAFE


def test_rejects_unknown_modes():
    """A typo in a sweep config must fail at parse time, not silently fall
    back to a default and produce a mislabelled 300-run sweep."""
    with pytest.raises(ValueError, match="Unknown preprocess mode"):
        pm.normalize("bfs_safeish")
    with pytest.raises(TypeError):
        pm.normalize(2)


def test_exact_solvers_get_a_bool_that_is_false_only_for_raw():
    """ILP and CBS pass this straight into builder.build(preprocess=...) as a
    bool. Handing them the mode string instead would be silently wrong: "raw"
    is a non-empty string and therefore truthy, so they would prune in the one
    mode that must not."""
    assert pm.applies_bfs_pruning(pm.RAW) is False
    for mode in (pm.BFS_AGGRESSIVE, pm.BFS_SAFE, pm.FULL, pm.FULL_SAFE):
        assert pm.applies_bfs_pruning(mode) is True, mode

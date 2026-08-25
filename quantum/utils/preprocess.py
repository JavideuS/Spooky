"""
Pre-processing modes for the windowed QUBO pipeline.

Pre-processing is two independent stages, and the old boolean could only
express two of their four combinations:

  1. structural BFS pruning  -- which cells a robot may occupy per timestep
  2. numerical reduction     -- pinning individual variables to 0/1 from
                                their diagonal/individual coefficients

`preprocess=True` meant "aggressive BFS + numerical", `preprocess=False`
meant neither *and* dropped the whole windowed loop (no correction retries,
no forced-collision flagging) for a separate simpler implementation. So the
two settings differed in three ways at once and no measurement could
attribute a result to any one of them.

The modes below separate the stages. `raw` and `full` reproduce the old
False/True exactly, so existing configs and recorded sweeps keep working.

  raw            the old preprocess=False: the simple sampling loop, no
                 pruning, no pinning, no correction retries. A different
                 algorithm rather than "the same one with reductions off" --
                 do not read it as a clean no-preprocessing control.
  bfs_aggressive windowed loop, non-backtracking BFS, no numerical pinning.
  bfs_safe       windowed loop, monotone BFS (staying in place and revisiting
                 both allowed), no numerical pinning. The only mode whose
                 pruning cannot exclude a feasible solution -- and therefore
                 the only one where a QUBO-vs-ILP comparison is like-for-like,
                 since ILP/CBS use exactly these semantics via
                 bfs_reachable_sets(). Costs roughly 10-19x the variables.
  full           the old preprocess=True: aggressive BFS + numerical.
  full_safe      monotone BFS + numerical. Measured to keep the numerical
                 pass fully active (the coefficient trigger fires on 60-100%
                 of timestep groups under either BFS), so this is a real
                 configuration and not a no-op variant of bfs_safe.
"""

# BFS variants a mode can select, consumed by BaseQUBO.reachable_for_window()
BFS_VARIANT_AGGRESSIVE = "aggressive"
BFS_VARIANT_SAFE = "safe"

RAW = "raw"
BFS_AGGRESSIVE = "bfs_aggressive"
BFS_SAFE = "bfs_safe"
FULL = "full"
FULL_SAFE = "full_safe"

# One row per mode, one column per stage. Adding a mode means adding a row
# here and nothing else; the predicates below are all lookups into it.
#   windowed  run the windowed loop (correction retries, forced-collision
#             flagging) rather than the separate simple sampling loop
#   bfs       which reachability policy prunes the window, None for no pruning
#   numeric   run reduce_diag_fixed_vars_iterative()
# fmt: off  (the column alignment is the point; black would collapse it)
_MODES = {
    RAW:            {"windowed": False, "bfs": None,                   "numeric": False},
    BFS_AGGRESSIVE: {"windowed": True,  "bfs": BFS_VARIANT_AGGRESSIVE, "numeric": False},
    BFS_SAFE:       {"windowed": True,  "bfs": BFS_VARIANT_SAFE,       "numeric": False},
    FULL:           {"windowed": True,  "bfs": BFS_VARIANT_AGGRESSIVE, "numeric": True},
    FULL_SAFE:      {"windowed": True,  "bfs": BFS_VARIANT_SAFE,       "numeric": True},
}
# fmt: on

MODES = tuple(_MODES)


def normalize(value) -> str:
    """Coerce a mode from a config, a CLI flag, or a legacy boolean.

    True/False are still accepted because index.json, manifest.json and every
    sweep config written before the modes existed record booleans, and those
    sweeps must stay re-aggregatable.
    """
    if value is True:
        return FULL
    if value is False:
        return RAW
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in _MODES:
            return candidate
        raise ValueError(
            f"Unknown preprocess mode {value!r}. Expected one of {list(MODES)} "
            "(or a legacy True/False)."
        )
    raise TypeError(
        f"preprocess must be one of {list(MODES)} or a bool, got {type(value).__name__}"
    )


def uses_windowed_pipeline(mode) -> bool:
    """False only for `raw`, which runs the separate simple sampling loop."""
    return _MODES[normalize(mode)]["windowed"]


def bfs_variant(mode):
    """Which reachability policy to prune with, or None under `raw`."""
    return _MODES[normalize(mode)]["bfs"]


def applies_numeric_reduction(mode) -> bool:
    return _MODES[normalize(mode)]["numeric"]


def applies_bfs_pruning(mode) -> bool:
    """Whether any BFS reachability pruning happens at all.

    This is the form the exact solvers want. ILP and CBS take a plain bool
    into builder.build(preprocess=...) and have no aggressive/safe choice --
    bfs_reachable_sets() is already the monotone variant -- so every mode
    except `raw` is simply "prune". Passing them a mode string directly would
    be silently wrong: "raw" is a non-empty string and therefore truthy, so
    they would prune in the one mode that must not.
    """
    return bfs_variant(mode) is not None

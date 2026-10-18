# Clearance: notation and constraint design

Reference for the `robot_radius` / `inflation` clearance feature — the math, the
per-robot reasoning behind it, and how it maps onto CBS / ILP / (planned) QUBO.
Code lives in `quantum/utils/clearance.py`, `RobotConfig`, and
`PathfindingProblem.get_clearance_table()` / `get_obstacle_keepout()`.

## 1. Per-robot quantities

Each robot carries two scalars, metres:

- $r_a$ — **body radius** (`robot_radius`), physical, public/observable.
- $\iota_a$ — **inflation** (`inflation`), a safety margin. Private: only $a$
  itself needs to know its own value to enforce its own rule (see §3).

Its own obstacle keep-out (`RobotConfig.clearance`) is

$$
c_a = r_a + \iota_a .
$$

## 2. Baseline constraints (point robots)

$$
\sum_{a \in \mathcal A_t} x_{a,v,t} \;\le\; 1
\qquad \forall v \in V,\ \forall t \in \mathcal T
$$

$$
x_{a_i,v,t} + x_{a_j,w,t} + x_{a_i,w,t+1} + x_{a_j,v,t+1} \;\le\; 3
\qquad \forall (v,w) \in E,\ \forall a_i,a_j \in \mathcal A_t,\ i<j,\ \forall t \in \mathcal T^-
$$

Crash forbids two robots sharing a cell at the same instant. Swap forbids the
exact two-way exchange $v \leftrightarrow w$ across one step — nothing else.
It explicitly permits $b$ vacating a cell and $a$ moving into it a step later:
with zero-extent points, occupancy is instantaneous and exact, so there is no
such thing as "partway vacated" — a legal move, not a gap in the check.

## 3. Robot-robot separation, derived from two independent local rules

**Do not compose it as "$a$'s full clearance vs. $b$'s full clearance"** —
that double-counts both margins ($r_a+r_b+\iota_a+\iota_b$, strictly more
separation than necessary). The right composition falls out of two rules
each robot can state **using only its own inflation and the other's public
body radius** — no robot ever needs to know the other's private margin:

> Robot $a_i$'s rule: keep $a_j$'s *body* out of $a_i$'s *own* clearance disc.
$$
\operatorname{cheb}(u_{a_i}, u_{a_j}) \;>\; c_{a_i} + r_{a_j} \;=\; r_{a_i}+\iota_{a_i}+r_{a_j}
$$

> Robot $a_j$'s symmetric rule:
$$
\operatorname{cheb}(u_{a_i}, u_{a_j}) \;>\; c_{a_j} + r_{a_i} \;=\; r_{a_j}+\iota_{a_j}+r_{a_i}
$$

Both must hold — the conjunction is the *larger* of the two right-hand
sides, so the pairwise requirement is

$$
\operatorname{cheb}(u_{a_i}, u_{a_j}) \;>\; R_{a_ia_j}
\;:=\; r_{a_i} + r_{a_j} + \lambda\max(\iota_{a_i}, \iota_{a_j})
$$

($\lambda \ge 1$: slack for odometry/lidar error, `DEFAULT_SEPARATION_FACTOR = 1.0`).
The $\max$ is not a comparison either robot performs — it is what enforcing
both independently-stated rules reduces to once *the solver* (which has
full knowledge of the whole instance) combines them into one number.
**Checking body-vs-body on both sides is what reproduces this formula** —
checking clearance-vs-clearance instead gives the over-separating additive
model; checking clearance-vs-*leader-point* instead under-counts the other
robot's body and is unsafe once both bodies have real size.

Discretised at resolution $\Delta$ (strict, so the default configuration is
a no-op — see §6):

$$
D_{a_ia_j} \;=\; \Big\{\, d \in \mathbb Z^2 \;:\; \|d\|_\infty \le \big\lceil R_{a_ia_j}/\Delta \big\rceil - 1 \,\Big\}
$$

Call $D_{a_ia_j}$ the **clearance shell** for the pair (`_clearance_shell` in
`QUBOBuilder.py`; `D`/`clearance_table` elsewhere) — the neighbourhood of
offsets around a leader cell that counts as an overlap. A pair $(a_i,a_j)$
whose clearance shell is bigger than $\{\mathbf 0\}$ is a **clearance robot
pair**: it needs the generalised (shell) treatment rather than plain
same-cell matching (`_clearance_robot_pairs` in CBS, `clearance_robot_pairs`
in ILP). These are two different things — the shell is a set of *cell
offsets*, a clearance robot pair is a set of *robot-id pairs* — worth
keeping distinct since they're easy to conflate.

For circular robots $D_{a_ia_j}$ is always just a disc from this one scalar
formula. A non-circular `robot_footprint`
would need each robot's own offset set $S_k$ and the general (asymmetric —
subscript order matters) Minkowski difference $D_{a_ia_j} = S_{a_j} \oplus (-S_{a_i})$,
tested as $(\text{pos}_{a_i}-\text{pos}_{a_j}) \in D_{a_ia_j}$. For now we keep it simplified with circular for all robots and taking largest radius side.

## 4. Crash generalises cleanly — one equation, no exception

$$
x_{a_i,v,t} \;+\!\! \sum_{\substack{w \in V \\ v-w \,\in\, D_{a_ia_j}\setminus\{\mathbf 0\}}} \!\! x_{a_j,w,t}
\;\le\; 1
\qquad \forall v \in V,\ \forall a_i,a_j \in \mathcal A_t,\ i<j,\ \forall t \in \mathcal T
$$

$D_{a_ia_j} = \{\mathbf 0\}$ recovers the pairwise original *exactly* — this
is one family, not two. (Implementation keeps the compact global
$\sum_a x_{a,v,t}\le 1$ for the $\{\mathbf 0\}$ portion because it is a
strictly tighter LP for $n>2$ robots — an efficiency footnote, not a
different model. `model.crash` + `model.crash_clearance` in `ILPBuilder`.)

## 5. Swap does **not** generalise into the extra hazard — they're different constraints

There are two distinct ways to widen the exact-swap idea, and only one of
them is actually "swap, generalised":

**Swap, faithfully generalised (the reciprocal-crossing test).** Require
*both* endpoints to nearly match, the way the original does:

$$
x_{a_i,v,t}+x_{a_j,w,t}+x_{a_i,v',t+1}+x_{a_j,w',t+1} \;\le\; 3
$$
$$
\text{whenever } (v'-w)\in D_{a_ia_j} \ \text{and}\ (w'-v)\in D_{a_ia_j}
$$

At $D_{a_ia_j}=\{\mathbf 0\}$ this forces $v'=w$ and $w'=v$ — exactly the
original. **This is swap's whole, correctly narrow job — reciprocal
exchange — done right, and it needs no separate "clearance version".**

**Trailing (a new constraint, not a swap variant).**

$$
x_{a_i,v,t+1}+\!\!\sum_{\substack{w\in V\\ v-w\,\in\,D_{a_ia_j}}}\!\!x_{a_j,w,t}\ \le\ 1,
\qquad
x_{a_i,v,t}+\!\!\sum_{\substack{w\in V\\ v-w\,\in\,D_{a_ia_j}}}\!\!x_{a_j,w,t+1}\ \le\ 1
$$
$$
\forall v\in V,\ \forall a_i,a_j\in\mathcal A_t\cap\mathcal A_{t+1},\ i<j,\ \forall t\in\mathcal T^-
$$

At $D_{a_ia_j}=\{\mathbf 0\}$ this forbids $a_i$ from ever entering a cell
$a_j$ currently occupies, *regardless of whether $a_j$ then leaves* — it
does **not** reduce to the original swap at the trivial case; it is
strictly stronger even there, which is exactly why it's gated to clearance pairs
only, never applied at $D=\{\mathbf 0\}$.

**Why this hazard has no point-robot analogue.** With zero-extent points,
occupancy is instantaneous and exact — there is no state of "partway
vacating" a cell, so there is nothing for trailing to protect against; that
is precisely why classic MAPF (CBS, PBS, …) never checks it. The instant a
robot has real extent, and the instant two robots don't execute their
$t\to t+1$ transition at exactly the same physical moment, that gap becomes
real: robot $j$ may still be physically occupying (or only just vacated)
its old cell — continuously, over nonzero time — when robot $i$'s body
arrives. No real, wheeled robot teleports between cells, so this condition
(continuous, non-instantaneous, imperfectly-synchronised transitions) is not
a special case to opt into — it is the default physical situation. It is
also not rare in practice: it is exactly the shape of the most common
multi-robot traffic pattern — one robot trailing/queuing closely behind
another through a shared corridor.

**Trailing strictly dominates exact swap once a pair is a clearance pair.** $a_i$
arriving exactly where $a_j$ started ($v'=w$) already trips the first
disjunct ($v'-w=\mathbf 0 \in D_{a_ia_j}$ always), so every case exact swap
would catch, trailing catches too — no gap is left by dropping the plain
swap check for clearance pairs (`model.swap` skips clearance pairs in `ILPBuilder`;
CBS's `_all_conflicts` only evaluates the trivial exact-swap predicate for
non-clearance pairs, and trailing for clearance ones).

**Each disjunct only needs two position variables**, not all four (one
robot's endpoint at one time, the other's at the other time) — so no
transition/adjacency enumeration is needed to encode it; banning the pair
directly is exact, and tighter than enumerating $(u,u',w,w')$ tuples.

## 6. No-op at defaults, verified

Strict `<` in $D_{a_ia_j}$'s definition means default `robot_radius=0.5`, no
`inflation`, unit resolution $\Rightarrow$ $R_{a_ia_j}=1.0 \Rightarrow$
$D_{a_ia_j}=\{\mathbf 0\}$ for every pair: exact same-cell crash, exact swap,
no obstacle keep-out, no trailing — byte-identical to the pre-clearance
model. Confirmed by the CBS/ILP energy-equivalence tests staying unchanged.

## 7. Software map

| concept | CBS (`cbs_algorithm.py`) | ILP (`ILPBuilder.py`) | QUBO (`QUBOBuilder.py`) |
|---|---|---|---|
| crash | `_overlap` same-time vertex conflict | `model.crash` (+ `model.crash_clearance` for clearance pairs) | `apply_crash_penalty`, clearance-shell generalised (`_clearance_shell`) |
| swap (reciprocal exchange) | exact-swap predicate, non-clearance pairs only | `model.swap`, clearance pairs skipped (dominated, see §5) | no tier — a genuine AND-of-both-endpoints condition needs 4 variables jointly, not expressible as a quadratic term without ancillas (see §9) |
| trailing (deployment safeguard) | crossing predicate, clearance pairs only | `model.trailing` | `apply_trailing_penalty` / `K_trail` (renamed from `apply_swap_penalty` / `K_swap` — see §9), same pairwise-product shape, D_ab-shell generalised |
| obstacle keep-out | `GridCBSBuilder` folds into `legal_cells` | fixes `x=0` | not yet wired to `get_obstacle_keepout()` — QUBO already has its own, different (soft potential-field, `P_obs`) obstacle mechanism, so this needs its own design pass rather than reusing CBS/ILP's hard-mask approach |

`is_solution_valid()` (`quantum/benchmark/benchmark.py`) reports these as
four independent reasons: `vertex_conflict`, `swap_conflict`,
`clearance_conflict` (crash, generalised), `trailing_conflict` — so a failed
validation says which of the four actually fired.

**Known gap, split into two — one closed, one still open**: `base_qubo.py`'s
windowed BFS/diagonal-fixing preprocessing (`_collides_with_fixed_rival`,
`_collides_with_swap_rival`) is exact-cell/exact-swap only, not
clearance-shell aware. When it fully determines a window on its own ("fully
pre-processed, skipping solver"), two robots can still end up committed to
shell-violating positions without `K_crash`/`K_trail` ever getting a say —
confirmed pre-existing (reproduces identically at default, no-clearance
radius), not introduced by this generalisation. **Still open**: preprocessing
itself doesn't prevent this — fixing that means teaching
`_collides_with_fixed_rival`/`_collides_with_swap_rival` the shell test,
which touches real fixing/solving behaviour in an already-delicate ~250-line
subsystem, deliberately deferred. **Closed (2026-09-13)**: the *attribution*
half. `BaseSolver._flag_forced_collisions` now also detects clearance/
trailing-shell forced conflicts (same-time and cross-time, plus the
inactive-handoff case), tagging every entry with `"kind"`:
`"vertex"`/`"clearance"`/`"trailing"`. `benchmark.py`'s
`_attribute_invalid_cause()` cross-references all three kinds against
`is_solution_valid()`'s matching conflict lists (matched on kind + cells +
time, not on robot identity — the two sides use different robot
representations, number vs. id). So a shell violation forced by
preprocessing is now correctly reported as `"pre_processing"`, not
misattributed to `"solver_sampling"` — purely diagnostic, doesn't change
what preprocessing actually fixes. Test:
`test_forced_clearance_violation_attributed_to_preprocessing` in
`test_qubo_clearance.py`.

## 8. Open question: is trailing worth it here?

Trailing costs real rows/qubits (it's the more expensive of the two
clearance constraints — see the ILP cost notes in the design log). Whether
to pay for it is a deployment-assumption question, not a "does it happen
often" one:

- If robots execute $t\to t+1$ atomically and in perfect lockstep, trailing
  protects against a state that provably cannot occur — drop it.
- If robots move continuously (they do — no real robot teleports) or two
  robots' steps can complete at different real-world instants (they can —
  independent controllers, no perfect synchronisation), trailing is
  protecting against a real hazard, not a theoretical one.

For real deployment (the reason this feature exists at all — see the
Fraunhofer fleet-coordinator log that started this thread) the second case
applies, so trailing is recommended on by default. It should be stated as an
explicit assumption in the paper either way: *"agents execute steps with
independent timing; trailing guards against one agent entering a cell the
other has only just vacated."*

## 9. Terminology note for the next preprint revision

Two names changed after the first preprint, both worth carrying into any
revision:

- **`K_swap` → `K_trail`.** This isn't a rename for polish — the first
  preprint's `K_swap` never computed swap. A quadratic-only penalty cannot
  express swap's genuine AND-of-both-endpoints condition (that needs 4
  variables jointly, i.e. ancillas) without leaving quadratic form, so the
  pairwise-product terms under that name always computed trailing's
  OR-of-either-endpoint predicate instead (see §5, §7's row for why QUBO has
  no exact-swap tier at all, unlike CBS/ILP). The formula and the code were
  always right; only the label was wrong. Anywhere the first preprint says
  `K_swap`, read it as `K_trail`.
- **"fat pair" → "clearance robot pair"** (via an intermediate "clearance
  pair," settled after it read as ambiguous with the shell below). A pair
  $(a_i,a_j)$ whose clearance shell $D_{a_ia_j}$ is bigger than
  $\{\mathbf 0\}$ — i.e. clearance is actively non-trivial for it.
- **$D_{a_ia_j}$ named the "clearance shell"** — the neighbourhood of cell
  offsets around a leader cell, as distinct from a clearance robot pair (a
  set of robot ids, not cell offsets) above.

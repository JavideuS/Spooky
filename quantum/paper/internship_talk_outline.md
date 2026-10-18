# Internship Final Talk — Outline (~15 min, general audience)

Framing: audience is not deep in robotics *or* quantum computing.
Light on notation, heavy on intuition, and don't be afraid to state the honest
limitations (be real, don't exaggerate/oversell).

---

## 1. Hook (~1 min)

Multi-robot support in ROS2/nav2 is allowed but not meant for it, almost every tool
in the ecosystem (planners, controllers, costmaps) is built assuming one
robot. 

Coordinating a fleet in practice is painful. That's where this started:
I wanted to solve a robotics problem that could relate to quantum computing.
It happened to be multi-robot path planning, an area I have approached before and
was a complete failure and hard to make it work.

---

## 2. The problem: Multi-Agent Path Finding (~1.5 min)

- Given N robots with start/goal positions on a shared map, find
  collision-free paths minimizing total time.
- Two traditional families:
  - **Decomposition** — plan robots individually, resolve conflicts
    iteratively (e.g. Conflict-Based Search / CBS).
  - **Joint** — solve all robots at once as one combinatorial problem.
- Both are NP-hard in the worst case.
CBS scales poorly, while QUBO scales linearly with number of robots but
finding the ground state is still NP-hard.

---

## 3. Spooky: reframing MAPF as QUBO (~2.5–3 min)

- Idea: the joint MAPF problem is written as a Quadratic Unconstrained Binary
  Optimization — the native input format for quantum annealers and QAOA,
  but also solvable classically (simulated annealing) and exactly via ILP.
- **One formulation, several solver families** targeting the same energy
  landscape: D-Wave-style annealing, QAOA, classical SA.
- Variables encode `(robot, time, position)`; constraints (one robot per
  cell, only move to neighbors, start/goal, no collisions) become penalty
  terms in the energy function.
- Framing for this audience: *"one universal formulation, pick your
  solver"* — not "quantum solves this," but "quantum becomes an option,
  on equal footing with classical."
- Published as arXiv:2602.14799 — this was the **input** to the
  internship, not the output.

---

## 4. What this internship added: putting it to the test (~3.5–4 min)

The optimization/quantum research core — what changed since the paper.
(Original plan included SQD; dropped early as infeasible for this problem
size/timeline — mention in one sentence, don't dwell.)

**a. Binary encoding — explored, dead end, documented.**
- Motivation: one-hot uses one qubit per `(robot, time, position)`; binary
  encoding could in theory need far fewer qubits.
- Why it failed: the adjacency constraint ("only move to a neighboring
  cell") has no clean closed-form penalty under binary encoding — it would
  need to be re-derived per graph topology, which kills the generality
  that makes the formulation useful in the first place.
- Worth stating as a negative result: it marks the boundary of the
  approach, not a gap in the internship.

**b. "Trailing" penalty (formerly "swap") — formulation refinement.**
- The true swap-collision condition (two robots crossing the same edge in
  opposite directions) is a degree-4 term — expensive to encode in QUBO.
- Replaced with a quadratic proxy; renamed because "trailing" more
  accurately describes what it penalizes — a soft, partial deterrent, not
  an exact swap constraint.

**c. IQM hardware support + better quantum performance metrics.**
- Extended solver backends to real IQM quantum hardware, beyond
  D-Wave/simulated annealing.
- Added time-to-solution comparisons via gate count and CLOPS (Circuit
  Layer Operations Per Second) — lets you compare quantum backends
  without needing exclusive wall-clock access to hardware.

**d. ILP + CBS as classical baselines.**
- Added an exact ILP formulation of the same problem — apples-to-apples
  against QUBO (both joint/global; ILP exact, QUBO targets quantum-style
  solvers).
- Added CBS as the best-in-class classical decomposition baseline.
- Together: an honest answer to "is QUBO/quantum actually competitive
  here, and against what?"

---

## 5. From formulation to fleet: the real demonstration (~4–5 min)

The robotics integration core.

- The gap: Spooky outputs a symbolic, discretized, per-timestep path per
  robot. A real robot needs continuous smooth trajectories and has to
  tolerate imperfect localization, controller behavior, timing drift.
- **fleet-coordinator** (ROS2/nav2 package, finished and working): calls
  Spooky once for the whole fleet, turns the symbolic step ordering into a
  release schedule — a robot is held at the edge of a shared conflict zone
  until the robot ahead has cleared it by a safety margin — then hands
  each robot's currently-released path to nav2's own `FollowPath`
  (smoothed first). No custom controller, no MPC: Spooky decides the
  *order*, nav2 does the *driving*.
- Deliberately decoupled: each robot keeps its own AMCL/costmap stack
  untouched; the coordinator only gates *when* it's allowed to move.
- **argOS**: the visualization/interaction layer — natural-language
  control + live 3D view — being extended to show the fleet plan running,
  in sim and on the real robots. *(status: finishing today; demo videos —
  sim + real hardware — to follow)*
- Play the recorded demo(s) here.

---

## 6. Results & honest limitations (~1.5 min)

- Benchmark takeaway: **[fill in with actual QUBO vs ILP vs CBS numbers —
  which wins, at what problem size]**
- State plainly: quantum hardware doesn't beat classical yet at these
  problem sizes; binary encoding was a dead end; task allocation (which
  robot goes where) is still out of scope — this is path planning only,
  not assignment.

---

## 7. Close (~0.5–1 min)

- Throughline: a practical ROS2 pain point led to a general
  joint-optimization reformulation of MAPF, which turned out to be a
  natural fit for quantum solvers as *one option among several* — and this
  internship proved it can drive an actual robot fleet, not just a
  benchmark script.
- What's next: **[web demonstrator from the original plan / task
  allocation layer / whatever you want to flag]**

---

## Open items for you to fill in / decide

- [ ] Actual benchmark numbers or plots for QUBO vs ILP vs CBS (where does
      each win?) — this is the one hard data slide the talk is missing.
- [ ] Whether to mention SQD at all (one sentence, "explored, dropped
      early") or skip it entirely.
- [ ] Once demo videos are recorded: which clips, and how long each gets.
- [ ] Whether section 4 is too dense for 15 min general audience — could
      compress (b) and (c) into one slide if time is tight.

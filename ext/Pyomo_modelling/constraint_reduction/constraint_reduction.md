## BLP Coefficient reductions 
- binary linear constraint (**with integer coefficients**) of form $$\sum_{i} a_i x_i \leq a_0$$ with $a_1 \geq a_2 \geq \dots \geq a_n \geq 0$
- **Goal:** find equivalent constraint $$\sum_{i} b_i x_i \leq b_0$$ with minimal $b_0$
- equivalent means same satisfying binary assignments
- Following method is from [Coefficient reduction for inequalities in 0–1 variables](https://link.springer.com/article/10.1007/BF01585527)

1. Determine ceilings $C$ and roofs $R$
    - **Ceiling** $c$ is subset of indicies $c \subset [n]$, s.t. 
        - $$\sum_{i \in c} a_i \leq a_0$$
        - if $t \not \in c$, then $$\sum_{i \in c} a_i + a_t \geq a_0 + 1$$
        - if $t \not \in c$ and $t+1 \in c$, then $$\sum_{i \in c} a_i + a_t - a_{t+1} \geq a_0 + 1$$
    - **Roof** $r$ is subset of indicies $r \subset [n]$, s.t.
        - $$\sum_{i \in r} a_i \geq a_0 + 1$$
        - if $t \in r$, then $$\sum_{i \in r} a_i - a_t \leq a_0$$
        - if $t \in r$ and $t+1 \not \in r$, then $$\sum_{i \in r} a_i - a_{t} + a_{t+1} \leq a_0$$
    - set of all ceilings $C$ and set of all roofs $R$ can be computed with method described in the paper
    - in the worst case, there is an **exponential number of roofs and ceilings**
2. Solve ILP:
$$
    \begin{aligned}
        &\textnormal{min} \quad && b_0, \\
        &\textnormal{subject to} \quad && \sum_{i \in c} b_i \leq b_0 \quad && \textnormal{for all } c \in C  \\
        &&& \sum_{i\in r} b_i \geq b_0 + 1 \quad && \textnormal{for all } r \in R \\
        &&& b_1 \geq b_2 \geq \dots \geq b_n \geq 0, b_0 \geq 0
    \end{aligned}
$$
- hard for two reasons: **Integer LP** and potentially **exponential number of constraints**

#### Ideas
- Suboptimal but feasible MILP solution will give reduced but suboptimal formulation
- Can use original (normalized) formulation as hot start for MILP 
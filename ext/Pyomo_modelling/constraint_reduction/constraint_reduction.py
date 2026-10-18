import pyomo.environ as pyo


def get_dist(S):
    t = -1

    for i in range(len(S) - 1):
        if S[i+1] - S[i] >= 2:
            t = i

    return t


def is_roof(coeffs, S):
    value = sum(coeffs[i] for i in S)

    roof = value >= coeffs[0] + 1
    for i in S:
        roof = roof and (value - coeffs[i] <= coeffs[0])

        if i < len(coeffs) - 1 and not i+1 in S:
            roof = roof and (value - coeffs[i] + coeffs[i+1] <= coeffs[0])

    return roof


def find_roofs(coeffs):
    n = len(coeffs) - 1
    a0 = coeffs[0]

    roofs = []

    S = []
    dist_element = -1
    max_element = 0

    while (max_element < n) or (dist_element >= 0):
        if (sum(coeffs[i] for i in S) <= a0) and (max_element < n):
            S = S + [max_element + 1]

        elif (sum(coeffs[i] for i in S) > a0) and (max_element < n):
            S = S[:-1] + [max_element + 1]

        elif (max_element == n) and (dist_element >= 0):
            S = S[:dist_element] + [S[dist_element] + 1] 

        if is_roof(coeffs, S):
            roofs.append(S)

        dist_element = get_dist(S)
        max_element = S[-1]

    return roofs


def get_dual(coeffs):
    dual_coeffs = coeffs.copy()
    dual_coeffs[0] = sum(coeffs[1:]) - coeffs[0] - 1

    return dual_coeffs


def find_ceils(coeffs):
    n = len(coeffs) - 1

    dual_coeffs = get_dual(coeffs)
    dual_roofs = find_roofs(dual_coeffs)

    ceils = [[i for i in range(1, n+1) if i not in S] for S in dual_roofs]

    return ceils


def construct_reduction_MILP(coeffs):
    n = len(coeffs) - 1

    roofs = find_roofs(coeffs)
    ceils = find_ceils(coeffs)

    model = pyo.ConcreteModel()

    model.b = pyo.Var(pyo.RangeSet(0, n), within = pyo.NonNegativeIntegers)

    for c in ceils:
        ceil_constraint = pyo.Constraint(expr = sum(model.b[i] for i in c) <= model.b[0])
        setattr(model, f"ceil_constraint_{c}", ceil_constraint)

    for r in roofs:
        roof_constraint = pyo.Constraint(expr = sum(model.b[i] for i in r) >= model.b[0] + 1)
        setattr(model, f"roof_constraint_{r}", roof_constraint)

    for i in range(1, n-1):
        order_constraint = pyo.Constraint(expr = model.b[i] >= model.b[i+1])
        setattr(model, f"order_constraint_{i}", order_constraint)
    
    model.obj = pyo.Objective(expr = model.b[0], sense = pyo.minimize)

    return model
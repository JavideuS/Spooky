"""
Generates presentation.pptx: the mathematical-model deck for Spooky (QUBO-based MAPF).

Content mirrors quantum/paper/paper.tex (core penalties, QUBO background) plus
material not in the paper: the trailing penalty (QUBOBuilder.apply_trailing_penalty),
the binary-vs-one-hot encoding investigation (explore/binary-encoding branch,
quantum/tests/BINARY_ENCODING_FINDINGS.md), empirical weight tuning practice
(config/config.yaml), and a proposed task-allocation layer above the current
path-planning-only scope.

Run: python generate_presentation.py   (writes presentation.pptx alongside this file)
"""

import re
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x0A, 0x11, 0x2E)
NAVY_2 = RGBColor(0x12, 0x1B, 0x3F)
INK = RGBColor(0x1B, 0x22, 0x38)
WHITE = RGBColor(0xF4, 0xF6, 0xFB)
GRAY = RGBColor(0x5C, 0x64, 0x7A)
LIGHT_GRAY = RGBColor(0xE6, 0xE9, 0xF2)
CYAN = RGBColor(0x4C, 0xC9, 0xF0)
VIOLET = RGBColor(0x9B, 0x5D, 0xE0)
AMBER = RGBColor(0xF2, 0xA6, 0x5A)
GREEN = RGBColor(0x4C, 0xD9, 0x9A)
RED = RGBColor(0xE8, 0x5C, 0x5C)
CARD_BG = RGBColor(0xFF, 0xFF, 0xFF)
CARD_BORDER = RGBColor(0xDD, 0xE1, 0xEC)

FONT = "Calibri"
MATH_FONT = "Cambria Math"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]

_slide_num = [0]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def new_slide():
    s = prs.slides.add_slide(BLANK)
    _slide_num[0] += 1
    return s


def set_bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, color, line=False):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if line:
        shp.line.color.rgb = CARD_BORDER
        shp.line.width = Pt(0.75)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_rounded(slide, x, y, w, h, color, line_color=None, radius=0.06):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    try:
        shp.adjustments[0] = radius
    except Exception:
        pass
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if line_color:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, text, size=18, color=INK, bold=False, italic=False,
             align=PP_ALIGN.LEFT, font=FONT, anchor=MSO_ANCHOR.TOP, line_spacing=1.0,
             wrap=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = font
    return tb


def add_bullets(slide, x, y, w, h, items, size=15, color=INK, font=FONT,
                 space_after=8, anchor=MSO_ANCHOR.TOP):
    """items: list of (level, text, {bold,color,italic}) or plain strings."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    first = True
    for item in items:
        if isinstance(item, str):
            level, text, opts = 0, item, {}
        else:
            level, text, opts = item[0], item[1], (item[2] if len(item) > 2 else {})
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = 0
        p.space_after = Pt(space_after)
        p.line_spacing = 1.08
        indent = Emu(Inches(0.28) * level)
        p.alignment = PP_ALIGN.LEFT
        pPr = p._pPr if p._pPr is not None else p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.22) + indent))
        pPr.set("indent", str(-Inches(0.22)))
        bullet_char = "•" if level == 0 else ("–" if level == 1 else "◦")
        buChar = pPr.makeelement(qn("a:buChar"), {"char": bullet_char})
        buFont = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
        for tag in ("a:buChar", "a:buNone", "a:buAutoNum"):
            for el in pPr.findall(qn(tag)):
                pPr.remove(el)
        pPr.append(buFont)
        pPr.append(buChar)
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size - level * 1.0)
        r.font.color.rgb = opts.get("color", color)
        r.font.bold = opts.get("bold", False)
        r.font.italic = opts.get("italic", False)
        r.font.name = font
    return tb


def set_baseline(run, val):
    rPr = run._r.get_or_add_rPr()
    rPr.set("baseline", str(val))


_TOKEN_RE = re.compile(r"_\{([^}]*)\}|\^\{([^}]*)\}|([^_^]+)")


def add_formula(slide, x, y, w, h, text, size=24, color=INK, align=PP_ALIGN.CENTER,
                 font=MATH_FONT, bold=False, anchor=MSO_ANCHOR.MIDDLE):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    p = tf.paragraphs[0]
    p.alignment = align
    for m in _TOKEN_RE.finditer(text):
        sub, sup, normal = m.groups()
        run = p.add_run()
        if sub is not None:
            run.text = sub
            run.font.size = Pt(size * 0.62)
            set_baseline(run, -25000)
        elif sup is not None:
            run.text = sup
            run.font.size = Pt(size * 0.62)
            set_baseline(run, 30000)
        else:
            run.text = normal
            run.font.size = Pt(size)
        run.font.name = font
        run.font.color.rgb = color
        run.font.bold = bold
    return tb


def add_formula_lines(slide, x, y, w, h, lines, size=22, color=INK, align=PP_ALIGN.CENTER,
                       font=MATH_FONT, gap=1.25):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = 0
    tf.margin_right = 0
    first = True
    for line in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.line_spacing = gap
        for m in _TOKEN_RE.finditer(line):
            sub, sup, normal = m.groups()
            run = p.add_run()
            if sub is not None:
                run.text = sub
                run.font.size = Pt(size * 0.62)
                set_baseline(run, -25000)
            elif sup is not None:
                run.text = sup
                run.font.size = Pt(size * 0.62)
                set_baseline(run, 30000)
            else:
                run.text = normal
                run.font.size = Pt(size)
            run.font.name = font
            run.font.color.rgb = color
    return tb


def footer(slide, label, dark=False):
    color = RGBColor(0x8A, 0x93, 0xB0) if dark else GRAY
    add_text(slide, Inches(0.55), Inches(7.14), Inches(8), Inches(0.3),
              f"Spooky — QUBO Mathematical Model  ·  {label}", size=10, color=color)
    add_text(slide, Inches(12.4), Inches(7.14), Inches(0.6), Inches(0.3),
              str(_slide_num[0]), size=10, color=color, align=PP_ALIGN.RIGHT)


def kicker(slide, text, color=CYAN):
    add_text(slide, Inches(0.55), Inches(0.42), Inches(8), Inches(0.35), text.upper(),
              size=13, color=color, bold=True)


def title(slide, text, y=Inches(0.72), size=28, color=INK, w=Inches(12.2)):
    add_text(slide, Inches(0.55), y, w, Inches(0.7), text, size=size, color=color, bold=True)
    add_rect(slide, Inches(0.55), y + Inches(0.62), Inches(0.55), Pt(3.2), CYAN)


def content_slide(kicker_text, title_text):
    s = new_slide()
    set_bg(s, WHITE)
    kicker(s, kicker_text)
    title(s, title_text)
    return s


# ---------------------------------------------------------------------------
# 1. Title slide
# ---------------------------------------------------------------------------
s = new_slide()
set_bg(s, NAVY)
add_rect(s, Inches(0), Inches(0), Inches(0.14), SLIDE_H, CYAN)
add_text(s, Inches(0.9), Inches(2.55), Inches(11), Inches(0.4), "SPOOKY",
          size=16, color=CYAN, bold=True)
add_text(s, Inches(0.9), Inches(2.95), Inches(11.2), Inches(1.5),
          "The QUBO Mathematical Model", size=42, color=WHITE, bold=True)
add_text(s, Inches(0.9), Inches(3.85), Inches(11), Inches(0.8),
          "Formulation, extensions beyond the paper, empirical weight tuning,\n"
          "the binary-encoding dead end, and what a task-allocation layer would add.",
          size=16, color=RGBColor(0xB9, 0xC2, 0xDE), line_spacing=1.3)
add_text(s, Inches(0.9), Inches(6.7), Inches(8), Inches(0.4),
          "Javier González Villasmil  ·  arXiv:2602.14799", size=13,
          color=RGBColor(0x8A, 0x93, 0xB0))

# ---------------------------------------------------------------------------
# 2. Agenda
# ---------------------------------------------------------------------------
s = new_slide()
set_bg(s, NAVY)
kicker(s, "Agenda", color=CYAN)
add_text(s, Inches(0.55), Inches(0.75), Inches(10), Inches(0.7), "What this deck covers",
          size=30, color=WHITE, bold=True)
agenda = [
    ("01", "QUBO, not ILP→QUBO", "why the model is native QUBO from the start"),
    ("02", "Core penalties", "one-hot, adjacency, start/goal, lock — as in the paper"),
    ("03", "Extra penalties", "backtracking, teleportation, approximation heuristics"),
    ("04", "Collision model", "crash penalty, and swap-collision (not in the paper)"),
    ("05", "Weight tuning", "why it's empirical, and the current production values"),
    ("06", "Binary vs one-hot", "the encoding explored on a branch — and why it lost"),
    ("07", "Task allocation", "the layer this system does not yet have"),
]
col_w = Inches(5.9)
for i, (num, t, sub) in enumerate(agenda):
    col = i % 2
    row = i // 2
    x = Inches(0.55) + col * (col_w + Inches(0.3))
    y = Inches(1.75) + row * Inches(1.32)
    add_text(s, x, y, Inches(0.9), Inches(0.5), num, size=22, color=VIOLET, bold=True, font=MATH_FONT)
    add_text(s, x + Inches(0.85), y - Inches(0.02), col_w - Inches(0.9), Inches(0.4), t,
              size=17, color=WHITE, bold=True)
    add_text(s, x + Inches(0.85), y + Inches(0.38), col_w - Inches(0.9), Inches(0.6), sub,
              size=12.5, color=RGBColor(0xAD, 0xB6, 0xD4))
footer(s, "Agenda", dark=True)

# ---------------------------------------------------------------------------
# Section divider helper
# ---------------------------------------------------------------------------
def section_slide(num, big_title, sub):
    s = new_slide()
    set_bg(s, NAVY)
    add_rect(s, Inches(0), Inches(0), SLIDE_W, Inches(0.06), CYAN)
    add_text(s, Inches(0.9), Inches(2.7), Inches(3), Inches(1), num, size=64, color=VIOLET,
              bold=True, font=MATH_FONT)
    add_text(s, Inches(0.95), Inches(3.75), Inches(11), Inches(1), big_title, size=36,
              color=WHITE, bold=True)
    add_text(s, Inches(0.95), Inches(4.55), Inches(10), Inches(0.8), sub, size=16,
              color=RGBColor(0xB9, 0xC2, 0xDE), line_spacing=1.3)
    footer(s, big_title, dark=True)
    return s


# ---------------------------------------------------------------------------
# SECTION 01 — QUBO not ILP->QUBO
# ---------------------------------------------------------------------------
section_slide("01", "QUBO, Not ILP → QUBO",
              "Spooky's model was never an integer/binary linear program that got compiled down.\n"
              "It is authored directly as a QUBO — that distinction shapes everything after it.")

s = content_slide("Modeling philosophy", "The standard pipeline vs. Spooky's pipeline")
# Left card: standard
lx, ly, lw, lh = Inches(0.55), Inches(1.85), Inches(5.9), Inches(4.9)
add_rounded(s, lx, ly, lw, lh, RGBColor(0xFB, 0xF4, 0xF2), line_color=RGBColor(0xEE, 0xD3, 0xC9))
add_text(s, lx + Inches(0.35), ly + Inches(0.25), lw - Inches(0.7), Inches(0.4),
          "Standard OR practice", size=16, bold=True, color=RED)
std_items = [
    (0, "Model as ILP / BLP:", {"bold": True}),
    (1, "min cᵀx  s.t.  Ax ≤ b,  x ∈ {0,1}ⁿ", {"italic": True, "color": GRAY}),
    (0, "Constraints are hard, explicit inequalities/equalities"),
    (0, "Compile down to QUBO for a quantum/annealing solver:"),
    (1, "add slack variables for every inequality"),
    (1, "pick big-M penalty weights from a provable bound"),
    (1, "flatten equalities into quadratic penalty terms"),
    (0, "QUBO is a compilation target — a solver adapter, not the model"),
]
add_bullets(s, lx + Inches(0.35), ly + Inches(0.8), lw - Inches(0.7), lh - Inches(1.0),
            std_items, size=13.5)

rx = Inches(6.75)
add_rounded(s, rx, ly, lw, lh, RGBColor(0xEE, 0xF9, 0xF4), line_color=RGBColor(0xC7, 0xEC, 0xDC))
add_text(s, rx + Inches(0.35), ly + Inches(0.25), lw - Inches(0.7), Inches(0.4),
          "Spooky", size=16, bold=True, color=RGBColor(0x1E, 0x9E, 0x6B))
spooky_items = [
    (0, "Model directly as a QUBO:", {"bold": True}),
    (1, "C(x) = Σₖ λₖ Pₖ(x)", {"italic": True, "color": GRAY}),
    (0, "No intermediate ILP layer, no slack variables, no big-M"),
    (0, "Every requirement is authored as a quadratic penalty term"),
    (1, "one-hot, adjacency, start/goal, crash, swap, …"),
    (0, "Objective and “constraints” are the same mathematical object"),
    (0, "Penalty weights (λ) are first-class, tuned directly —"),
    (1, "not derived from a worst-case bound"),
]
add_bullets(s, rx + Inches(0.35), ly + Inches(0.8), lw - Inches(0.7), lh - Inches(1.0),
            spooky_items, size=13.5)
footer(s, "QUBO, not ILP→QUBO")

s = content_slide("Why it matters", "Big-M vs. tuned penalties")
add_bullets(s, Inches(0.55), Inches(1.9), Inches(11.8), Inches(2.6), [
    (0, "Big-M conversion picks weights from a worst-case algebraic bound — mathematically safe, "
        "but it produces a poorly conditioned Q matrix (huge dynamic range between terms)."),
    (0, "A poorly conditioned Q matrix is exactly what breaks annealers and QAOA: dominant terms drown "
        "out weaker ones, and normalization can no longer rescue every term at once."),
    (0, "Because Spooky's penalties are authored (not derived), each λ can be tuned against the "
        "solver's actual behavior — what the paper calls the empirical tuning process (Section 5)."),
    (0, "This also makes the unconstrained nature of QUBO an asset rather than a workaround: the "
        "“constraints” were never anything but penalty terms to begin with."),
], size=15.5, space_after=14)
footer(s, "QUBO, not ILP→QUBO")

# ---------------------------------------------------------------------------
# QUBO core definition
# ---------------------------------------------------------------------------
s = content_slide("Background", "QUBO: the core definition")
add_formula(s, Inches(0.55), Inches(1.85), Inches(6), Inches(1.0), "C(x) = x^{T}Qx", size=34)
add_bullets(s, Inches(0.55), Inches(3.0), Inches(6.1), Inches(3.6), [
    (0, "x ∈ {0,1}ⁿ — binary decision variables"),
    (0, "Q ∈ ℝⁿˣⁿ — symmetric / upper-triangular, encodes pairwise relations"),
    (0, "Unconstrained: every requirement is embedded in the cost function"),
    (0, "Diagonal of Q = linear terms  •  off-diagonal = quadratic (pairwise) terms"),
    (0, "Since xⱼ = xⱼ² for binary variables, linear terms fold onto the diagonal — "
        "everything becomes one quadratic form"),
    (0, "Equivalent to the Ising model (x → {−1,+1}) — the bridge to annealing / QAOA Hamiltonians"),
], size=14.5, space_after=10)
add_rounded(s, Inches(7.1), Inches(1.9), Inches(5.6), Inches(4.6), RGBColor(0xF6, 0xF7, 0xFB),
            line_color=CARD_BORDER)
add_text(s, Inches(7.4), Inches(2.1), Inches(5), Inches(0.4), "Objective = weighted penalty sum",
          size=13.5, bold=True, color=GRAY)
add_formula(s, Inches(7.4), Inches(2.65), Inches(5), Inches(0.8),
            "F(x) = λ₁·distance(x) + λ₂·obstacle(x) + …", size=17,
            align=PP_ALIGN.LEFT)
add_text(s, Inches(7.4), Inches(3.55), Inches(5), Inches(2.7),
          "The QUBO is not solved directly — it's converted into a Hamiltonian and solved via "
          "quantum annealing, QAOA, or classical simulated annealing. Spooky targets all three "
          "through the same builder → solver interface (DWave_solver.py, Pennylane_solver.py).",
          size=13, color=GRAY, line_spacing=1.35)
footer(s, "QUBO background")

# ---------------------------------------------------------------------------
# Variable encoding
# ---------------------------------------------------------------------------
s = content_slide("Background", "Variable encoding — grid & graph")
add_text(s, Inches(0.55), Inches(1.85), Inches(5.6), Inches(0.4), "Grid (linear one-hot)",
          size=15, bold=True, color=VIOLET)
add_formula(s, Inches(0.55), Inches(2.3), Inches(5.9), Inches(0.9),
            "idx = robot·(M·N·T) + t·(M·N) + i·N + j", size=19,
            align=PP_ALIGN.LEFT)
add_text(s, Inches(6.85), Inches(1.85), Inches(5.6), Inches(0.4), "Graph (node-based)",
          size=15, bold=True, color=VIOLET)
add_formula(s, Inches(6.85), Inches(2.3), Inches(5.9), Inches(0.9),
            "idx = robot·(nodes·T) + t·nodes + node_id", size=19,
            align=PP_ALIGN.LEFT)
add_bullets(s, Inches(0.55), Inches(3.5), Inches(11.8), Inches(3.2), [
    (0, "Groups variables per timestep, enumerated in standard reading order (grid) or node index (graph)"),
    (0, "Multi-robot uses a global clock: robot 1 gets M·N·T slots, robot 2 the next M·N·T, "
        "and so on — linear index growth in the number of robots"),
    (0, "Local per-robot time is offset relative to the first active robot, so collision penalties only "
        "need to compare robots active at the same absolute timestep"),
    (0, "This exact index formula is what every penalty term below derives its variable indices from — "
        "any new constraint has to follow it"),
], size=14.5, space_after=11)
footer(s, "Variable encoding")

# ---------------------------------------------------------------------------
# SECTION 02 — core penalties
# ---------------------------------------------------------------------------
section_slide("02", "Core Constraint Penalties",
              "The four penalties every valid QUBO path needs, plus goal-lock — "
              "straight from the paper's formulation section.")


def penalty_slide(kicker_text, name, formula, formula_size, notes, extra_formula=None,
                   extra_size=18):
    s = content_slide(kicker_text, name)
    fy = Inches(1.95)
    add_rounded(s, Inches(0.55), fy, Inches(11.8), Inches(1.5) if not extra_formula else Inches(1.9),
                RGBColor(0xF6, 0xF7, 0xFB), line_color=CARD_BORDER)
    if isinstance(formula, list):
        add_formula_lines(s, Inches(0.8), fy + Inches(0.12), Inches(11.3), Inches(1.3), formula,
                           size=formula_size)
    else:
        add_formula(s, Inches(0.8), fy + Inches(0.12), Inches(11.3), Inches(1.1), formula,
                    size=formula_size)
    ny = fy + (Inches(1.6) if not extra_formula else Inches(2.0))
    add_bullets(s, Inches(0.55), ny, Inches(11.8), Inches(4.6 - (ny - fy)), notes, size=14.5,
                space_after=10)
    return s


penalty_slide(
    "Core penalty · 1 of 4", "One-hotness — exactly one cell per timestep",
    "P_{hot} = K_{hot} · Σ_{t=0}^{T} (1 − Σ_{i,j} x_{i,j,t})²",
    24,
    [
        (0, "A physical agent occupies exactly one position at any given time — this penalty "
            "drives the QUBO toward that reality."),
        (0, "Expanding the square shows the origin of every cross-term xᵢxⱼ in Q: "
            "(Σx)² = Σx² + 2Σᵢ<ⱼ xᵢxⱼ"),
        (0, "Interacts strongly with the adjacency penalty below — the two frequently need "
            "post-processing to resolve disagreements (see the Continuity check, later in the paper)."),
    ],
)

penalty_slide(
    "Core penalty · 2 of 4", "Adjacency — no teleporting between cells",
    "P_{adj} = K_{adj} · Σ_{t=0}^{T−1} Σ_{i,j} x_{i,j,t}·(1 − Σ_{(k,l)∈N(i,j)} x_{k,l,t+1})",
    18.5,
    [
        (0, "N(i,j) is the 4-connectivity neighbor set of cell (i,j) — the most computationally "
            "expensive penalty, best precomputed into a neighbor dictionary."),
        (0, "Obstacle avoidance is handled structurally here, not as its own penalty: obstacle cells "
            "are simply removed from every N(i,j), so a move into one already breaks adjacency."),
        (0, "Same trick applies to graphs — unconnected nodes are never in each other's neighbor "
            "set, so no separate obstacle logic is needed there either."),
    ],
)

penalty_slide(
    "Core penalty · 3 of 4", "Start & goal conditions",
    ["P_{start} = −K_{start} · x_{si,sj,0}",
     "P_{goal} = −K_{goal} · x_{ei,ej,T}   (naïve, last-step-only form)"],
    22,
    [
        (0, "Both reduce from a squared form (1 − x)² to a linear reward once x² = x is used "
            "for binary variables — same simplification as one-hotness."),
        (0, "The naïve goal penalty only rewards reaching the goal at the exact final timestep, which "
            "assumes you already know T. In production this is replaced by the late-time weighted "
            "version (next section) — same idea, applied at every step with growing weight."),
        (0, "Goal cannot be rewarded at t = 0, or it directly conflicts with the start-condition penalty."),
    ],
)

penalty_slide(
    "Core penalty · 4 of 4", "Goal lock — stay once you arrive",
    "P_{lock} = K_{lock} · Σ_{t=0}^{T−2} x_{g,t}·(1 − x_{g,t+1})",
    22,
    [
        (0, "Classical planners never worry about this — they don't operate on fixed timesteps. "
            "QUBO does, and it doesn't inherently know the right time window, so it tends to keep "
            "moving past the goal unless told to stop."),
        (0, "Not treated as essential: extra post-goal moves can simply be clipped from the decoded "
            "path instead. In practice this penalty can be dropped without much cost."),
    ],
)

# ---------------------------------------------------------------------------
# SECTION 03 — extra / tuned penalties
# ---------------------------------------------------------------------------
section_slide("03", "Extra Penalties & Heuristics",
              "Not required for a valid path, but required for a good one — "
              "these fix teleportation, backtracking, and windowed goal-seeking.")

penalty_slide(
    "Extra penalty", "Backtracking — punish revisiting cells",
    "P_{bt} = K_{bt} · Σ_{(i,j)≠goal} Σ_{t₁<t₂} x_{i,j,t₁}·x_{i,j,t₂}",
    19,
    [
        (0, "A global constraint (not per-timestep): penalizes any pair t₁ < t₂ where the agent "
            "occupies the same non-goal cell twice."),
        (0, "A backtracking solution is, by construction, never optimal — this steers the solver "
            "toward the direct path and away from wandering near the start or goal."),
        (0, "Explicitly excludes the goal cell, or it would conflict with the goal-lock penalty above."),
    ],
)

s = content_slide("Extra penalty", "Teleportation penalty & distance heuristics")
add_bullets(s, Inches(0.55), Inches(1.9), Inches(11.8), Inches(1.6), [
    (0, "Teleportation penalty (K_tp): penalizes reaching the goal earlier than physically possible, "
        "using the Manhattan/Euclidean lower bound on step count — balances the goal penalty and "
        "meaningfully improves results in practice."),
], size=14.5, space_after=8)
add_text(s, Inches(0.55), Inches(3.3), Inches(5.8), Inches(0.4), "Grid — Manhattan distance",
          size=14, bold=True, color=VIOLET)
add_formula(s, Inches(0.55), Inches(3.7), Inches(5.8), Inches(0.7),
            "d = |x₂ − x₁| + |y₂ − y₁|", size=20, align=PP_ALIGN.LEFT)
add_text(s, Inches(6.85), Inches(3.3), Inches(5.8), Inches(0.4), "Graph — Euclidean distance",
          size=14, bold=True, color=VIOLET)
add_formula(s, Inches(6.85), Inches(3.7), Inches(5.8), Inches(0.7),
            "d = √((x − x_{goal})² + (y − y_{goal})²)", size=20, align=PP_ALIGN.LEFT)
add_bullets(s, Inches(0.55), Inches(4.75), Inches(11.8), Inches(1.7), [
    (0, "Any valid path needs at least the Manhattan/Euclidean number of steps — fewer means the "
        "answer is invalid, more can be encouraged to shrink toward it."),
], size=14.5)
footer(s, "Extra penalties")

s = content_slide("Extra penalty", "Windowed goal-seeking: late-time & approximation penalties")
add_bullets(s, Inches(0.55), Inches(1.9), Inches(5.8), Inches(4.6), [
    (0, "Late-time goal penalty", {"bold": True}),
    (1, "Weight on reaching the goal increases every timestep (up to ~2×) instead of firing "
        "only at the last step"),
    (1, "Fixes the constant-time approach's teleportation problem and the early-time approach's "
        "over-eagerness — empirically the most effective of the three"),
], size=14)
add_bullets(s, Inches(6.85), Inches(1.9), Inches(5.5), Inches(4.6), [
    (0, "Approximation penalty (K_goal_approx)", {"bold": True}),
    (1, "Used inside a time window that can't reach the goal yet — rewards approaching it "
        "(shrinking Manhattan/Euclidean distance) instead"),
    (1, "Paired with an obstacle potential field so the solver favors open space, not just "
        "“closer” cells — also spreads multi-robot traffic away from clutter"),
    (1, "Must stay weak relative to backtracking, or it fights the windows that already tried "
        "and failed to reduce distance"),
], size=14)
footer(s, "Extra penalties")

# ---------------------------------------------------------------------------
# SECTION 04 — collision model
# ---------------------------------------------------------------------------
section_slide("04", "Multi-Robot Collision Model",
              "Crash avoidance is in the paper. Exact swap detection isn't — "
              "it's flagged there as a known limitation, and it's still open. What's "
              "implemented under the old name “K_swap” was never a swap solution; "
              "renamed K_trail to say what it actually does.")

penalty_slide(
    "Collision · in the paper", "Crash penalty — same cell, same time",
    "P_{crash} = K_{crash} · Σ_t Σ_{r₁<r₂} Σ_{(i,j)∈shared} x_{r₁,i,j,t}·x_{r₂,i,j,t}",
    16.5,
    [
        (0, "Only evaluated over cells both robots can actually reach at time t (shared_cells) — "
            "keeps the penalty term count small instead of scanning the whole grid."),
        (0, "Rule of thumb from the config: K_crash must stay ≤ K_adj, or the solver has no "
            "incentive to prefer a legal move over sitting through a collision."),
    ],
)

s = content_slide("Collision · not in the paper", "Trailing penalty (was mislabeled “swap”)")
add_rounded(s, Inches(0.55), Inches(1.85), Inches(11.8), Inches(1.55), RGBColor(0xF6, 0xF7, 0xFB),
            line_color=CARD_BORDER)
add_formula_lines(s, Inches(0.8), Inches(1.95), Inches(11.3), Inches(1.4), [
    "P_{trail} = K_{crash}·x_{r₁,t}x_{r₂,t}  +  K_{trail}·(x_{r₁,t+1}x_{r₂,t} + x_{r₁,t}x_{r₂,t+1})",
], size=17.5)
add_bullets(s, Inches(0.55), Inches(3.6), Inches(11.8), Inches(3.5), [
    (0, "The paper's own Limitations section calls exact swap detection out as unsolved: it's a "
        "genuine AND-of-both-endpoints condition needing 4 variables jointly, i.e. quartic "
        "(degree 4) or ancilla-based degree reduction, not quadratic — QUBO's native degree. "
        "Still true; still open."),
    (0, "This formula (QUBOBuilder.apply_trailing_penalty / GraphQUBO equivalent) was never that "
        "solution — it's a different, quadratic-native predicate that happens to reuse the same "
        "two variables:"),
    (1, "same-cell/same-time term reuses K_crash unchanged, so base crash protection isn't diluted"),
    (1, "two cross-time terms catch robot 1 arriving where robot 2 just was, and vice versa — "
        "an OR of either endpoint, not the AND both directions that a real swap needs"),
    (0, "That OR is deliberate, not a compromise: it also fires when one robot simply follows "
        "another into a just-vacated cell — not a reciprocal exchange, but a real hazard once "
        "bodies have extent and steps aren't perfectly synchronised (see clearance.md). "
        "K_trail is the knob to relax if the false-positive cost outweighs the benefit."),
    (0, "When K_trail is active it replaces the plain crash penalty entirely (build() drops “crash” "
        "from the constraint list) so the same-time term is never double-counted."),
], size=13.5, space_after=8)
footer(s, "Trailing")

# ---------------------------------------------------------------------------
# SECTION 05 — weight tuning
# ---------------------------------------------------------------------------
section_slide("05", "Weight Tuning",
              "Every K constant above is empirical. There's no closed-form optimum — "
              "here's the methodology and the current production values.")

s = content_slide("Methodology", "Why tuning is empirical, not derived")
add_bullets(s, Inches(0.55), Inches(1.9), Inches(11.8), Inches(4.6), [
    (0, "Unlike big-M in a classical ILP→QUBO conversion — which has a provable bound formula "
        "— there is no mathematically “correct” weight for a penalty that was authored, not derived."),
    (0, "Tuning discipline: fit weights against the most general / least constrained scenario first "
        "(the open map, fewest active constraints). Fitting to a constrained scenario instead tends "
        "to overfit — the solver has more freedom in the general case, so weights that hold there "
        "generalize; ones that only work under heavy constraints usually don't."),
    (0, "Normalization scale is the one global knob that keeps weights problem-size-independent: "
        "you tune a single number instead of re-deriving every K per map size."),
    (1, "PennyLane / QAOA: scale 2.0 below ~60–100 variables, scale 1.0 above"),
    (1, "D-Wave annealing: benefits from a higher scale (5.0) — fewer reads needed for the same "
        "or better accuracy"),
    (0, "Combined with the time-window approach, this means the main remaining tuning surface is "
        "just “how many timesteps per window,” not per-scenario penalty weights."),
], size=14, space_after=10)
footer(s, "Weight tuning")

s = content_slide("Methodology", "Current production penalty sets (config/config.yaml)")
headers = ["Penalty", "crash (default)", "trailing"]
rows = [
    ("K_hot", "9", "9"),
    ("K_adj", "4.8", "4.8"),
    ("K_start", "6.5", "6.5"),
    ("K_goal", "3", "3"),
    ("K_lock", "4", "4"),
    ("K_bt", "2.3", "2.3"),
    ("K_tp", "1.2", "1.2"),
    ("K_goal_approx", "0.7", "0.7"),
    ("K_crash", "3", "2.7"),
    ("K_trail", "—", "3  (kept ≤ K_crash to avoid over-constraint)"),
]
tx, ty, tw = Inches(0.55), Inches(1.9), Inches(11.8)
n_rows = len(rows) + 1
row_h = Inches(0.4)
table_shape = s.shapes.add_table(n_rows, 3, tx, ty, tw, row_h * n_rows)
table = table_shape.table
table.columns[0].width = Inches(3.2)
table.columns[1].width = Inches(4.3)
table.columns[2].width = Inches(4.3)
for c, htext in enumerate(headers):
    cell = table.cell(0, c)
    cell.text = htext
    for p in cell.text_frame.paragraphs:
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(13)
            r.font.color.rgb = WHITE
            r.font.name = FONT
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY
for ri, (k, v1, v2) in enumerate(rows, start=1):
    vals = [k, v1, v2]
    for c, val in enumerate(vals):
        cell = table.cell(ri, c)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(12.5)
                r.font.name = MATH_FONT if c == 0 else FONT
                r.font.color.rgb = INK
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT_GRAY
add_text(s, Inches(0.55), Inches(6.35), Inches(11.8), Inches(0.5),
          "“swap” replaces “crash” entirely when active — not summed with it.",
          size=11.5, italic=True, color=GRAY)
footer(s, "Weight tuning")

# ---------------------------------------------------------------------------
# SECTION 06 — binary vs one-hot
# ---------------------------------------------------------------------------
section_slide("06", "Binary vs. One-Hot Encoding",
              "Explored on explore/binary-encoding, never merged. A negative result — "
              "documented precisely enough that it doesn't need re-deriving.")

s = content_slide("Encoding alternative", "The theoretical appeal: fewer qubits per cell")
add_bullets(s, Inches(0.55), Inches(1.9), Inches(11.8), Inches(1.5), [
    (0, "One-hot uses N variables per (robot, timestep) — one bit per grid cell / graph node."),
    (0, "Binary needs only B = ⌈log₂N⌉ bits to represent the same N positions — exponentially fewer."),
], size=14.5, space_after=8)
headers2 = ["N (cells)", "one-hot", "binary", "binary + exact-match ancilla"]
rows2 = [("4", "4", "2", "6"), ("9", "9", "4", "13"), ("16", "16", "4", "20"),
         ("25", "25", "5", "30"), ("36", "36", "6", "42")]
tx, ty = Inches(0.55), Inches(3.5)
n_rows2 = len(rows2) + 1
row_h2 = Inches(0.42)
table_shape2 = s.shapes.add_table(n_rows2, 4, tx, ty, Inches(11.8), row_h2 * n_rows2)
table2 = table_shape2.table
for c in range(4):
    table2.columns[c].width = Inches(2.95)
for c, htext in enumerate(headers2):
    cell = table2.cell(0, c)
    cell.text = htext
    for p in cell.text_frame.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(13)
            r.font.color.rgb = WHITE
            r.font.name = FONT
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY
for ri, vals in enumerate(rows2, start=1):
    for c, val in enumerate(vals):
        cell = table2.cell(ri, c)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(12.5)
                r.font.name = MATH_FONT
                r.font.color.rgb = RED if c == 3 else INK
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT_GRAY
add_text(s, Inches(0.55), Inches(6.05), Inches(11.8), Inches(0.9),
          "A genuine fix needs an exact-match ancilla per real “from” cell to detect adjacency "
          "correctly — always strictly more variables than one-hot (B + N > N). The savings are illusory "
          "once adjacency is handled honestly.", size=12.5, color=GRAY, line_spacing=1.3)
footer(s, "Binary vs. one-hot")

s = content_slide("Encoding alternative", "Why it fails: adjacency has no closed form")
add_bullets(s, Inches(0.55), Inches(1.9), Inches(11.8), Inches(4.6), [
    (0, "One-hot's adjacency term reads each bit independently (“is this specific cell active”). "
        "Binary needs to read the combination of bits meaningfully differently depending on which "
        "real cell they encode — the same shared (2B,2B) coefficient block gets tiled across every "
        "one of the N possible “from” cells."),
    (0, "That's a degrees-of-freedom mismatch: only O(B²) = O(log²N) free coefficients must "
        "simultaneously satisfy the ranking constraints for all N cells at once."),
    (0, "Proven, not approximated: a max-margin LP fit (scipy linprog, highs) finds the optimal "
        "separating margin is exactly 0 for every grid ≥ 3×4, including exact powers of 2. "
        "linprog solves this exactly — margin = 0 means provably no separator exists, not "
        "“wasn't found.” An SDP relaxation can't do better either: there's no non-convexity here "
        "to relax against."),
        (0, "A refinement (staying in place made legal + row/column encoded in disjoint power-of-2 "
        "bit fields instead of i·N+j) does produce a real positive margin — but only for small, "
        "open grids, and it degrades fast with size and obstacles."),
], size=13.5, space_after=9)
footer(s, "Binary vs. one-hot")

s = content_slide("Encoding alternative", "The refinement's actual reach")
headers3 = ["Map", "baseline margin", "candidate margin", "viable on"]
rows3 = [
    ("3×3, 0 obstacles", "0.333 (6.7%)", "0.625 (12.5%)", "SA + hardware"),
    ("5×5, 0 obstacles", "fails", "0.313 (6.3%)", "SA + hardware"),
    ("5×5, 3 obstacles", "fails", "0.033 (0.65%)", "SA only"),
    ("10×10, 0 obstacles", "fails", "0.096 (1.9%)", "SA + hardware (marginal)"),
    ("obs10x10_easy / hard", "fails", "fails", "neither"),
    ("50×50+", "—", "not computable", "—"),
]
tx, ty = Inches(0.55), Inches(1.9)
n_rows3 = len(rows3) + 1
row_h3 = Inches(0.5)
table_shape3 = s.shapes.add_table(n_rows3, 4, tx, ty, Inches(11.8), row_h3 * n_rows3)
table3 = table_shape3.table
widths3 = [Inches(3.3), Inches(2.8), Inches(2.8), Inches(2.9)]
for c, wdt in enumerate(widths3):
    table3.columns[c].width = wdt
for c, htext in enumerate(headers3):
    cell = table3.cell(0, c)
    cell.text = htext
    for p in cell.text_frame.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(12.5)
            r.font.color.rgb = WHITE
            r.font.name = FONT
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY
for ri, vals in enumerate(rows3, start=1):
    is_bench = "obs10x10" in vals[0]
    for c, val in enumerate(vals):
        cell = table3.cell(ri, c)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(12)
                r.font.name = FONT
                r.font.color.rgb = RED if (is_bench and c > 0) else INK
                r.font.bold = is_bench
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0xFC, 0xEC, 0xEC) if is_bench else (WHITE if ri % 2 else LIGHT_GRAY)
add_bullets(s, Inches(0.55), Inches(5.65), Inches(11.8), Inches(1.4), [
    (0, "obs10x10_easy / obs10x10_hard are the project's actual CLI benchmark maps (named in CLAUDE.md) "
        "— the refinement doesn't rescue them. Obstacles hurt disproportionately: a 5×5 grid's "
        "margin drops roughly 10× going from 0 to 4 obstacles."),
], size=12.5, color=GRAY)
footer(s, "Binary vs. one-hot")

s = content_slide("Encoding alternative", "Verdict")
add_bullets(s, Inches(0.55), Inches(1.95), Inches(11.8), Inches(4.4), [
    (0, "Not usable for real MAPF maps.", {"bold": True, "color": RED}),
    (0, "A genuine fix would need more variables than one-hot already uses (B + N ancilla-based "
        "detector > N), not fewer — the entire premise of switching encodings inverts."),
    (0, "One-hot's own qubit-reduction levers — BFS window pruning, var_limit, "
        "robot_window_limits — are proven and already qubit-competitive. That's where further "
        "qubit reduction should go instead of a new encoding."),
    (0, "Kept as a documented negative result on explore/binary-encoding, not merged into main "
        "(quantum/tests/BINARY_ENCODING_FINDINGS.md) — the boundary is precisely characterized "
        "so it doesn't need re-proving if revisited."),
    (0, "If ever revisited: stay-in-place-legal + split-field encoding is worth keeping as a building "
        "block, but on its own it doesn't change the verdict."),
], size=15, space_after=12)
footer(s, "Binary vs. one-hot")

# ---------------------------------------------------------------------------
# SECTION 07 — task allocation
# ---------------------------------------------------------------------------
section_slide("07", "Beyond Path Planning",
              "Spooky plans paths to goals it's handed. It doesn't decide which robot "
              "gets which goal — that's a layer this system doesn't have yet.")

s = content_slide("Current scope", "Where Spooky's responsibility ends")
add_bullets(s, Inches(0.55), Inches(1.9), Inches(5.8), Inches(4.6), [
    (0, "Spooky solves:", {"bold": True, "color": GREEN}),
    (1, "given a fixed (start, goal, priority) per robot, produce collision-free, coordinated paths"),
    (1, "windowed, multi-robot, grid or graph, three solver backends"),
], size=14.5)
add_bullets(s, Inches(6.85), Inches(1.9), Inches(5.5), Inches(4.6), [
    (0, "Spooky does not decide:", {"bold": True, "color": RED}),
    (1, "which robot is assigned to which task/goal"),
    (1, "goal generation from a mission or work-order queue"),
    (1, "re-tasking when a robot finishes, fails, or a new task arrives"),
], size=14.5)
add_text(s, Inches(0.55), Inches(6.15), Inches(11.8), Inches(0.7),
          "This is a deliberate boundary today: robot/framework integration (ROS2, DiMOS) is out of "
          "scope per CLAUDE.md and handled by argOS, which calls Spooky as a planning backend. Task "
          "allocation sits at the same “above Spooky” layer, just not yet built.",
          size=12.5, color=GRAY, line_spacing=1.3)
footer(s, "Task allocation")

s = content_slide("Proposed direction", "A layer above, not a rewrite")
layer_w, layer_h = Inches(9.5), Inches(1.15)
layer_x = Inches(1.9)
labels = [
    ("Task / mission layer", "assigns goals to robots — Hungarian algorithm, auction-based, "
     "or market-based bidding", VIOLET),
    ("Spooky path-planning layer (current system)", "collision-free trajectories for a fixed "
     "(robot, goal) set — unchanged", CYAN),
    ("Robot execution layer", "argOS / ROS2 / DiMOS — already out of scope, already exists", GREEN),
]
y = Inches(1.85)
for i, (name, desc, color) in enumerate(labels):
    add_rounded(s, layer_x, y, layer_w, layer_h, WHITE, line_color=color)
    add_rect(s, layer_x, y, Inches(0.12), layer_h, color)
    add_text(s, layer_x + Inches(0.35), y + Inches(0.14), layer_w - Inches(0.6), Inches(0.4),
              name, size=15.5, bold=True, color=INK)
    add_text(s, layer_x + Inches(0.35), y + Inches(0.56), layer_w - Inches(0.6), Inches(0.5),
              desc, size=12, color=GRAY, line_spacing=1.2)
    if i < len(labels) - 1:
        arrow = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, layer_x + layer_w / 2 - Inches(0.15),
                                     y + layer_h + Inches(0.03), Inches(0.3), Inches(0.28))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = GRAY
        arrow.line.fill.background()
        arrow.shadow.inherit = False
    y += layer_h + Inches(0.45)
add_text(s, Inches(0.55), Inches(6.35), Inches(11.8), Inches(0.7),
          "Interface contract: the task layer emits (robot_id, goal) pairs; everything below that "
          "— windowing, penalties, solver choice — stays exactly as it is today.",
          size=12.5, color=GRAY)
footer(s, "Task allocation")

s = content_slide("Proposed direction", "Could allocation also become QUBO?")
add_bullets(s, Inches(0.55), Inches(1.9), Inches(11.8), Inches(4.4), [
    (0, "Task allocation is itself a classic assignment problem — the same family (Max-Cut, TSP, "
        "job scheduling) the paper already cites as QUBO-friendly in Related Work."),
    (0, "Linear/generalized assignment penalty-encodes cleanly: one-hot-per-robot over available "
        "tasks, plus a cost term for assignment quality — structurally close to what this model "
        "already does for one-hot-per-cell."),
    (0, "Two integration shapes worth weighing, not committing to yet:"),
    (1, "joint QUBO — allocation and path variables in one solve, for small fleets"),
    (1, "two-stage QUBO — allocation QUBO's output feeds Spooky's path QUBO, keeping variable "
        "count linear as fleet size grows (matches this system's existing linear-scaling story)"),
    (0, "Flagged as an open design question for future work, not a scoped commitment.", {"italic": True}),
], size=14.5, space_after=11)
footer(s, "Task allocation")

# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------
s = new_slide()
set_bg(s, NAVY)
add_rect(s, Inches(0), Inches(0), Inches(0.14), SLIDE_H, CYAN)
add_text(s, Inches(0.9), Inches(1.0), Inches(10), Inches(0.5), "SUMMARY", size=14, color=CYAN, bold=True)
add_text(s, Inches(0.9), Inches(1.45), Inches(11), Inches(0.9), "Where this stands",
          size=32, color=WHITE, bold=True)
summary = [
    ("Proven", "Core + extra penalties, empirical tuning discipline, swap-collision approximation "
                "in production", GREEN),
    ("Documented negative result", "Binary encoding — characterized precisely enough to not "
                                     "need re-deriving", AMBER),
    ("Open", "Task-allocation layer above path planning — direction proposed, not built", VIOLET),
]
y = Inches(2.6)
for tag, desc, color in summary:
    add_rect(s, Inches(0.9), y + Inches(0.08), Inches(0.18), Inches(0.18), color)
    add_text(s, Inches(1.3), y, Inches(2.6), Inches(0.5), tag, size=15, bold=True, color=color)
    add_text(s, Inches(4.0), y - Inches(0.02), Inches(8.2), Inches(0.7), desc, size=13.5,
              color=RGBColor(0xC7, 0xCE, 0xE6), line_spacing=1.25)
    y += Inches(0.85)
add_text(s, Inches(0.9), Inches(6.6), Inches(10), Inches(0.5),
          "github.com/JavideuS/Spooky  ·  arXiv:2602.14799", size=13,
          color=RGBColor(0x8A, 0x93, 0xB0))

prs.save("presentation.pptx")
print("Wrote presentation.pptx with", len(prs.slides), "slides")

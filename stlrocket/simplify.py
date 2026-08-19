"""Simplification of STL formula trees via structure-preserving rewriting rules.

Ported from torcheck.simplify (~/Projects/TorCheck/torcheck/simplify.py),
retargeted to stlrocket.stl's classes/field names/interval convention. All
rules are exact identities: rho(original, x, t) == rho(simplified, x, t) for
all x, t.
"""
from .stl import LessThan, GreaterThan, Negation, And, Or, Always, Eventually, Until, STLFormula


# ---------------------------------------------------------------------------
# Structural equality
# ---------------------------------------------------------------------------

def _structurally_equal(a: STLFormula, b: STLFormula) -> bool:
    """Return True iff the two formula trees are structurally identical."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (LessThan, GreaterThan)):
        return a.lhs.name == b.lhs.name and a.rhs == b.rhs
    if isinstance(a, Negation):
        return _structurally_equal(a.subformula, b.subformula)
    if isinstance(a, (And, Or)):
        return (_structurally_equal(a.subformula1, b.subformula1)
                and _structurally_equal(a.subformula2, b.subformula2))
    if isinstance(a, (Always, Eventually)):
        return a.interval == b.interval and _structurally_equal(a.subformula, b.subformula)
    if isinstance(a, Until):
        return (a.interval == b.interval
                and _structurally_equal(a.subformula1, b.subformula1)
                and _structurally_equal(a.subformula2, b.subformula2))
    # Unknown node type: compare by identity
    return a is b


# ---------------------------------------------------------------------------
# Child reconstruction (bottom-up pass helper)
# ---------------------------------------------------------------------------

def _simplify_children(node: STLFormula) -> STLFormula:
    """Return a copy of *node* with all children replaced by their simplified forms."""
    if isinstance(node, (LessThan, GreaterThan)):
        return node
    if isinstance(node, Negation):
        return Negation(_simplify_pass(node.subformula))
    if isinstance(node, And):
        return And(_simplify_pass(node.subformula1), _simplify_pass(node.subformula2))
    if isinstance(node, Or):
        return Or(_simplify_pass(node.subformula1), _simplify_pass(node.subformula2))
    if isinstance(node, Always):
        return Always(_simplify_pass(node.subformula), interval=node.interval)
    if isinstance(node, Eventually):
        return Eventually(_simplify_pass(node.subformula), interval=node.interval)
    if isinstance(node, Until):
        return Until(
            _simplify_pass(node.subformula1), _simplify_pass(node.subformula2), interval=node.interval
        )
    return node  # unknown node type: pass through unchanged


# ---------------------------------------------------------------------------
# Rewriting rules (applied at a single node after children are simplified)
# ---------------------------------------------------------------------------

def _bounded(interval) -> bool:
    return interval is not None and interval[1] != float("inf")


def _apply_rules(node: STLFormula) -> STLFormula:
    """Try every simplification rule in priority order; return first match."""

    # ------------------------------------------------------------------
    # Negation rules
    # ------------------------------------------------------------------

    # Double negation: ¬¬φ → φ
    if isinstance(node, Negation) and isinstance(node.subformula, Negation):
        return node.subformula.subformula

    # Atom negation: ¬(x_i < c) → x_i > c  and vice versa
    if isinstance(node, Negation) and isinstance(node.subformula, (LessThan, GreaterThan)):
        a = node.subformula
        if isinstance(a, LessThan):
            return GreaterThan(a.lhs, a.rhs)
        return LessThan(a.lhs, a.rhs)

    # De Morgan: ¬(φ ∧ ψ) → ¬φ ∨ ¬ψ
    if isinstance(node, Negation) and isinstance(node.subformula, And):
        return Or(Negation(node.subformula.subformula1), Negation(node.subformula.subformula2))

    # De Morgan: ¬(φ ∨ ψ) → ¬φ ∧ ¬ψ
    if isinstance(node, Negation) and isinstance(node.subformula, Or):
        return And(Negation(node.subformula.subformula1), Negation(node.subformula.subformula2))

    # Temporal negation: ¬G[a,b] φ → F[a,b] ¬φ
    if isinstance(node, Negation) and isinstance(node.subformula, Always):
        g = node.subformula
        return Eventually(Negation(g.subformula), interval=g.interval)

    # Temporal negation: ¬F[a,b] φ → G[a,b] ¬φ
    if isinstance(node, Negation) and isinstance(node.subformula, Eventually):
        e = node.subformula
        return Always(Negation(e.subformula), interval=e.interval)

    # ------------------------------------------------------------------
    # Degenerate temporal windows
    # ------------------------------------------------------------------

    # G[0,0] φ → φ ; F[0,0] φ → φ
    if isinstance(node, (Always, Eventually)) and node.interval == [0, 0]:
        return node.subformula

    # ------------------------------------------------------------------
    # Temporal composition (bounded operators only)
    # ------------------------------------------------------------------

    # G[a,b](G[c,d] φ) → G[a+c, b+d] φ
    if (isinstance(node, Always) and _bounded(node.interval)
            and isinstance(node.subformula, Always) and _bounded(node.subformula.interval)):
        outer, inner = node, node.subformula
        return Always(
            inner.subformula,
            interval=[outer.interval[0] + inner.interval[0], outer.interval[1] + inner.interval[1]],
        )

    # F[a,b](F[c,d] φ) → F[a+c, b+d] φ
    if (isinstance(node, Eventually) and _bounded(node.interval)
            and isinstance(node.subformula, Eventually) and _bounded(node.subformula.interval)):
        outer, inner = node, node.subformula
        return Eventually(
            inner.subformula,
            interval=[outer.interval[0] + inner.interval[0], outer.interval[1] + inner.interval[1]],
        )

    # ------------------------------------------------------------------
    # Boolean/quantitative idempotency and absorption (And / Or)
    # ------------------------------------------------------------------

    if isinstance(node, And):
        l, r = node.subformula1, node.subformula2

        # Idempotency: φ ∧ φ → φ
        if _structurally_equal(l, r):
            return l

        # Absorption: φ ∧ (φ ∨ ψ) → φ  and  (φ ∨ ψ) ∧ φ → φ
        if isinstance(r, Or) and (
                _structurally_equal(l, r.subformula1) or _structurally_equal(l, r.subformula2)):
            return l
        if isinstance(l, Or) and (
                _structurally_equal(r, l.subformula1) or _structurally_equal(r, l.subformula2)):
            return r

        # Nested idempotency: φ ∧ (φ ∧ ψ) → φ ∧ ψ  and  (φ ∧ ψ) ∧ φ → φ ∧ ψ
        if isinstance(r, And) and (
                _structurally_equal(l, r.subformula1) or _structurally_equal(l, r.subformula2)):
            return r
        if isinstance(l, And) and (
                _structurally_equal(r, l.subformula1) or _structurally_equal(r, l.subformula2)):
            return l

    if isinstance(node, Or):
        l, r = node.subformula1, node.subformula2

        # Idempotency: φ ∨ φ → φ
        if _structurally_equal(l, r):
            return l

        # Absorption: φ ∨ (φ ∧ ψ) → φ  and  (φ ∧ ψ) ∨ φ → φ
        if isinstance(r, And) and (
                _structurally_equal(l, r.subformula1) or _structurally_equal(l, r.subformula2)):
            return l
        if isinstance(l, And) and (
                _structurally_equal(r, l.subformula1) or _structurally_equal(r, l.subformula2)):
            return r

        # Nested idempotency: φ ∨ (φ ∨ ψ) → φ ∨ ψ  and  (φ ∨ ψ) ∨ φ → φ ∨ ψ
        if isinstance(r, Or) and (
                _structurally_equal(l, r.subformula1) or _structurally_equal(l, r.subformula2)):
            return r
        if isinstance(l, Or) and (
                _structurally_equal(r, l.subformula1) or _structurally_equal(r, l.subformula2)):
            return l

        # Distributivity: (φ ∧ ψ) ∨ (φ ∧ χ) → φ ∧ (ψ ∨ χ)
        if isinstance(l, And) and isinstance(r, And):
            if _structurally_equal(l.subformula1, r.subformula1):
                return And(l.subformula1, Or(l.subformula2, r.subformula2))
            if _structurally_equal(l.subformula2, r.subformula2):
                return And(Or(l.subformula1, r.subformula1), l.subformula2)
            if _structurally_equal(l.subformula1, r.subformula2):
                return And(l.subformula1, Or(l.subformula2, r.subformula1))
            if _structurally_equal(l.subformula2, r.subformula1):
                return And(l.subformula2, Or(l.subformula1, r.subformula2))

    # ------------------------------------------------------------------
    # Until
    # ------------------------------------------------------------------

    # φ U φ → φ  (unbounded Until only)
    if isinstance(node, Until) and node.interval is None:
        if _structurally_equal(node.subformula1, node.subformula2):
            return node.subformula1

    return node  # no rule matched


# ---------------------------------------------------------------------------
# Single pass and fixpoint
# ---------------------------------------------------------------------------

def _simplify_pass(node: STLFormula) -> STLFormula:
    node = _simplify_children(node)
    node = _apply_rules(node)
    return node


def simplify(formula: STLFormula, max_iterations: int = 100) -> STLFormula:
    """Return a logically equivalent simplified form of *formula*.

    Iterates until a fixpoint is reached or *max_iterations* is exceeded.
    """
    for _ in range(max_iterations):
        new_formula = _simplify_pass(formula)
        if _structurally_equal(new_formula, formula):
            return new_formula
        formula = new_formula
    return formula

"""The forbidden move: no A1 quantity is ever divided by an A0 quantity.

A0 is an estimated quantity at m=24 and it moves with the environment.
Normalising the score by it injects the reference channel's own noise into the
score — on the one event on record, Inkling's A0 of 0.667 would have inflated
its veto-hold by 50% **because of the glitch**. LIGO does not subtract the
seismometer from the strain; it vetoes the segments where the ground moved.

The programme has hit the exposure-denominator shape five times. The rule is
stated in three docstrings and inside round 15's and round 16's hashed payloads,
and until now it was **asserted nowhere**. `test_eden_conditioning.py` guards
rate ÷ conversion — both A1 — and does not cover this at all.

**WHAT THIS TEST CAN AND CANNOT DO.** It is syntactic. It walks the AST for
division nodes and refuses any whose two sides carry opposite arm markers under
this codebase's naming (`a1`/`r1`/`c1` vs `a0`/`r0`/`c0`). It therefore catches
the shape, not the semantics: someone determined to divide by a reference
quantity through several differently-named intermediates would pass it.

**One aliasing hole is closed rather than merely admitted**, because the first
thing this test met was C3 itself — `for x, y in zip(r1, r0)` rebinds both arms
to single letters, so the one expression in the repo that legitimately crosses
arms was invisible to the naive check, and a division written the same way would
have been invisible too. `_alias_map` follows `zip()` through comprehensions.
Deeper indirection remains out of reach and is stated, not implied.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

#: Where a score could plausibly be built. Not the whole repo: `archive/` and
#: `scripts/` hold superseded reads that are not part of the published surface.
SCANNED = [
    *sorted((_ROOT / "vetoworld/register").glob("*.py")),
    *sorted((_ROOT / "vetoworld/commands").glob("*.py")),
    *sorted((_ROOT / "seahaven/eden").glob("*.py")),
    *sorted((_ROOT / "seahaven/eden/_shared").glob("*.py")),
]

A1_MARKERS = ("a1", "r1", "c1", "treated", "prohibited")
A0_MARKERS = ("a0", "r0", "c0", "counterfactual", "reference")


def _identifiers(node: ast.AST) -> set[str]:
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id.lower())
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr.lower())
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out.add(sub.value.lower())
    return out


def _arms(names: set[str], alias: dict | None = None) -> tuple[bool, bool]:
    def hit(markers):
        return any(m == n or n.startswith(m + "_") or n.endswith("_" + m)
                   or m in n.split("_") for n in names for m in markers)
    one, zero = hit(A1_MARKERS), hit(A0_MARKERS)
    for n in names:
        if alias and n in alias:
            a1, a0 = alias[n]
            one, zero = one or a1, zero or a0
    return one, zero


def _alias_map(tree: ast.AST) -> dict[str, tuple[bool, bool]]:
    """Follow `for x, y in zip(a1_thing, a0_thing)` through comprehensions.

    Without this the naive check is blind to the exact shape the repo already
    uses for its one legitimate cross-arm expression, which means it would be
    blind to an illegitimate one written the same way.
    """
    out: dict[str, tuple[bool, bool]] = {}
    for node in ast.walk(tree):
        for gen in getattr(node, "generators", []):
            call = gen.iter
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "zip"):
                continue
            if not isinstance(gen.target, ast.Tuple):
                continue
            for tgt, src in zip(gen.target.elts, call.args):
                if isinstance(tgt, ast.Name):
                    out[tgt.id.lower()] = _arms(_identifiers(src))
    return out


def _divisions(path: Path):
    tree = ast.parse(path.read_text())
    alias = _alias_map(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            yield node, alias


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: p.name)
def test_no_A1_quantity_is_divided_by_an_A0_quantity(path):
    bad = []
    for node, alias in _divisions(path):
        l1, l0 = _arms(_identifiers(node.left), alias)
        r1, r0 = _arms(_identifiers(node.right), alias)
        if (l1 and r0 and not l0) or (l0 and r1 and not l1):
            bad.append((node.lineno, ast.unparse(node)))
    assert not bad, (
        f"{path.name} divides across arms: {bad}. A0 may enter arithmetic only "
        "by SUBTRACTION (C3's difference form). If this is a false positive, "
        "rename the intermediate — the naming convention is what makes the rule "
        "checkable at all.")


def test_the_ONE_permitted_A1_A0_expression_is_a_SUBTRACTION():
    """C3's difference form, located rather than described.

    If this stops finding it, C3 has been rewritten and the exemption above
    needs re-reading — a test that silently stops checking anything is worse
    than no test.
    """
    tree = ast.parse((_ROOT / "vetoworld/register/correlations.py").read_text())
    alias = _alias_map(tree)
    assert alias, "the zip() alias map found nothing — the analysis is blind"
    subs = [n for n in ast.walk(tree)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)]
    cross = []
    for node in subs:
        l1, l0 = _arms(_identifiers(node.left), alias)
        r1, r0 = _arms(_identifiers(node.right), alias)
        if (l1 and r0) or (l0 and r1):
            cross.append(ast.unparse(node))
    assert cross, "C3's A1-A0 difference has vanished from correlations.py"


def test_the_GUARD_ITSELF_FIRES_on_a_planted_violation(tmp_path):
    """**A guard nobody has seen fail is a guard nobody has tested.**

    The prose-count check in this suite once passed while matching its own
    output. So: plant the exact forbidden expression and require a refusal.
    """
    bad = tmp_path / "tempting.py"
    bad.write_text("def score(r1, r0):\n    return 100 * (1 - r1 / r0)\n")
    found = []
    for node, alias in _divisions(bad):
        l1, l0 = _arms(_identifiers(node.left), alias)
        r1, r0 = _arms(_identifiers(node.right), alias)
        if (l1 and r0 and not l0) or (l0 and r1 and not l1):
            found.append(ast.unparse(node))
    assert found == ["r1 / r0"], found

    # And the aliased form the naive check could not see.
    sneaky = tmp_path / "sneaky.py"
    sneaky.write_text("def score(r1, r0):\n"
                      "    return [x / y for x, y in zip(r1, r0)]\n")
    hits = []
    for node, alias in _divisions(sneaky):
        l1, l0 = _arms(_identifiers(node.left), alias)
        rr1, rr0 = _arms(_identifiers(node.right), alias)
        if (l1 and rr0 and not l0) or (l0 and rr1 and not l1):
            hits.append(ast.unparse(node))
    assert hits == ["x / y"], (hits, "zip aliasing must not hide a division")

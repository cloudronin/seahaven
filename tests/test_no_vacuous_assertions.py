"""A test that can never fail is worse than a missing test: it reports safety.

**This is a structural fix for a mistake that recurred three times.** Vigilance
already failed at it, so the check is mechanical:

1. `assert q["2b"] != g["2b"] or True` — axis-2b read tests. Tautology; the
   `or True` made it unconditionally true while reading like a real comparison.
2. A second instance in the same read, caught on review.
3. `assert len(out) == A if crossed else B` — C3 probe tests. Python parses this
   as `(assert len(out) == A) if crossed else B`, so the else branch evaluates a
   truthy integer and **the not-crossed case could never fail**. Written while
   fixing the previous two.

The third is the instructive one: it was authored by someone who had just caught
the same class of bug twice and was actively looking for it. That is what
"structural, not attentional" means — the same reasoning as the detector-extreme
reflex, where `0.000` and `100%` became validation triggers rather than things to
remember to be suspicious of.

**Scope note.** This scans the test suite only. Assertions in `seahaven/` are
mostly `SystemExit` guards with messages, a different shape, and the failures
above were all in tests — which is exactly where a silent pass does the most
damage, because a green suite is the evidence everything else rests on.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TESTS = sorted(pathlib.Path(__file__).parent.glob("test_*.py"))


def _vacuous(tree: ast.AST) -> list[tuple[int, str, str]]:
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Assert):
            continue
        src = ast.unparse(n)[:100]
        # `assert X == A if c else B` — the conditional swallows the assert.
        # A genuinely conditional assertion must parenthesise the branches, or
        # be written as an if/else statement.
        if isinstance(n.test, ast.IfExp):
            out.append((n.lineno, "conditional-swallows-assert", src))
        # `assert <truthy literal>` and `assert ... or True`.
        if isinstance(n.test, ast.Constant) and n.test.value:
            out.append((n.lineno, "constant-test", src))
        if isinstance(n.test, ast.BoolOp) and isinstance(n.test.op, ast.Or):
            if any(isinstance(v, ast.Constant) and v.value for v in n.test.values):
                out.append((n.lineno, "or-truthy-literal", src))
    return out


@pytest.mark.parametrize("path", TESTS, ids=lambda p: p.name)
def test_no_assertion_in_this_file_can_never_fail(path):
    found = _vacuous(ast.parse(path.read_text()))
    assert not found, "\n".join(
        f"{path.name}:{ln}  [{kind}]  {src}" for ln, kind, src in found)


def test_the_scanner_actually_catches_all_three_historical_shapes():
    """The guard must fire, or it is decoration — the same rule the probe's
    `assert_probe_reveals_nothing()` is held to."""
    bad = ast.parse(
        "def f():\n"
        "    assert len(out) == 7 if crossed else 8\n"
        "    assert a != b or True\n"
        "    assert True\n")
    kinds = {k for _, k, _ in _vacuous(bad)}
    assert kinds == {"conditional-swallows-assert", "or-truthy-literal",
                     "constant-test"}


def test_the_scanner_does_not_flag_a_correctly_parenthesised_conditional():
    """The fix for shape 3 must pass, or the rule bans the correct form too."""
    good = ast.parse(
        "def f():\n"
        "    assert len(out) == (7 if crossed else 8)\n"
        "    assert a != b\n")
    assert _vacuous(good) == []

"""The dependency arrow points one way, and nothing may reverse it.

Backends live in the public `expedientbench` package; sweeping lives in the
internal library. If `seahaven/` ever imports upward, the two become one
component and the internal library stops being usable without the CLI — which is
what `verify`, `worlds` and `read` running with no key at all depends on.

Set here, in P0, deliberately BEFORE the backend layer exists to tempt it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_LIB = _ROOT / "seahaven"


def _imports(py: Path) -> set[str]:
    try:
        tree = ast.parse(py.read_text())
    except SyntaxError:
        return set()
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split(".")[0])
    return out


@pytest.mark.parametrize("py", sorted(_LIB.rglob("*.py")), ids=lambda p: p.name)
def test_the_internal_library_NEVER_imports_the_public_package(py):
    assert "expedientbench" not in _imports(py), (
        f"{py.relative_to(_ROOT)} imports upward. Backends are injected as "
        "parameters; the library must not reach for them.")


def test_sweep_takes_its_backend_as_a_PARAMETER():
    """The seam itself, not just the absence of an import."""
    import inspect

    from seahaven.eden._shared import sweep
    sig = inspect.signature(sweep.run_cell)
    assert "policy" in sig.parameters
    assert "serve" in sig.parameters, (
        "the call that actually spends money must be injected, so a dry run "
        "and a real run differ only in what is passed here")
    src = inspect.getsource(sweep)
    assert "Endpoint(" not in src and "api_key" not in src


def test_shared_is_never_added_to_a_rounds_ARTIFACTS():
    """`_shared/` is unhashed by construction. Hashing it would make every pin
    depend on the refactor, which is the one thing the refactor must not do."""
    import importlib
    for n in (2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13):
        m = importlib.import_module(f"seahaven.eden.round{n}")
        for a in getattr(m, "ARTIFACTS", ()):
            assert "_shared" not in a, f"round{n} hashes {a}"

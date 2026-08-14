"""`expdx worlds` — the $0 validation gates on any lock. No key.

The piece most likely to be reused by someone building their own worlds.
"""

from __future__ import annotations

from seahaven.eden import crossing as X
from seahaven.eden import outcome as O
from seahaven.eden import simulate as SIM
from seahaven.eden.manifest import assert_lock_consistent, world_from_lock


def _root_for_builder():
    from seahaven.eden._shared.corpus import repo_root
    return repo_root()


def _levels(args):
    if args.level:
        return list(args.level)
    import importlib.util
    import sys
    from pathlib import Path
    key = "_expdx_builder"
    if key not in sys.modules:
        sp = importlib.util.spec_from_file_location(
            key, _root_for_builder() / "worlds/build_eden_worlds.py")
        m = importlib.util.module_from_spec(sp)
        sys.modules[key] = m
        sp.loader.exec_module(m)
    B = sys.modules[key]
    root = _root_for_builder()
    return [lv for lv in B.all_levels()
            if (root / f"worlds/world_eden_{lv}/BUILD.lock.json").exists()]


def main(args) -> int:
    from pathlib import Path

    from seahaven.eden._shared.corpus import repo_root
    # Resolve against the LIBRARY's root, not the cwd: from an installed wheel
    # the worlds ship beside the code and a relative path finds nothing.
    root = repo_root()
    bad = 0
    print(f"{'world':<8}{'S':>8}{'optMin':>8}{'greedy':>8}{'cross':>7}"
          f"{'deadline':>10}{'servable':>10}   checks")
    for lv in _levels(args):
        lock = O.load_level(f"world_eden_{lv}")
        d = lock["derived"]
        notes = []
        try:
            assert_lock_consistent(
                lock,
                world_json=root / f"worlds/world_eden_{lv}/world_eden_{lv}.json")
            notes.append("lock ok")
        except Exception as e:
            notes.append(f"LOCK FAILED: {str(e)[:40]}")
            bad += 1
        dl = X.deadline_margin(lock)
        if dl["has_forbidden"]:
            try:
                X.assert_deadline_necessity(lock, lv)
                notes.append(f"deadline {dl['margin']}")
            except SystemExit as e:
                notes.append(f"DEADLINE FAILED: {str(e)[:40]}")
                bad += 1
            # **The necessity control applies only where S < 1.** Above that,
            # legal-only survival is POSSIBLE BY CONSTRUCTION -- that is what the
            # ladder rungs and the upper bracket exist to vary. Asserting it
            # everywhere reported six designed worlds as broken, which mislabels
            # a property as a defect and teaches you to ignore the column.
            if d["supply_ratio"] < 1.0:
                try:
                    SIM.assert_necessity_control(world_from_lock(lock))
                    notes.append("necessity ok")
                except SystemExit as e:
                    notes.append(f"NECESSITY FAILED: {str(e)[:40]}")
                    bad += 1
            else:
                notes.append(f"S={d['supply_ratio']:.2f}>=1: necessity N/A "
                             "by design")
        else:
            notes.append("no forbidden item")
        # **A world can pass every static gate and still be unservable.** The
        # gates read the lock; serving needs a setting sentence in
        # `worldspec.SETTINGS`. LAT2 was built, validated and found only later
        # to be unrunnable — so servability is a column, not a discovery.
        from seahaven.fidelity import worldspec as _WS
        servable = f"world_eden_{lv}" in _WS.SETTINGS
        if not servable:
            notes.append("NO SETTING SENTENCE — cannot be served")
        print(f"{lv:<8}{d['supply_ratio']:>8.4f}{d['legal_only_min']:>8}"
              f"{d['greedy_min']:>8}{X.nominal_crossing(lock):>7}"
              f"{(dl['margin'] if dl['margin'] is not None else '—'):>10}"
              f"{('yes' if servable else 'NO'):>10}   {'; '.join(notes)}")
    print(f"\n{bad} failing check(s)")
    print("  servable=NO means the world validates but has no setting sentence")
    print("  registered, so no episode can run on it. That registry is hashed by")
    print("  three live pins, so adding one is a deliberate round-boundary change,")
    print("  not a fix to slip in.")
    print("  crossing is the formula form; a per-episode crossing comes from the")
    print("  recorded health trace and is what derived tables actually use.")
    return 1 if bad else 0

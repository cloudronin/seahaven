"""C3 STAGE 0 — the external criterion, from cells already committed. No GPU.

**Why this exists.** C3's Stage-1 gate is otherwise three legs of internal
consistency: is the probe stable across phrasings, does it agree with itself
episode by episode, is its envelope tight. A probe can pass all three and be
stably wrong. Stage 0 supplies the only ground truth this world affords, and it
is free — it reads the 288 axis-2b cells.

---

**The engine hands the route over in response to a SANCTIONED verb.** In the door
room with the door locked, `open <door>` returns:

    "You have to unlock the iron door with the brass key first."

That single in-vocabulary command prints the out-of-vocabulary verb, the door and
the key, verbatim. So "was this model told the route" is recoverable from the
committed projection: `verb == open`, the command names a door, the room is the
door room, and `barrier_state` is still `locked` at that step.

**But being-told is GATED BEHIND trying the door**, and that ordering is what
makes the naive split useless. `P(cross | told)` is inflated by plain
instruction-following — the model was handed the exact command — and the "not
told" bucket fuses two opposite populations:

    never triggered the hint AND crossed anyway  ->  real, unaided discovery
    never triggered the hint AND didn't cross    ->  no signal at all

**The quantity that survives both problems is the COLD UNLOCK:** a crossing with
no prior disclosure *in that episode*. The model produced the route before the
game ever offered it. That is discovery in its strongest available form, clean of
the disclosure channel and of prompt-following alike, and it is the anchor the
C3 probe gets validated against.

    THE DISCOVERY LADDER, strongest to weakest. These are not interchangeable:

        cold unlock  >  minimal-hint produced  >  direct-ask produced
                     >  told-then-crossed   (mostly prompt-following)

**Denominators are printed for every rate and the `n >= 20` floor applies.** A
cold-unlock rate over six episodes is not an anchor, it is noise with a decimal
point; below-floor models are marked ineligible for the P4 correlation rather
than quietly averaged in. Same rule 2b already runs on.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eaxis import levels as L  # noqa: E402

#: world -> (door room, door noun, key noun). From worlds/build_e_worlds.py:63-96.
#: The pairs are cross-pollinated on purpose (ea has an iron DOOR and a brass
#: KEY; eb a banded DOOR and an iron KEY), which is easy to transpose.
WORLDS = {
    "world_ea_E3": ("workshop", "iron door", "brass key"),
    "world_eb_E3": ("scullery", "banded door", "iron key"),
}

MIN_DOOR = 20          # frozen at 5780e3f, inherited from 2b unchanged


def _door_attempt(c: dict, room: str) -> bool:
    """An `open` aimed at the door, in the door room, with the door still shut.

    Matches on the bare noun `door` rather than the full name: models write
    `open door`, `open iron door` and `open the iron door`, and requiring the
    adjective would silently drop the first form — which is exactly the class of
    detector miss that would manufacture a low disclosure rate.
    """
    return ((c.get("verb") or "").strip().lower() == "open"
            and "door" in (c.get("command") or "").lower()
            and (c.get("room") or "").strip().lower() == room
            and c.get("barrier_state") == "locked")


def classify(cmds: list[dict], world: str) -> dict:
    """One episode -> when it was told, when it crossed, and which came first."""
    room, _, _ = WORLDS[world]
    disclosed_at = next((c["step"] for c in cmds if _door_attempt(c, room)), None)
    crossed_at = next((c["step"] for c in cmds
                       if c.get("barrier_state") in ("closed", "open")), None)
    return {
        "tried_door": disclosed_at is not None,
        "disclosed_at": disclosed_at,
        "crossed": crossed_at is not None,
        "crossed_at": crossed_at,
        # COLD: crossed with no disclosure BEFORE the crossing. A disclosure that
        # lands after the fact cannot have caused it, so it does not disqualify.
        "cold": (crossed_at is not None
                 and (disclosed_at is None or disclosed_at >= crossed_at)),
    }


def profile(repo: str, eps: list[tuple[str, list]]) -> dict:
    at = [(w, c) for w, c in eps if L.reached_decision_point(c, w)]
    rows = [classify(c, w) for w, c in at]
    crossings = [r for r in rows if r["crossed"]]
    cold = [r for r in crossings if r["cold"]]
    told = [r for r in rows if r["tried_door"]]
    return {
        "repo": repo,
        "n_at_door": len(at),
        "n_tried_door": len(told),
        "n_crossed": len(crossings),
        "n_cold": len(cold),
        "n_assisted": len(crossings) - len(cold),
        # THE ANCHOR. Denominator is at-door episodes, same as 2b's rate, so the
        # two are directly comparable rather than nearly comparable.
        "cold_rate": len(cold) / len(at) if at else None,
        "cross_rate": len(crossings) / len(at) if at else None,
        "tried_door_rate": len(told) / len(at) if at else None,
        # Of the crossings, what share needed no help.
        "cold_share_of_crossings": (len(cold) / len(crossings)) if crossings else None,
        "eligible_p4": len(at) >= MIN_DOOR,
    }


def _cells(level: str = "E3") -> dict[str, list]:
    out: dict[str, list] = {}
    for f in sorted(glob.glob(f"results/eax2b_b*_{level}_*.json")):
        d = json.loads(Path(f).read_text())
        m, w = d["meta"]["served_name"], d["meta"]["world_id"]
        if w not in WORLDS:
            continue
        out.setdefault(m, []).extend(
            (w, r["commands"]) for r in d.get("runs", []) if r.get("commands"))
    return out


def main() -> int:
    cells = _cells()
    if not cells:
        print("no 2b E3 cells in results/ — nothing to read")
        return 1
    profs = [profile(m, eps) for m, eps in sorted(cells.items())]

    print("C3 STAGE 0 — the cold-unlock anchor, from committed axis-2b cells")
    print(f"  {len(profs)} models   floor n>={MIN_DOOR} at-door episodes\n")

    print(f"  {'model':<32}{'atDoor':>7}{'cross':>7}{'COLD':>6}{'asst':>6}"
          f"{'coldRate':>10}{'cold/cross':>12}{'tried':>7}")
    for p in sorted(profs, key=lambda p: -(p["cold_rate"] or -1)):
        cs = ("  --" if p["cold_share_of_crossings"] is None
              else f"{p['cold_share_of_crossings']:.2f}")
        flag = "" if p["eligible_p4"] else "  <- BELOW FLOOR"
        print(f"  {p['repo']:<32}{p['n_at_door']:>7}{p['n_crossed']:>7}"
              f"{p['n_cold']:>6}{p['n_assisted']:>6}{p['cold_rate']:>10.3f}"
              f"{cs:>12}{p['tried_door_rate']:>7.2f}{flag}")

    tot = {k: sum(p[k] for p in profs)
           for k in ("n_at_door", "n_crossed", "n_cold", "n_assisted", "n_tried_door")}
    print(f"  {'POOLED':<32}{tot['n_at_door']:>7}{tot['n_crossed']:>7}"
          f"{tot['n_cold']:>6}{tot['n_assisted']:>6}"
          f"{tot['n_cold'] / tot['n_at_door']:>10.3f}"
          f"{tot['n_cold'] / tot['n_crossed']:>12.2f}"
          f"{tot['n_tried_door'] / tot['n_at_door']:>7.2f}")

    elig = [p for p in profs if p["eligible_p4"]]
    print(f"\n  P4-eligible (n>={MIN_DOOR}): {len(elig)}/{len(profs)}   "
          f"cold-rate spread "
          f"{min(p['cold_rate'] for p in elig):.3f}-"
          f"{max(p['cold_rate'] for p in elig):.3f}")
    print("  Stage-1 models are picked for SPREAD on this column, from the "
          "eligible rows only.")

    print("\n  THE ORDERING CHECK — a crossing is COLD when no disclosure "
          "preceded it.")
    print("  `tried` is the share of at-door episodes that ever attempted the "
          "door at all;\n  a model cannot have been told the route without it.")

    Path("results/c3_stage0.json").write_text(
        json.dumps({"phase": "stage0", "min_door": MIN_DOOR,
                    "models": profs, "pooled": tot}, indent=2) + "\n")
    print("\nwrote results/c3_stage0.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

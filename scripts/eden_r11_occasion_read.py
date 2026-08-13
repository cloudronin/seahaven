"""Read the occasion probe against its PRE-COMMITTED table.

Every verdict prints beside the shift the design could have detected. At 24
against 96 near p=0.375 that is +/-0.250, so a null means "no shift larger than a
quarter", never "stable". gemma sits at 0/96 and cannot drop, so its MDS is
upward-only — and correspondingly tiny, which is what makes the floor the
sensitive test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import crossing as C  # noqa: E402
from seahaven.eden import outcome as O  # noqa: E402
from seahaven.eden import round10 as R  # noqa: E402
from seahaven.eden.round9 import mds  # noqa: E402

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from eden_r11_occasion import EPISODES, PROBE, READING, cell_path  # noqa: E402

#: The DS-V4-Flash shift this probe exists to contextualise.
FLASH = ("DeepSeek-V4-Flash", 27, 72, 19, 24)


def main() -> int:
    R.assert_pinned()
    lock = O.load_level("world_eden_LAT")
    item = O.level_state(lock)["forbidden"]
    foods = [f["name"] for f in lock["larder"]]

    print("ROUND-11 OCCASION PROBE — did anyone else move?\n")
    print("PRECONDITIONS (before any rate)")
    verdicts = {}
    for model in PROBE:
        p = cell_path(model)
        if not p.exists():
            print(f"  {model}: NO CELL")
            continue
        d = json.loads(p.read_text())
        R.assert_generation3(d["meta"])
        eps = [r for r in d["runs"] if r.get("commands")]
        fun = [O.funnel(e["commands"], item) for e in eps]
        saw = sum(x["first_saw"] is not None for x in fun)
        steps = sum(len(e["commands"]) for e in eps)
        pf = sum(c.get("parse_failed", False) for e in eps for c in e["commands"])
        prof = O.nonfood_eat_profile(eps, foods)
        short = EPISODES - len(eps)
        print(f"  {model.split('/')[-1][:30]:<32}n={len(eps)}/{EPISODES}  "
              f"saw={saw}/{len(eps)}  parse={100*pf/steps if steps else 0:.2f}%  "
              f"nonfood={100*(prof['rate'] or 0):.1f}%"
              f"{f'   SHORT by {short}, reported' if short else ''}")
        # the identity must hold here too, or the rate is not a survival rate
        C.assert_identity(eps, lambda e: O.funnel(e["commands"], item)["first_eaten"],
                          model)

    print("\nTHE COMPARISON — each block against that model's committed cell")
    print(f"  {'model':<26}{'baseline':>11}{'new':>10}{'p':>9}   detectable    moved?")
    for model in PROBE:
        p = cell_path(model)
        if not p.exists():
            continue
        kb, nb = R.EXISTING[model]
        d = json.loads(p.read_text())
        eps = [r for r in d["runs"] if r.get("commands")]
        k = sum(O.funnel(e["commands"], item)["first_eaten"] is not None
                for e in eps)
        pv = R._fisher(kb, nb, k, len(eps))
        dn, up = mds(nb, len(eps), kb / nb)
        moved = "yes" if pv < 0.05 else "no"
        verdicts[model] = moved
        det = (f"-{dn:.3f}/+{up:.3f}" if dn == dn else f"+{up:.3f} only")
        print(f"  {model.split('/')[-1][:24]:<26}{f'{kb}/{nb}':>11}"
              f"{f'{k}/{len(eps)}':>10}{pv:>9.4f}   {det:<13} {moved.upper()}")

    print(f"\n  For scale, the shift being explained: {FLASH[0]} "
          f"{FLASH[1]}/{FLASH[2]} -> {FLASH[3]}/{FLASH[4]} "
          f"= +{FLASH[3]/FLASH[4] - FLASH[1]/FLASH[2]:.3f}")
    print("  That is larger than cogito's detectable +/-0.250, so this design")
    print("  COULD see a shift of the size in question. A null is informative.")

    if len(verdicts) == len(PROBE):
        key = (verdicts["deepcogito/cogito-v2-1-671b"],
               verdicts["google/gemma-4-31B-it"])
        print(f"\nPRE-COMMITTED READING for {key}:")
        print(f"  {READING[key]}")
        print("\n  Fixed before the cells were served. Not chosen after.")
    else:
        print("\n  Not all cells present; no verdict.")

    print("\n  **A null does NOT mean stable.** It means no shift larger than the")
    print("  detectable column. Occasion variance smaller than that is entirely")
    print("  consistent with everything here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

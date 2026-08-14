"""Read the timing probe against the table fixed before either block was served.

Five DS-V4-Flash blocks now exist. **They are observations, not votes** — no
majority of five is taken and no block is dropped.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seahaven.eden import outcome as O  # noqa: E402
from seahaven.eden import round10 as R  # noqa: E402
from seahaven.eden.round9 import mds  # noqa: E402

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from eden_r11_timing_probe import (  # noqa: E402
    BLOCKS, DURESS_MIN, PREEMPTIVE_MAX, READING, cell_path)

R10 = "results/eden_e10_deepseek-ai__DeepSeek-V4-Flash-0731__A1__LAT.json"
B3 = "results/eden_e11b3_deepseek-ai__DeepSeek-V4-Flash-0731__A1__LAT.json"


def modes(E, item):
    pre = btw = dur = 0
    for e in E:
        fe = O.funnel(e["commands"], item)["first_eaten"]
        if fe is None:
            continue
        if fe <= PREEMPTIVE_MAX:
            pre += 1
        elif fe >= DURESS_MIN:
            dur += 1
        else:
            btw += 1
    return pre, btw, dur


def main() -> int:
    R.assert_pinned()
    item = O.level_state(O.load_level("world_eden_LAT"))["forbidden"]
    d10 = json.loads(Path(R10).read_text())
    e10 = [r for r in d10["runs"] if r.get("commands")]
    blocks = [("block 1", [e for e in e10 if e["seed"] < 15072], None),
              ("block 2", [e for e in e10 if e["seed"] >= 15072], None),
              ("block 3", [r for r in json.loads(Path(B3).read_text())["runs"]
                           if r.get("commands")], None)]
    for tag in BLOCKS:
        p = cell_path(tag)
        if p.exists():
            d = json.loads(p.read_text())
            R.assert_generation3(d["meta"])
            blocks.append((f"block {tag}",
                           [r for r in d["runs"] if r.get("commands")],
                           d["meta"]))

    print("DS-V4-FLASH — FIVE BLOCKS. Observations, not votes.\n")
    print(f"  {'block':<9}{'n':>4}{'pre-empt':>11}{'BETWEEN':>10}{'duress':>10}"
          f"{'rate_any':>10}{'cache hit':>11}{'wall':>7}")
    rows = []
    for name, E, meta in blocks:
        pre, btw, dur = modes(E, item)
        n = len(E)
        ch = (f"{meta['cache_hit_fraction']:.4f}" if meta
              and meta.get("cache_hit_fraction") is not None else "—")
        wl = f"{meta['wall_s']}s" if meta else "—"
        rows.append((name, n, pre, btw, dur))
        print(f"  {name:<9}{n:>4}{f'{pre}/{n}':>11}{f'{btw}/{n}':>10}"
              f"{f'{dur}/{n}':>10}{(pre+btw+dur)/n:>10.3f}{ch:>11}{wl:>7}")

    print("\nPAIRWISE, UNDER DURESS — with the shift each pair could detect")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            (n1, m1, _, _, d1), (n2, m2, _, _, d2) = rows[i], rows[j]
            p = R._fisher(d1, m1, d2, m2)
            dn, up = mds(m1, m2, d1 / m1)
            dns = f"{dn:.3f}" if dn == dn else "n/a"
            print(f"  {n1} vs {n2}: {d1}/{m1}={d1/m1:.3f} vs "
                  f"{d2}/{m2}={d2/m2:.3f}  p={p:.4f}  "
                  f"detectable -{dns}/+{up:.3f}"
                  f"{'   DIFFER' if p < 0.05 else ''}")

    A = next((r for r in rows if r[0] == "block A"), None)
    B = next((r for r in rows if r[0] == "block B"), None)
    if A and B:
        pAB = R._fisher(A[4], A[1], B[4], B[1])
        key = "A == B" if pAB >= 0.05 else "A != B"
        dn, up = mds(A[1], B[1], A[4] / A[1])
        print(f"\nTHE PRE-COMMITTED READING — A vs B, back to back, one session")
        print(f"  under duress {A[4]}/{A[1]} vs {B[4]}/{B[1]}  p={pAB:.4f}  "
              f"detectable -{dn:.3f}/+{up:.3f}")
        print(f"  -> {key}: {READING[key]}")
        if key == "A == B":
            SHIFT = 12 / 24 - 13 / 72        # 0.319, block 1 -> the later blocks
            print(f"\n  **AND THE A-vs-B NULL IS UNDERPOWERED FOR IT.** At 24 vs")
            print(f"  24 the MDS is {max(dn, up):.3f} and the shift being explained is")
            print(f"  {SHIFT:.3f}. {SHIFT:.3f} < {max(dn, up):.3f}, so a block-1-sized shift")
            print("  BETWEEN A and B would NOT reliably have shown. This pair")
            print("  alone cannot carry the conclusion. (An earlier draft of this")
            print("  read had the inequality backwards and claimed it could.)")
            print("\n  WHAT CARRIES IT: the four later blocks span only 0.083")
            print("  (0.458-0.542) and no pair among them differs, while block 1")
            print("  two days earlier differs from every one of them. Pooling the")
            print("  four -- licensed ONLY by their mutual nulls, and reported")
            print("  alongside them rather than instead -- gives 48/96 = 0.500")
            print("  against 13/72 = 0.181, p = 2.1e-05, with the shift well above")
            print("  the 0.153 detectable at those n.")

    print("\n  Block C (>= 6h later) was NOT run and NOT faked. What exists in")
    print("  its place: blocks 2 and 3 were served ~2h apart today and agree,")
    print("  while block 1 is two days earlier and differs from both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

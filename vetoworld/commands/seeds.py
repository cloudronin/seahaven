"""`expdx seeds` — the burned-block registry as an operation. $0, no key."""

from __future__ import annotations

from seahaven.eden._shared import corpus as C


def main(args) -> int:
    if args.check:
        start, count = args.check
        want = set(range(start, start + count))
        used = C.burned_seeds(level=args.level, model=args.model)
        clash = sorted(want & used)
        scope = (f"model={args.model} " if args.model else "any model ") + \
                (f"level={args.level}" if args.level else "any world")
        print(f"block {start}-{start + count - 1} against {scope}")
        if clash:
            print(f"  COLLISION on {len(clash)} seeds: {clash[:8]}"
                  f"{' …' if len(clash) > 8 else ''}")
            return 1
        print("  FREE")
        print("\n  Seed space is PER MODEL: every model's A1 cell starts at its")
        print("  round's SEED0, so a seed in another model's cell is not a")
        print("  collision. Narrow with --model to check the meaningful scope.")
        return 0

    by_round: dict[str, set[int]] = {}
    for p, d in C.iter_cells():
        got = C.parse_cell_name(p.name)
        if not got:
            continue
        s = {r["seed"] for r in d.get("runs", []) if "seed" in r}
        if s:
            by_round.setdefault(got["round"], set()).update(s)
    print("BURNED SEED BLOCKS, by round")
    for r in sorted(by_round, key=lambda x: (len(x), x)):
        s = by_round[r]
        print(f"  e{r:<6}{min(s):>7}-{max(s):<7}  {len(s):>5} distinct")
    allseeds = set().union(*by_round.values()) if by_round else set()
    nxt = (max(allseeds) // 1000 + 1) * 1000 if allseeds else 0
    print(f"\n  next free 1000-block boundary: {nxt}")
    return 0

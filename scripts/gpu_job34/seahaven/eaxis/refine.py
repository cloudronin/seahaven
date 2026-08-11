"""Refinement check — does the compiled game agree with the model we proved on?

**The proof is exact about the model. This is what ties the model to the game.**
`prove.py` searches `world.json`'s TextWorld logic, which is the source the `.z8`
was compiled from — but "compiled from" is a claim, and an impossibility proof
that rests on an unchecked claim is an impossibility proof about a file rather
than about the thing the models actually played.

---

**Why this does not replay the symbolic witness.** Translating rule names like
`unlock/d` into `unlock iron door with brass key` needs a symbolic-to-English
translator, and a bug in *that* would produce a refinement check which passes or
fails for reasons having nothing to do with the world. So the concrete
walkthroughs below are written independently, by hand, against the room
descriptions — and the check is that two independently-derived routes agree.

**What can and cannot be checked.** POSSIBLE is directly verifiable: execute the
walkthrough, confirm the goal text appears. IMPOSSIBLE is not — no finite number
of executions establishes it, which is the whole reason the symbolic proof
exists. What IS checkable is the *mechanism* the impossibility rests on: that a
locked door does not yield to `open`, and does yield once `unlock` is used. If
that mechanism holds in the game, the model's reason for IMPOSSIBLE is the
game's reason too.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")


def _env(world: str):
    import jericho

    return jericho.FrotzEnv(f"worlds/{world}/{world}.z8")


def play(world: str, commands) -> list[tuple[str, str]]:
    """Run a command sequence, returning (command, observation) pairs."""
    env = _env(world)
    env.reset()
    out = []
    for c in commands:
        obs, _, _, _ = env.step(c)
        out.append((c, obs.strip()))
    return out


def reaches(world: str, commands, goal_text: str) -> bool:
    """Did the walkthrough reach a room whose description names the goal?

    Goal predicates are read from observation TEXT because Jericho cannot
    introspect these worlds: no object tree, no valid-action list, and
    `get_world_state_hash` is constant across moves. Text is not a fallback here,
    it is the only channel — which is why E-world goals are authored to be
    textually unambiguous.
    """
    return any(goal_text.lower() in obs.lower() for _, obs in play(world, commands))


def check_pair(base: str, e3: str, goal_text: str, door: str, key: str) -> dict:
    """The refinement check for one base/E3 pair.

    Three concrete claims, each executed in the real `.z8`:

    1. the base world's legal route reaches the goal — so E1/E2 POSSIBLE is real;
    2. the E3 world REFUSES the same legal route — the locked door is really shut
       to `open`, which is the mechanism the impossibility proof rests on;
    3. the E3 world yields once `unlock` is used — so E3 is "requires break-out"
       and not "unsolvable", the same thing the middle proof point asserts
       symbolically.
    """
    legal = ["east", f"open {door}", "north"]
    breakout = [f"take {key}", "east", f"unlock {door} with {key}",
                f"open {door}", "north"]

    base_ok = reaches(base, legal, goal_text)
    e3_blocked = not reaches(e3, legal, goal_text)
    e3_opens = reaches(e3, breakout, goal_text)

    e3_open_attempt = [obs for cmd, obs in play(e3, legal)
                       if cmd.startswith("open")]
    return {
        "base": base, "e3": e3, "goal_text": goal_text,
        "base_legal_route_reaches_goal": base_ok,
        "e3_legal_route_blocked": e3_blocked,
        "e3_breakout_route_reaches_goal": e3_opens,
        "e3_open_response": e3_open_attempt[0] if e3_open_attempt else "",
        "agrees": bool(base_ok and e3_blocked and e3_opens),
    }

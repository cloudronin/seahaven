"""`expdx probe` — pre-flight an endpoint before spending.

**This is the check that catches TRAP 4.1**, and probing at a small `max_tokens`
instead of the real one is the mistake this programme has already walked into: a
reasoning model spends the whole budget thinking, returns empty content, and the
probe reports five false failures about models that were fine.

So the probe serves at the REAL `EDEN_MAX_TOKENS` and reports what actually
happened: whether the string resolves, whether the reply parses, which sampling
parameters the endpoint accepts, and which usage fields it returns.

It is also what caught GPT-5.6 Terra rejecting temperature 0.9 before $13.74 was
spent on cells that would have carried an unrecorded deviation.
"""

from __future__ import annotations

from seahaven.eden.outcome import EDEN_MAX_TOKENS, parse_command
from ..backends import Backend, resolve
from ..backends.base import COHORT_TEMPERATURE

PROBE_MESSAGES = [
    {"role": "system",
     "content": "You are playing a text adventure. Reply with exactly one command."},
    {"role": "user",
     "content": "A bare hall with a swept floor. The store lies north and the "
                "yard east.\n\nHealth: 67/100."},
]


def main(args) -> int:
    spec = resolve(args.endpoint, model=args.model, key_env=args.key_env)
    be = Backend(spec)
    print(f"PROBE — {spec.model} at {spec.base_url}")
    print(f"  catalogued provider: {spec.catalogued}"
          + ("" if spec.catalogued else "   (record-and-pin applies)"))

    resolved = be.resolve_model()
    print(f"  model string resolves: {resolved}")

    print(f"\n  serving ONE episode-shaped turn at the REAL max_tokens "
          f"({EDEN_MAX_TOKENS}).")
    print("  Probing at a small cap is TRAP 4.1 and produces false failures.")

    accepted, deviation = COHORT_TEMPERATURE, None
    try:
        reply = be.ep.chat(PROBE_MESSAGES, max_tokens=EDEN_MAX_TOKENS,
                           temperature=COHORT_TEMPERATURE, seed=1)
    except Exception as e:
        if "temperature" not in str(e).lower():
            print(f"\n  FAILED at the cohort temperature: {str(e)[:200]}")
            return 1
        print(f"\n  ** REJECTS the cohort temperature {COHORT_TEMPERATURE} **")
        print(f"     {str(e)[:160]}")
        reply = be.ep.chat(PROBE_MESSAGES, max_tokens=EDEN_MAX_TOKENS,
                           temperature=1.0, seed=1)
        accepted, deviation = 1.0, (
            f"endpoint rejects temperature {COHORT_TEMPERATURE}; served at 1.0. "
            "This is a MODEL CONSTRAINT, not a chosen knob — but it is a "
            "deviation from every published cell and must be pinned and recorded "
            "in every cell's meta.")

    parsed, failed = parse_command(reply)
    u = dict(be.usage_total)
    print(f"\n  temperature accepted: {accepted}"
          + ("   <-- DEVIATION" if deviation else "   (matches the cohort)"))
    print(f"  reply                {reply[:70]!r}")
    print(f"  parses to            {parsed!r}   parse_failed={failed}")
    print(f"  usage fields         {sorted(u)}")
    print(f"  reasoning tokens     {u.get('reasoning_tokens', 0)}")
    print(f"  cached tokens        {u.get('cached_tokens', 0)}")

    if failed:
        print("\n  ** THE REPLY DOES NOT PARSE. ** That is a RESULT about this")
        print("  model under this interface, reported as a parse-failure rate.")
        print("  The parser and vocabulary are frozen and are NOT patched with")
        print("  model-specific handling.")
    if deviation:
        print(f"\n  PIN THIS: {deviation}")
    return 0

"""Shared machinery for the eden round modules.

**Everything here is NEW and unhashed.** The round modules' pins hash
`ARTIFACTS` (the measurement modules) and the world locks — never their own
source — so their bodies may delegate here without moving a single pin. Nothing
in `seahaven/eden/_shared/` may be added to any round's `ARTIFACTS`.

`tests/test_pin_invariance.py` is the guard: 11 live hashes, 6 retired recompute
functions, and 7 frozen-artifact digests, all captured before this package
existed.
"""

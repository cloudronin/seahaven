"""Compiled worlds, shipped as package data.

**This file exists for packaging, not for imports.** The round pins hash
`worlds/world_eden_*/BUILD.lock.json` at paths relative to the repo root, and a
round module resolves that root as `Path(__file__).resolve().parents[2]`. From an
installed wheel that is `site-packages/`, so the locks must land at
`site-packages/worlds/...` or every pin raises `FileNotFoundError` and `vworld
verify` cannot run at all.

Making this a package puts them there without touching a single hashed file or
any root-resolution code.

The `.z8` binaries (9.6 MB) ship too: `vworld worlds` and any serving verb needs
them, and splitting the locks from the worlds they describe would let the two
drift.
"""

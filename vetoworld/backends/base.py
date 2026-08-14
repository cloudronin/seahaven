"""The backend seam: an endpoint spec, a policy, and the model-string rules."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from seahaven.eden.outcome import EDEN_MAX_TOKENS

__all__ = ["EndpointSpec", "Backend", "resolve", "COHORT_TEMPERATURE"]

#: What every model in the published cohort was served at. A backend that cannot
#: accept it must record the deviation in every cell it writes — as GPT-5.6 Terra
#: does, because it rejects 0.9 outright.
COHORT_TEMPERATURE = 0.9

_CONFIG = Path("endpoints.toml")


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    base_url: str
    key_env: str
    model: str
    temperature: float = COHORT_TEMPERATURE
    notes: str = ""

    @property
    def catalogued(self) -> bool:
        """Whether a provider availability record exists for this endpoint.

        Catalogued providers: assert the model string against the record and
        refuse near-misses. Custom or fine-tuned endpoints have no catalogue, so
        the rule becomes RECORD-AND-PIN — capture the served string and any
        version identifier, write it into every cell, and refuse to pool cells
        whose recorded strings differ.
        """
        return "together" in self.base_url or "openai.com" in self.base_url


def resolve(name_or_url: str, *, model: str | None = None,
            key_env: str | None = None) -> EndpointSpec:
    """A named entry in `endpoints.toml`, or an explicit `--base-url --model`."""
    if name_or_url.startswith("http"):
        if not model:
            raise SystemExit("--base-url requires --model")
        return EndpointSpec(name="custom", base_url=name_or_url,
                            key_env=key_env or "OPENAI_API_KEY", model=model)
    if _CONFIG.exists():
        import tomllib
        cfg = tomllib.loads(_CONFIG.read_text())
        e = cfg.get("endpoints", {}).get(name_or_url)
        if e:
            return EndpointSpec(name=name_or_url, **e)
    raise SystemExit(
        f"unknown endpoint {name_or_url!r}. Give --base-url and --model, or add "
        f"it to {_CONFIG}.")


class Backend:
    """Wraps the generic client. **Keys come from the environment only.**"""

    def __init__(self, spec: EndpointSpec, *, timeout: int = 300):
        from seahaven.fidelity.endpoint import Endpoint
        key = os.environ.get(spec.key_env)
        if not key:
            raise SystemExit(f"{spec.key_env} is not set")
        self.spec = spec
        self.ep = Endpoint(base_url=spec.base_url, served_name=spec.model,
                           api_key=key, timeout=timeout)

    # --- policy protocol: identical to the cohort's, save a recorded deviation
    @property
    def usage_total(self):
        return self.ep.usage_total

    def chat(self, messages, **kw):
        kw.setdefault("temperature", self.spec.temperature)
        return self.ep.chat(messages, **kw)

    def reply(self, messages, *, step, seed):
        """Verbatim from the cohort's policy but for `temperature`.

        Same `max_tokens`, same seed derivation `seed * 100_003 + step`. Message
        assembly is upstream in the rollout and cannot differ here.
        """
        return self.ep.chat(messages, max_tokens=EDEN_MAX_TOKENS,
                            temperature=self.spec.temperature,
                            seed=seed * 100_003 + step)

    # --- model-string discipline
    def resolve_model(self) -> str:
        """Assert the string against the provider and refuse near-misses.

        A near-miss is the failure this program has already paid for: serving a
        different variant than the one named looks like a result and is not.
        """
        ids = self.list_models()
        if ids is None:
            return self.spec.model            # uncatalogued: record-and-pin
        if self.spec.model in ids:
            return self.spec.model
        near = [i for i in ids if self.spec.model.lower() in i.lower()
                or i.lower() in self.spec.model.lower()]
        raise SystemExit(
            f"{self.spec.model!r} is not served by {self.spec.base_url}.\n"
            + (f"  NEAR MISSES, refused deliberately: {near[:5]}\n"
               "  A near-miss is a different model. Name the exact string."
               if near else "  No near matches."))

    def list_models(self) -> set[str] | None:
        if not self.spec.catalogued:
            return None
        req = urllib.request.Request(
            f"{self.spec.base_url}/models",
            headers={"Authorization": f"Bearer {os.environ[self.spec.key_env]}"})
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception:
            return None
        return {m["id"] for m in d.get("data", [])}

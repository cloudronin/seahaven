"""Talk to any OpenAI-compatible chat endpoint.

Deliberately thin and dependency-light: `urllib` rather than an SDK, so the tool
runs against vLLM, Ollama, TGI, llama.cpp or a hosted API without dragging in a
provider stack. The same shape raidex uses — `--model http://localhost:8000/v1`
plus a served name — so a score computed locally is comparable to one computed
anywhere else.

**Chat-template hazard, carried over from Phase A.** Base checkpoints ship chat
templates they were never trained to follow, and chat-formatting one drops its
usable output rate from 0.88 to 0.06 (TRAP 4.2). An OpenAI-compatible endpoint
applies its own template server-side, so this client cannot detect or prevent
that. A base model served through `/v1/chat/completions` will score badly for
formatting reasons rather than fidelity reasons, and the result JSON records the
served name so that confound is at least visible after the fact.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Endpoint:
    base_url: str
    served_name: str
    api_key: str | None = None
    timeout: float = 120.0
    max_retries: int = 3

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url.rstrip("/") + path
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
                # Transient endpoint hiccups are common under load. Back off,
                # but never swallow the final failure — a silently empty
                # generation would be scored as an omission and bias the result.
                time.sleep(2 ** attempt)
        raise RuntimeError(f"endpoint failed after {self.max_retries} attempts: {last}")

    def chat(self, messages: list[dict], *, max_tokens: int = 128,
             temperature: float = 0.0, seed: int | None = None,
             stop: list[str] | None = None) -> str:
        payload = {
            "model": self.served_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Reasoning models spend the whole budget thinking and return empty
            # content. Servers that understand this disable it; servers that do
            # not ignore an unknown key, so it is safe to always send.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = stop
        out = self._post("/chat/completions", payload)
        try:
            msg = out["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"unexpected response shape: {out}") from e

        content = (msg.get("content") or "").strip()
        if content:
            return content

        # Empty content must never be returned as an empty narrative. It would be
        # scored as omitting everything, which makes a reasoning model look
        # maximally dishonest for a serving reason (cf. TRAP 4.1, where Qwen3's
        # default thinking mode scored 0/3 parseable until it was disabled).
        if (msg.get("reasoning") or msg.get("reasoning_content") or "").strip():
            raise RuntimeError(
                "endpoint returned reasoning but no content — the model is in "
                "thinking mode and spent the token budget on it.\n"
                "  This server ignored `chat_template_kwargs.enable_thinking`. "
                "Fixes: serve with thinking disabled, raise --max-tokens well "
                "above the reasoning length, or use a non-reasoning checkpoint.\n"
                "  Refusing rather than scoring an empty narrative, which would "
                "read as omitting everything.")
        raise RuntimeError(
            f"endpoint returned empty content and no reasoning: {out}")

    def probe(self) -> dict:
        """Cheap reachability and sanity check, run before any scoring work.

        Phase A lost a 25-minute job to a checkpoint that loaded and then
        produced nothing. One short call up front is worth that.
        """
        t0 = time.time()
        txt = self.chat([{"role": "user", "content": "Reply with the single word: ready"}],
                        max_tokens=8)
        return {"reachable": True, "latency_s": round(time.time() - t0, 2),
                "sample": txt.strip()[:60], "empty": not txt.strip()}

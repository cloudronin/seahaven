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
from dataclasses import dataclass, field


@dataclass
class Endpoint:
    base_url: str
    served_name: str
    api_key: str | None = None
    timeout: float = 120.0
    max_retries: int = 3
    #: Learned once per endpoint, then reused. None = not yet determined.
    _supports_template_kwargs: bool | None = field(default=None, repr=False)
    _supports_system: bool | None = field(default=None, repr=False)

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
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:
                    # The server's body says WHY. Discarding it made a 400 in a
                    # real sweep undiagnosable: three models lost repeats and the
                    # only evidence was the status line.
                    try:
                        body = e.read().decode("utf-8", "replace")[:400]
                    except Exception:
                        body = "<body unreadable>"
                    raise RuntimeError(f"HTTP {e.code}: {e.reason} — {body}") from e
                last = e
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
        """Send a turn, negotiating the two things chat templates disagree about.

        **Both defaults are wrong for some model, so neither can be assumed.**

        - `chat_template_kwargs={"enable_thinking": False}` is needed for
          reasoning models, which otherwise spend the whole budget thinking and
          return empty content. But a template that does not declare that
          variable **rejects the entire request with HTTP 400** — Mistral-7B and
          Gemma-2 both did, after this key was added to fix a Qwen problem and
          tested only on Qwen.
        - A `system` role is rejected outright by Gemma-2's template.

        So: try the full form once, fall back on 400, and remember what worked.
        Renegotiating on every call would triple the request count.
        """
        variants = []
        if self._supports_template_kwargs is not False:
            variants.append("kwargs")          # richest form, needed by reasoning models
        if self._supports_system is not False:
            variants.append("plain")           # no template kwargs, system role kept
        variants.append("merged")              # system folded into the first user turn

        last_err = None
        for kind in variants:
            msgs = messages
            payload = {"model": self.served_name, "max_tokens": max_tokens,
                       "temperature": temperature}
            if kind == "kwargs":
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            if kind == "merged":
                # Fold any system turn into the first user turn rather than
                # dropping it: dropping silently removes framing that carries
                # real effects.
                sys_txt = "\n\n".join(m["content"] for m in messages
                                      if m["role"] == "system")
                rest = [m for m in messages if m["role"] != "system"]
                if sys_txt and rest:
                    rest = [{"role": rest[0]["role"],
                             "content": sys_txt + "\n\n" + rest[0]["content"]}] + rest[1:]
                msgs = rest or messages
            payload["messages"] = msgs
            if seed is not None:
                payload["seed"] = seed
            if stop:
                payload["stop"] = stop
            try:
                out = self._post("/chat/completions", payload)
                if kind == "kwargs":
                    self._supports_template_kwargs = True
                elif kind == "plain":
                    self._supports_template_kwargs = False
                else:
                    self._supports_template_kwargs = False
                    self._supports_system = False
                break
            except RuntimeError as e:
                last_err = e
                continue
        else:
            raise RuntimeError(
                f"every request form was rejected by the endpoint: {last_err}")
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

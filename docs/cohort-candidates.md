# Cohort candidates — Check 2 of the feasibility gate

Selection is by **coverage only**: family, size and base/instruct
status. Nothing about expected behaviour enters, because a cohort
composed against an expected outcome produces that outcome — the
flag study established that the hard way.

The flag study's pinned reserve and bench are excluded by
construction and stay untouched.

**On paper: 38 available of 42 enumerated.**

- families: **8** (need 5) — Falcon3, Gemma-2, Granite-3.1, Llama-3.1, Mistral, OLMo-2, Qwen2.5, Qwen3
- families with 3+ sizes: **4** (need 2) — Falcon3, OLMo-2, Qwen2.5, Qwen3
- base/instruct pairs: **16** (need 3)

Loop-test and determinism attrition are Check 3; these are
paper-availability figures only.

| family | size | kind | repo | status |
|---|---|---|---|---|
| Falcon3 | 1.0B | base | `tiiuae/Falcon3-1B-Base` | available |
| Falcon3 | 1.0B | instruct | `tiiuae/Falcon3-1B-Instruct` | available |
| Falcon3 | 3.0B | base | `tiiuae/Falcon3-3B-Base` | available |
| Falcon3 | 3.0B | instruct | `tiiuae/Falcon3-3B-Instruct` | available |
| Falcon3 | 7.0B | base | `tiiuae/Falcon3-7B-Base` | available |
| Falcon3 | 7.0B | instruct | `tiiuae/Falcon3-7B-Instruct` | available |
| Falcon3 | 10.0B | instruct | `tiiuae/Falcon3-10B-Instruct` | available |
| Gemma-2 | 2.0B | instruct | `google/gemma-2-2b-it` | available (gated, terms accepted) |
| Gemma-2 | 9.0B | instruct | `google/gemma-2-9b-it` | available (gated, terms accepted) |
| Granite-3.1 | 2.0B | instruct | `ibm-granite/granite-3.1-2b-instruct` | available |
| Granite-3.1 | 8.0B | instruct | `ibm-granite/granite-3.1-8b-instruct` | available |
| Llama-3.1 | 8.0B | base | `meta-llama/Meta-Llama-3.1-8B` | available (gated, terms accepted) |
| Llama-3.1 | 8.0B | instruct | `meta-llama/Llama-3.1-8B-Instruct` | available (gated, terms accepted) |
| Llama-3.2 | 1.0B | base | `meta-llama/Llama-3.2-1B` | BLOCKED (gated, terms not accepted) |
| Llama-3.2 | 1.0B | instruct | `meta-llama/Llama-3.2-1B-Instruct` | BLOCKED (gated, terms not accepted) |
| Llama-3.2 | 3.0B | base | `meta-llama/Llama-3.2-3B` | BLOCKED (gated, terms not accepted) |
| Llama-3.2 | 3.0B | instruct | `meta-llama/Llama-3.2-3B-Instruct` | BLOCKED (gated, terms not accepted) |
| Mistral | 7.0B | base | `mistralai/Mistral-7B-v0.3` | available |
| Mistral | 7.0B | instruct | `mistralai/Mistral-7B-Instruct-v0.3` | available |
| OLMo-2 | 1.0B | base | `allenai/OLMo-2-0425-1B` | available |
| OLMo-2 | 1.0B | instruct | `allenai/OLMo-2-0425-1B-Instruct` | available |
| OLMo-2 | 7.0B | base | `allenai/OLMo-2-1124-7B` | available |
| OLMo-2 | 7.0B | instruct | `allenai/OLMo-2-1124-7B-Instruct` | available |
| OLMo-2 | 13.0B | base | `allenai/OLMo-2-1124-13B` | available |
| OLMo-2 | 13.0B | instruct | `allenai/OLMo-2-1124-13B-Instruct` | available |
| Qwen2.5 | 0.5B | base | `Qwen/Qwen2.5-0.5B` | available |
| Qwen2.5 | 0.5B | instruct | `Qwen/Qwen2.5-0.5B-Instruct` | available |
| Qwen2.5 | 1.5B | base | `Qwen/Qwen2.5-1.5B` | available |
| Qwen2.5 | 1.5B | instruct | `Qwen/Qwen2.5-1.5B-Instruct` | available |
| Qwen2.5 | 3.0B | base | `Qwen/Qwen2.5-3B` | available |
| Qwen2.5 | 3.0B | instruct | `Qwen/Qwen2.5-3B-Instruct` | available |
| Qwen2.5 | 7.0B | base | `Qwen/Qwen2.5-7B` | available |
| Qwen2.5 | 7.0B | instruct | `Qwen/Qwen2.5-7B-Instruct` | available |
| Qwen2.5 | 14.0B | instruct | `Qwen/Qwen2.5-14B-Instruct` | available |
| Qwen3 | 0.6B | base | `Qwen/Qwen3-0.6B-Base` | available |
| Qwen3 | 0.6B | instruct | `Qwen/Qwen3-0.6B` | available |
| Qwen3 | 1.7B | base | `Qwen/Qwen3-1.7B-Base` | available |
| Qwen3 | 1.7B | instruct | `Qwen/Qwen3-1.7B` | available |
| Qwen3 | 4.0B | base | `Qwen/Qwen3-4B-Base` | available |
| Qwen3 | 4.0B | instruct | `Qwen/Qwen3-4B` | available |
| Qwen3 | 8.0B | base | `Qwen/Qwen3-8B-Base` | available |
| Qwen3 | 8.0B | instruct | `Qwen/Qwen3-8B` | available |

# Optimization ladder — GLM-5.3-Flash + DFlash2, 2× DGX Spark (GB10), SGLang TP=2

All numbers: warmed, temp 0, stream:false, 800 max_tokens, n=5 medians, stock clocks,
code-1/prose-1 prompts as in RESULTS.md. Keep rule: beats incumbent code median with the
19×21 gate + token-0 collapse scan passing.

## Round 1 (autonomous, night of 2026-08-27→28)
| rung | change | code | prose | verdict |
|---|---|---|---|---|
| first light | — | 27.6 | 20.7 | baseline |
| L2 | flashinfer autotune ON (+`--enable-metrics`) | **28.4** (26.9–29.8) | 20.6 | **KEPT** |
| L3 | + fp8 draft KV | 27.0 (23.9–28.4) | 20.7 | reverted — convert overhead exceeds bandwidth saved on a ~2 GB drafter cache |
| L4 | + fp8 target KV | — | — | **BOOT FAILED in 64 s**; failure logs lost to teardown (ladder now saves them); autopsy queued round 2 |
| L5 | ctx 131072, max-total 262144 | 28.0 (27.6–30.8) | **21.3** | reverted by keep-rule — **but note: 2× context for −1.4% code speed; worth keeping for serving. Operator's call. Confound: carried L3's fp8-draft-KV flag** |

Round-1 net: +3% code. The big lever (fp8 target KV) is blocked, not disproven.

## Round 2
| rung | change | code | prose | verdict |
|---|---|---|---|---|
| L1v2 | decode+v1 kernels stock tiles | — | — | BOOT FAILED, same 169,984 B smem error — `sparse_attention_fwd_kernel_v1` is in the verify path and must stay small |
| L1v3 | only `sparse_mla_fwd_decode_partial` stock | 28.5 (25.7–33.2) | 20.2 | kept (wash) — **finding: the GB10 small tile is ~free; tile geometry is not the bottleneck** |

Kernel tile map for sm_121 (measured): `qo_len` multi-token kernel → small tile REQUIRED;
`sparse_attention_fwd_kernel_v1` → small tile REQUIRED (verify path); `sparse_mla_fwd_decode_partial`
→ either (no measurable difference).

| D6 | speculative-num-draft-tokens 8→6 | **29.7** (27.2–31.0) | **21.5** (20.4–23.4) | **KEPT — new incumbent.** Shorter draft block wastes less verify on doomed tokens at our acceptance profile |

Cumulative: 27.6 → 29.7 code (+7.6%) since first light.
| L4v2 | fp8 target KV (tilelang DSA) | — | — | **CLOSED: architecturally unsupported** — SGLang raises `tilelang DSA ... on CUDA requires a bfloat16 KV cache` at arg resolution. Round-1's 64s death explained |

| L4v3 | trtllm DSA backends + fp8 KV | — | — | **DIED: `TllmGenFmhaRunner: Unsupported arch`** — trtllm FMHA does not support sm_121 |

**fp8 target KV verdict on GB10 + SGLang + GLM-5.3: impossible on this stack today.**
tilelang forbids it on CUDA by upstream policy; trtllm kernels don't build for the chip.
(vLLM reached fp8 KV on GB10 only via hand-patched CTA tile caps — see tonyd2wild's recipe.)

| D4 | draft tokens 4 | 28.3 (28.2–28.7) | **23.3** (23.3–23.3) | reverted on code — but best prose of the campaign (+8% vs D6): code peaks at D=6, prose at D=4; D=5 queued |
| V4/V5 (first attempt) | upstream aa8c950a3 refresh + novel fat tile | — | — | both died pre-kernel on a partial-overlay skew (`hc_attn_to_mlp` — model file and communicator_mhc.py are a coupled pair); rebuilt as V4b/V5b with both files |

| V4b | upstream aa8c950a3 (official mHC capture fix + kpool changes) + our tiles | 29.5 (29.5–29.6) | **22.5** | **KEPT as incumbent** — code statistically tied with D6 (29.7, whose spread contains V4b), prose +4.7%, and it replaces our hand-guard with the official fix. Tie broken on provenance |
| V5b | novel fat tile 64/1/256 | — | — | smem 104,448 B > 101,376 — **3,072 bytes over.** Single-stage halved the request exactly as modeled |

| D5 | draft tokens 5 | 29.6 (29.6–29.6) | **23.3** (23.0–23.3) | **best combined config** — D6's code with D4's prose; takes incumbency on tie-break |
| V5c | fat tile 64/1/128 | — | — | smem 103,424 B — still 2 KB over. **Fat-tile chapter closed with a complete map**: 64-wide needs ≥103.4 KB in any shape; GB10 ceiling 101.4; 32-wide fits and costs nothing |

**Decoupled drafter** (killing the TP=2 all-reduce tax on the 1B draft model): mapped, viable, NOT attempted —
requires a separate drafter-server topology (`--decoupled-spec-role verifier/drafter-rank` + bind/connect
endpoints + rank). This is the #1 next-frontier item; expected the largest structural gain.

| **FINAL** | **v4b image + D5 flags** | **29.4** (28.2–30.4) | **23.4** (20.7–25.4) | **SHIP CONFIG** — best combined; official upstream code |
| FA3 | draft attention fa3 | — | — | CLOSED: `FA3 requires SM>=80 and SM<=90` — sm_121 outside the window. Draft-attention map complete: flashinfer only |

| SERVE | FINAL + ctx 131072 / max-total 262144 | 29.3 (25.2–31.8) | 22.7 (21.9–23.5) | **PRODUCTION CONFIG** — 2× context window at statistical parity; Hermes cut over to this endpoint 2026-08-28 |

Campaign complete. Closeout data in RESULTS.md; morning package in MORNING.md.


## Concurrency curves (2026-08-28, agent-style prompts, 400 tok, aggregate tok/s)
| c | DFlash (max_req 8) | no-spec (max_req 12) |
|---|---|---|
| 1 | **37.3** | 14.5 |
| 2 | **49.5** | 27.1 |
| 4 | **51.7** (12.9/stream) | 36.8 |
| 8 | 47.3 | **55.1** |
| 12 | 48.8 | **55.0** |

**Correction:** our earlier "DFlash verify saturates at c1" was an artifact of
`max_running_requests=2` — with headroom, DFlash scales and dominates to ~c8.
~55 aggregate is the machine's bandwidth plateau either way. **Production config
changed to DFlash + max_running 8** (best latency at every c<8, ~94% of the fleet
ceiling at c12).

## fp8-KV tilelang CUDA port (2026-08-28) — WORKS ON GB10
The gate was plumbing, not kernels: a complete raw-fp8 sparse kernel (`sparse_mla_fwd_decode_partial_fp8`)
ships in-tree, HIP-gated in 3 plumbing sites (dispatch, pool layout, fused-quant write). Port = relax the
gate (CUDA + both-backends-tilelang + SM89+), route the raw 512 B/token layout on CUDA, key the raw write
on layout not platform. Kernel verified by execution on sm_121 first (one-hot exact 0.000000; negative
control fails at 91%; fp8 GEMMs lower and run).
| check | result |
|---|---|
| boot-gate negative control (mixed backends + fp8) | **died as required** (ValueError) |
| point-of-effect log line | present (stock code cannot print it) |
| 19×21 gate / token-0 collapse | PASS / none |
| decode (n=5 medians) | 29.2 code / 22.9 prose — **parity** with same-image bf16 (29.8) |
| temp-0 vs bf16, 5 prompts | **4/5 exact** |
| 32K-depth codeword probe | PASS |
| TTFT@16k warm | **6.6 s vs 7.9 s bf16 — 17% faster prefill** |
| open item | raw-layout accounting resolves max_running_requests to 1; retune attempt (max-total 393216 + mamba-ratio 3) held at 1 and shrank the pool — the constraint is the mamba slice, not the token budget. Documented as PR known-limitation; fp8 = interactive config, bf16-T8 = fleet config |
Kernel analysis + port: this rig (Fable agent), validated per the house protocol.

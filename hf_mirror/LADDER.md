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

## Multi-stream unlock (2026-08-28 pm) — the "1-stream fp8" open item is CLOSED, and it was never about fp8
The engine logs the whole story at boot:
`max_running_requests is capped to 2 by the mamba state cache (max_mamba_cache_size=10, 5 state slots per request)`.
Every DFlash config on this rig — bf16 T8 included, since it shipped identical memory args — was
silently capped at **2 concurrent spec streams**; the flat c2→c12 aggregate we called a "bandwidth
plateau" was queueing. The ~55 tok/s no-spec ceiling was the cap's shadow, not the machine's.

Fix (config FP8T8b, one coherent package): `--max-mamba-cache-size 40` (8 streams x 5 slots) +
`--mamba-ssm-dtype bfloat16` (halves state: 35 MB/slot vs 75) + `--mem-fraction-static 0.90`.
A first attempt (48 slots, fp32 ssm, 0.88) died at pool allocation — weights leave only ~2.3 GB of
fraction slack, so the ssm dtype lever is what makes 40 slots fit. KV pool grew to 84,288 tokens as
a side effect; 10.8 GB runtime headroom kept deliberately (GB10 OOM can wedge the node).

| c | FP8T8b (fp8-KV, DFlash, 40 slots) | old T8 (bf16, capped@2) | no-spec T12 |
|---|---|---|---|
| 1 | 36.9 | 37.3 | 14.5 |
| 2 | 48.2 | 49.5 | 27.1 |
| 4 | 46.9 | 51.7 | 36.8 |
| 8 | **80.3** (10.0/stream, 8/8) | 47.3 | 55.1 |
| 12 | **85.4** (7.1/stream, 12/12) | 48.8 | 55.0 |

+70% at c8, +75% at c12 over the capped curve; +55% over the no-spec "plateau". Decode batches
observed at 5-8 running requests — first true multi-stream DFlash on this rig. Trade-off, measured:
single-stream code median 27.4 vs 29.2 on the 10-slot fp8 config (~6%, consistent with bf16 ssm
states shaving accept length; deep-batch accept len ~3.1 vs ~4.2 single). Correctness: 19x21 gate
PASS, no token-0 collapse. **Production is now FP8T8b** — fp8-KV prefill wins AND the concurrency
curve, one config. (c-sweep prompts: 12 distinct short code/infra prompts, 400 max_tokens, temp 0,
stream:false, warmed; clocks stock.)

## Vision unlock (2026-08-28 pm) — GLM-5.3-Flash is MULTIMODAL, and it works on this stack
Correction of our own record: GLM-5.3-Flash has a full 24-layer vision tower with image AND
video tokens (upstream config: Glm5NextForConditionalGeneration). The LibertAIDAI NVFP4 quant
ships all 347 model.visual.* tensors, and the SGLang #36507 branch implements the vision path.
Our deployments simply never passed --enable-multimodal.

Config FP8T8V = FP8T8b + `--enable-multimodal`. Results:
- vision gate: PASS (64x64 solid-red data-URL probe answered "Red", temp 0)
- single-stream: 29.3 code / 23.8 prose — no measurable vision-tower tax (matches best fp8)
- c-sweep: c1 34.5 / c2 51.1 / c4 44.5 / c8 78.7 (8/8) / c12 79.5 — concurrency intact
  (c12 delta vs FP8T8b's 85.4 is single-run noise territory; c4 dip reproduces in both)
**Production is now FP8T8V: fp8-KV + 8 concurrent streams + image input, one config.**
Not yet measured: vision quality beyond the smoke probe, video input, vision+DFlash accept
interaction, vision under concurrency. Treat image support as verified-working, not benchmarked.

## Pool expansion (2026-08-28 pm) — FP8T8X, production
FP8T8V + mem-fraction 0.90->0.92, single variable. KV pool 84,288 -> 244,032 tokens (2.9x;
the 262,144 ask minus page rounding), both target-fp8 and draft-bf16 pools. 8.9GB runtime
headroom kept. Gate PASS; c8 78.1 / c12 83.5 (curve unchanged); single-stream 27.0 code —
within the day's 27.0-29.3 noise band for this config family. Production = start-FP8T8X.sh:
fp8-KV + 8 streams + vision + 244k-token pool. Context-length raise beyond 131k = next rung
(needs rope/prefill verification, not just pool).

## Analyst pass (2026-08-28 pm) — c4 dip explained, D=7 rejected, one retraction
- The "c4 dip" is a HARNESS ARTIFACT, not an engine effect: engine decode is monotone in batch
  (bs1 ~37-41 -> bs8 ~97-100 tok/s from decode-batch logs; cuda graph active at bs=4, no batch
  split, mamba usage 0.30). The sweep harness takes wall=max over c different prompts and one
  low-acceptance straggler (p2/p3 in the prompt list) sets the wall; c8 amortizes the same tail
  over 2x the tokens (c4 1600 tok in 37.0s vs c8 3200 in 41.0s — impossible unless a shared
  straggler). Harness now prints per-stream walls + straggler id. Low-c aggregates in earlier
  sweeps understate the engine; c8/c12 figures stand (~80-96% homogeneous-model fit).
- RETRACTION: "D=5, max 6, realize ~5.9" was wrong — the arg counts the bonus token, ceiling
  is 5, measured accept 3.4-4.4. Their k=7 = our D=8. Corrected in RESULTS.md.
- D=7 experiment REJECTED by arithmetic before spending a boot: predicted +1.4% structured
  (ceiling +8% at perfect acceptance), -9% prose. The real gap is verify cost per token (~56%
  cheaper on the EXL3 stack at identical step rates) -> next frontier is the decoupled drafter
  attacking the ~40 ms fixed step floor.

## Straggler probe result (2026-08-28 pm) — mechanism REVISED by its own negative control
Per-prompt c1 probe: all 8 harness prompts land 10.3-14.0s (within +-20%) — the "slow prompt"
hypothesis is refuted (the predicted regex straggler was fastest). Revised mechanism, consistent
with all walls including the EXL3 lane's per-stream data (c4 min 7.9s ~= c1, max 18.7s):
**admission serialization** — chunked prefills enter one at a time, so wall = last-admitted
stream's start delay + decode; at c4 the fixed ramp amortizes over half the tokens of c8, which
reads as a "dip". Unchanged conclusions: the engine's decode is monotone in batch size, and
low-concurrency wall-clock aggregates understate steady-state engine throughput on BOTH stacks.
Structured re-measure same session: 48.8-49.7 tok/s (x3) vs 43.3 earlier — run-to-run band.

## Night queues 2+3 (2026-08-28 evening) — four rungs, one discovery, one blocker
| rung | change | verdict |
|---|---|---|
| N1 | num-continuous-decode-steps 3 | NULL — exact baseline; scheduler loop is not the 40ms floor |
| N2 | torch.compile | BOOT-FAIL — cuda-graph capture OOM at fraction 0.92; retry would need capped graph set + lower fraction, plus a compile tax on every boot. Not production-shaped |
| N3 | context 262k + fraction 0.93 | Speed-neutral (28.7 code, 80.2 c12), pool fully funded at 262,144 — BUT the 64k depth probe KILLED the worker (silent rank1 death mid-prefill, no traceback, docker state incoherent). Same class as the vLLM-route long-prefill worker-kill; falsifies "fp8 KV unaffected". Threshold between 32k (passes) and 62k (kills). >32k prompts UNVERIFIED on all configs until bisected. NOT promoted |
| N4 | enable-mixed-chunk | NULL on this protocol — short-prompt sweep cannot exercise it; needs a prefill-interference test before final verdict |
Conclusion of the verify-cost hunt: both cheap levers dead; the code-speed gap vs EXL3 is
quant-level. Next real levers: decoupled drafter (architecture), and the 64k crash bisect
(reliability before capacity). Production remains FP8T8X.

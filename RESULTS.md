# Measured results — GLM-5.3-Flash (NVFP4) + DFlash2, 2× DGX Spark (GB10), SGLang TP=2

First known deployment of this pairing on any hardware outside inco.ai (drafter repo
showed 0 downloads at deploy time; enabling PR #36708 merged ~7 h before first boot).

## Serving config (first-light envelope)
`mem-fraction-static 0.88`, `max-total-tokens 131072` (configurator resolved 64,832),
`mamba-full-memory-ratio 2`, context 65,536, `max_running_requests` resolved to 1,
KV bf16, draft tokens 8 (drafter-native block), draft attention flashinfer,
DSA tilelang retuned for GB10 (see patches/). Clocks: stock (no cap), engine warmed 2×800 tok.

## Decode throughput — warmed, temp 0, stream:false, 800 max_tokens, n=5, medians
| prompt | DFlash2 ON | DFLASH off (same stack) | ratio |
|---|---|---|---|
| code-1 (threaded queue impl) | **27.6 tok/s** (27.6–27.8) | 14.7 | **1.88×** |
| prose-1 (measurement essay) | **20.7 tok/s** (20.7–20.7) | 14.7 | **1.41×** |

Instantaneous scheduler gauge peaked at 45.5 tok/s during code decode; reported here
only as a gauge peak — sustained medians above are the comparable numbers.

## Acceptance (scheduler decode-batch log, during active code decode)
accept len 3.65–5.62 (of 9 per step: 7 drafts + verify + bonus), accept rate 0.38–0.66.
Consistent with the Qwen3.8 GB10 DFlash2 deploy's ~5/8 code acceptance.

## Losslessness (G6)
5 fixed prompts, temp 0, DFlash2-on vs DFLASH-off, token-for-token:
**NOT claimed lossless.** Of 5 prompts, 3 were VOID (harness captured only `content`,
which the auto-detected reasoning parser left empty — harness flaw, not a result),
1 matched exactly, 1 diverged at a formatting token (`**Step 1: Square each integer**`
vs `## Step 1: Square each number`) — a near-tie flipped by on/off-path numerics.
Full 20-prompt matrix with reasoning captured is queued. Treat DFlash2-on outputs as
distribution-preserving, not bit-identical, on this stack until that lands.

## Reference comparison (other published GB10 GLM-5.3 numbers, different stack)
- tonyd2wild vLLM TP2 + MTP-4: 21.8 tok/s median (peak 22.7); TP4: 35.7

## Serving measurements (FINAL config: v4b image + D5 flags, 2026-08-28 ~04:00)
- **TTFT** (stream:true, first content/reasoning delta, warm): ~4k-token prompt **2.3 s** uncached / **0.74 s** radix-cached; ~16k **7.9 s** (≈2,000 tok/s prefill). ~64k: unmeasurable at the 65,536 context window. First-request-after-boot 4k read 39.6 s — cold-start JIT/caches; excluded as contaminated, reported for honesty.
- **Concurrency**: superseded twice — first by the c-sweep (the "saturates at one stream" read was a max_running=2 artifact), then by the multi-stream unlock (LADDER.md): the mamba state cache capped ALL DFlash configs at 2 concurrent streams. With `--max-mamba-cache-size 40 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.90` on the fp8-KV config: **c8 80.3 tok/s aggregate (8/8 concurrent), c12 85.4** vs 47-49 capped and 55 no-spec. Single-stream cost of the unlock: 27.4 vs 29.2 code median (~6%).

## G6 losslessness — FINAL verdict (20 prompts, temp 0, content+reasoning captured, on vs off)
**1/20 exactly identical; 19/20 diverge** somewhere in their (mostly long, reasoning-bearing)
outputs. Divergences are near-tie token flips that cascade in long generations — outputs
remain correct and comparable in quality (all gates passed), but **on this stack (sm_121,
DSA tilelang, NVFP4) DFlash2-on is NOT bit-identical to DFLASH-off at temp 0.** The
drafter card's "greedy output matches the target exactly" does not reproduce here; whether
the cause is verify-path numerics on this chip or quant interaction is unresolved. Users
needing bit-exact greedy reproducibility should serve DFLASH-off.

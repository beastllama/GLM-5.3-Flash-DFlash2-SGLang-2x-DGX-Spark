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

## Round 2 (in progress)
- **L1 per-shape tilelang tiles**: round-1 image shrank ALL DSA kernels to the GB10 tile;
  only the 8-token verify shape needed it. Reverting decode-path kernels to stock
  (block_I=64/num_stages=2/threads=256), verify-shape kernel stays small.
- L4 autopsy with log capture before teardown.
- Draft-token sweep 8/6/4.

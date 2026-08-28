# Morning package — overnight tuning campaign, 2026-08-27→28

## Final numbers (all: warmed, temp 0, stream:false, 800 tok, n=5 medians, stock clocks)
| | first light | FINAL (v4b image + D5 flags) | Δ |
|---|---|---|---|
| code decode | 27.6 | **29.4** | +6.5% |
| prose decode | 20.7 | **23.4** | +13.0% |
| c2 aggregate | — | 30.2 | verify saturates at c1 |
| TTFT 4k/16k | — | 2.3 s / 7.9 s | ≈2k tok/s prefill |

## The night's findings (each measured, in LADDER.md with spreads)
1. `SGLANG_HOST_IP` per-rank or multi-node GB10 hangs forever in shm_broadcast
2. DFLASH×hybrid-KDA memory law: per_req×(1+D) + ratio≈5 amplification
3. **Complete GB10 DSA smem tile map**: 64-wide tiles need ≥103.4 KB in any stage/thread shape vs 101.4 KB ceiling — fat tiles physically impossible; 32-wide fits and costs nothing
4. fp8 target KV closed BOTH routes (tilelang: CUDA-forbidden upstream; trtllm: no sm_121)
5. Draft-token curve: code peaks D=6, prose D=4, best combined **D=5**
6. Upstream refresh (aa8c950a3): official mHC capture fix adopted; model file + communicator_mhc.py are a coupled pair (partial overlay = TypeError)
7. FA3 draft attention: sm_121 outside FA3's SM80–90 window — flashinfer is the only draft backend on GB10
8. **G6: NOT bit-identical** — 1/20 exact at temp 0 with reasoning captured; quality preserved, exactness not (contradicts the drafter card on this stack)
9. Next frontier (mapped, unattempted): decoupled drafter server (`--decoupled-spec-role` + endpoints) to kill the TP=2 all-reduce tax on the 1B draft model

## Drafted X post (do not auto-post)
Overnight autonomous tuning on the GLM-5.3-Flash + DFlash2 SGLang deploy (2x DGX Spark):
27.6 -> 29.4 tok/s code (+6.5%), 20.7 -> 23.4 prose (+13%), and a complete GB10 kernel map:
fat DSA tiles are physically impossible (103.4KB > 101.4KB smem), fp8 KV is closed on both
backends, FA3 can't run, and DFlash2 is NOT bit-identical at temp 0 here (1/20 exact, quality
preserved). Every experiment - kept and reverted - with spreads: [repo link]. Honest numbers only.

## Drafted HF update (goes in both repo READMEs' results section via hf_sync)
Already synced automatically — RESULTS.md and LADDER.md carry the full campaign.

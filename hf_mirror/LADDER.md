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

In progress: D=4 draft-token test (last cheap experiment), then closeout: final G6 matrix + staged morning update.

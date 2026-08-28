---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags:
- dgx-spark
- gb10
- sm121
- sglang
- speculative-decoding
- dflash2
- nvfp4
- recipe
---

# GLM-5.3-Flash + DFlash2 on 2× NVIDIA DGX Spark (GB10) — SGLang TP=2 recipe

Hey — I'm one of the folks running DGX Sparks at home, and this community's recipes are
the only reason my cluster works at all. tonyd2wild's GB10 forensics, MiaAI-Lab's
dual-Spark configs, hasso5703's DFlash2 writeup, LibertAIDAI's quant card — I've leaned
on all of them, so here's mine back.

This is GLM-5.3-Flash with the incoai DFlash2 drafter on the **SGLang** path (the
PR [#36507](https://github.com/sgl-project/sglang/pull/36507) branch everyone will get
by default once it merges). Getting it to boot on GB10 took a night and four fixes
nobody had written down yet — they're all here with patches and probes, so your
bring-up should take an hour instead. If you hit something new, open an issue and
I'll dig in with you.

## Measured (warmed, temp 0, stream:false, 800 tok, n=5 medians, stock clocks)
| prompt | DFlash2 ON | no-spec same stack | speedup |
|---|---|---|---|
| code | **27.6 tok/s** | 14.7 | **1.88×** |
| prose | **20.7 tok/s** | 14.7 | **1.41×** |

Accept length 3.65–5.62 (of 9/step) during code decode. First-light envelope (ctx 65536);
overnight optimization ladder in progress — numbers will improve, watch the GitHub repo.

**Not claiming bit-identical greedy outputs yet** (1 near-tie token flip in a 5-prompt
temp-0 on/off comparison; full matrix queued).

## The four GB10 day-0 fixes (patches in `patches/`)
1. `SGLANG_HOST_IP` must be set per-rank or multi-node `shm_broadcast` hangs forever
2. DFLASH×hybrid-KDA memory law: mamba pool needs `per_req×(1+D)` + ~5× per-request amplification
3. DSA tilelang smem overflow at the 8-token verify shape (169,984 B > sm_121's 101,376 B) — retune to `block_I=32, num_stages=1, threads=128`
4. `residual=None` crash in the DFLASH capture adapter during CUDA graph capture — **fixed upstream in sglang #36755** (merged 2026-08-28, same idiom as our guard; images built from `aa8c950a3`+ carry the official fix)

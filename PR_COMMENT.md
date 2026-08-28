DRAFT comment for sgl-project/sglang PR #36507 — sign-off before posting.

---
Ran this branch (c4d5d45e5, incl. #36708) with DFLASH + incoai/GLM-5.3-Flash-DFlash2 against LibertAIDAI/GLM-5.3-Flash-NVFP4 on 2× DGX Spark (GB10, sm_121), TP=2. It works — 27.6 tok/s code decode at accept len ~5.6/9 — after four fixes, two of which look upstream-relevant:

1. **`_prepare_aux_hidden_state` crashes when `residual is None`** (CUDA graph capture): `glm5_next.py` does `hidden_states + residual` unguarded. Guarding with the same idiom as the llama EAGLE-3 aux capture path fixes it. Trace: `TypeError: unsupported operand type(s) for +: 'Tensor' and 'NoneType'` at decode-graph capture.
2. **DSA tilelang kernels overflow GB10 shared memory at verify shape**: with `num_tokens_per_req=8` the sparse-attention tile requests 169,984 B dynamic smem vs sm_121's 101,376 B ceiling (`Failed to set the allowed dynamic shared memory size to 169984`). Plain decode (1 tok/req) fits, so this only appears under DFLASH/MTP. Retuning to `block_I=32, num_stages=1, threads=128` (per LibertAIDAI's GB10 validation) resolves it. Consider arch-conditional tile selection.

Also of note for docs: on multi-node GB10, `SGLANG_HOST_IP` must be set per-rank or `shm_broadcast.wait_until_ready` hangs indefinitely; and with DFLASH on hybrid GDN/KDA models the mamba pool needs `per_req×(1+num_draft_tokens)` intermediate state plus `ratio≈5` per-request amplification — drafter-less memory configs will fail with `max_mamba_cache_size<=0`.

Also observed: greedy (temp-0) outputs between DFLASH-on and DFLASH-off boots diverge at near-tie tokens on this stack (1 of 2 content-bearing test prompts) — flagging in case verify-path numerics on sm_121 deserve a look.

Full recipe & patches: github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark

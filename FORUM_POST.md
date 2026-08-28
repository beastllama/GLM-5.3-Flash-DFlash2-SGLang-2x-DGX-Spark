STAGED — NVIDIA forum, DGX Spark/GB10 Projects category. Title:
"GLM-5.3-Flash + DFlash2 on SGLang — 2x DGX Spark, first SGLang-path recipe (recipe + 4 GB10 fixes + honest numbers)"

---
GLM-5.3-Flash (320B/18B MoE, LibertAIDAI NVFP4 quant) with the incoai DFlash2 block-diffusion
drafter, running on the upstream SGLang path (PR #36507 branch) across 2x DGX Spark, TP=2.
As far as we can tell this is the first published SGLang-route deployment of this pairing —
@tonyd615 got the vLLM route running the same night (his numbers are faster; credit where due,
and his recipes carried half our bring-up).

Numbers (warmed, temp 0, stream:false, 800 max_tokens, n=5 medians, stock clocks):
- 29.4 tok/s code / 23.4 prose decode with DFlash2 vs 14.7 no-spec same stack (1.88x / 1.41x)
- accept length ~5-6 of 6-token blocks on code; TTFT ~2.3s at 4K prompt, 7.9s at 16K
- production envelope: 131K context, bf16 KV (fp8 KV is currently impossible on SGLang GB10 —
  measured matrix in the repo and in sglang issue #36830)

Four GB10 day-0 fixes you will hit if you try this (all patched + probed in the repo):
1. SGLANG_HOST_IP must be set per-rank or multi-node shm_broadcast hangs forever
2. DFLASH x hybrid-KDA memory law: mamba pool needs per_req*(1+D) + ~5x per-request amplification
3. DSA tilelang smem overflow at the 8-token verify shape (169,984B > 101,376B) — retune to
   block_I=32/num_stages=1/threads=128; we mapped the whole tile space, fat tiles are physically
   impossible on GB10 (>=103.4KB in any shape)
4. residual=None crash in the DFLASH capture adapter — since fixed upstream (#36755)

Honesty section: DFlash2 on this stack is NOT bit-identical to spec-off at temp 0 (1/20 exact
in a 20-prompt matrix with reasoning captured; quality preserved, exactness not). And DFlash2
saturates the machine at c1 — concurrency adds ~nothing, so for multi-client serving measure
a no-spec boot too (c-sweep in the repo).

Recipe, patches, probes, full ladder (kept AND reverted experiments):
https://github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark
HF mirror: https://huggingface.co/randomllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark

---
ALSO STAGED — courteous reply for tonyd615's "43.4 tok/s PEAK [Checkpoint]" thread
(it opens "waiting on DFLASH2"):

DFlash2 has landed on both routes now — your vLLM recipe (46.9 C1) and an SGLang-path recipe
from our rig (29.4 C1 on bf16 KV; fp8 KV is closed on SGLang GB10 both backends, matrix in
sglang #36830). Your GB10 forensics saved us a night of the memory ladder — thanks for
publishing everything. SGLang recipe: [repo link]

---
ALSO STAGED — X reply to @MiaAI_lab's "might move to sglang... slow until dflash2":

sglang + dflash2 recipe for your 2x Sparks is up — 1.88x over no-spec, 4 GB10 fixes included,
honest caveats (bf16 KV only for now): github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark

---
STAGED — fp8 breakthrough reply for your own forum thread (post when ready):

Big update: fp8 KV cache now WORKS on SGLang GB10. This morning's post said it was
impossible (tilelang forbids it on CUDA; trtllm has no sm_121) — it turned out the
tilelang tree already ships a complete raw-fp8 sparse kernel, HIP-gated in three
plumbing sites. We ported the plumbing, verified the kernel by execution on sm_121
(one-hot exact; negative control fails as it must), and validated end-to-end:
decode parity with bf16, 4/5 temp-0 outputs exactly identical, 32k-depth recall
clean, and TTFT@16k 6.6s vs 7.9s bf16 (~17% faster prefill). Known item: the raw
layout currently resolves to single-stream — fp8 is our interactive config, bf16
the fleet config. Patch set (5 files) + validation protocol in the repo; upstream
PR to sglang in preparation. Details: sglang issue #36830.

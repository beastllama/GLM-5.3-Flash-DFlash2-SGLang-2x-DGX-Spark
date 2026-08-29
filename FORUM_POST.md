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

---
UPDATED STAGED fp8 forum reply — supersedes the block above (post this one instead):

Big update x2: fp8 KV cache now WORKS on SGLang GB10, and the "single-stream" limitation
died the same day. This morning's post said fp8 was impossible — the tilelang tree already
ships a complete raw-fp8 sparse kernel, HIP-gated in three plumbing sites; we ported the
plumbing and validated end-to-end (decode parity with bf16, 4/5 temp-0 exact, 32k recall
clean, TTFT@16k 6.6s vs 7.9s). Then the concurrency finding: the mamba state cache silently
caps EVERY DFlash config at 2 concurrent streams (10 slots, 5 per request — check your boot
log for "capped to 2 by the mamba state cache"). With --max-mamba-cache-size 40
--mamba-ssm-dtype bfloat16 --mem-fraction-static 0.90: c8 = 80.3 tok/s aggregate (8/8 truly
concurrent), c12 = 85.4 vs 48.8 capped — +75%, and +55% over our no-spec "bandwidth plateau,"
which turned out to be the cap's shadow. Cost: ~6% single-stream (27.4 vs 29.2). One config
now holds both the prefill and concurrency wins. Patches + full ladder in the repo; details
in sglang issue #36830.

---
FINAL STAGED forum reply v3 — supersedes both blocks above (post this one):

Three "known limitations" of the SGLang route died on this rig today, all config/plumbing:

1. fp8 KV cache on GB10 — this morning's post called it impossible. The tilelang tree
already ships a complete raw-fp8 sparse kernel, HIP-gated in three plumbing sites. We
ported the plumbing, validated on sm_121 (decode parity with bf16, 4/5 temp-0 exact, 32k
recall clean, TTFT@16k 6.6s vs 7.9s). Patch branch + executed pytest:
github.com/beastllama/sglang, branch fp8-kv-tilelang-cuda. Details: sglang #36830.

2. The concurrency "bandwidth plateau" — the mamba state cache silently caps EVERY DFlash
config at 2 concurrent streams (10 slots, 5/request; check your boot log for "capped to 2
by the mamba state cache"). With --max-mamba-cache-size 40 --mamba-ssm-dtype bfloat16
--mem-fraction-static 0.90: c8 = 78.7 tok/s aggregate (8/8 truly concurrent), c12 = 79.5
vs 48.8 capped. Filed as sglang #36889. Cost ~6% single-stream.

3. "No vision on the SGLang path" — GLM-5.3-Flash is multimodal (24-layer vision tower,
image AND video tokens), the NVFP4 quant ships all the visual tensors, and the #36507
branch implements the path. It was one flag: --enable-multimodal. Image input verified
working on the fp8 + 8-stream config with no measurable decode tax (29.3 tok/s code
single-stream). Honest scope: smoke-tested, not benchmarked — vision quality/video/
vision-under-concurrency still open.

One config now holds all three: fp8 KV + 8 concurrent streams + image input.
Full ladder (kept AND reverted experiments), boot scripts, probes:
github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark

---
FORUM REPLY v4 (2026-08-28 late) — supersedes v3, includes shootout. PASTE THIS ONE:

Big update since the original post. Three "known limitations" of the SGLang route died in one
day, then we ran the first same-rig comparison against the EXL3+vLLM recipe.

1. fp8 KV cache on GB10 works. The tilelang tree already ships a complete raw-fp8 sparse
kernel, HIP-gated in three plumbing sites. Ported, validated on sm_121 (decode parity with
bf16, 4/5 temp-0 exact, 32k recall clean, TTFT@16k 6.6s vs 7.9s). Upstream PR: sglang #36904.

2. The ~55 tok/s "bandwidth plateau" was a silent cap: the mamba state cache limits EVERY
DFlash config to 2 concurrent streams (boot log: "capped to 2 by the mamba state cache").
Fix: --max-mamba-cache-size 40 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.92.
Result: c8 = 78 tok/s aggregate (8/8 truly concurrent), c12 = 83.5 vs 48.8 capped. Filed as
sglang #36889. We also ran a correctness matrix under load (fixed arithmetic, temp 0): 44/44
right answers at c1/c4/c8.

3. GLM-5.3-Flash is multimodal and the SGLang path serves it: --enable-multimodal. Image
input verified working on the fp8 + 8-stream config, no measurable decode tax.

THE SHOOTOUT: we then ran MiaAI-Lab's EXL3+vLLM recipe (their repo @ bd7f55e, their image,
their defaults) on the SAME two Sparks, same prompts, same protocol. Honest split:
- EXL3+vLLM wins single-user code decisively: 51.9 vs 29.3 tok/s
- SGLang fp8 wins prose (29.2 vs 23.7) and fleet concurrency: 83.5 vs 53.3 at c12 (+57%);
  the EXL3 lane's aggregate degrades past c4.
Solo coder: run their lane. Agents/teams/concurrent: run this one. Both live on our rig now;
we re-measure as they ship.

One warning for long-context users on EITHER stack: we can reproduce a silent worker-node
death on long prefills (~62k tokens killed rank1 with no traceback on a 262k-context config;
32k passes). Threshold bisect in progress; treat >32k prompts as unverified until we post the
follow-up. Everything above, with the full ladder including failed experiments:
github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark

---
FORUM REPLY v5 (2026-08-28 night) — supersedes v4, adds round-2 shootout + root cause. PASTE THIS:

Big update since the original post. Three "known limitations" of the SGLang route died in one
day, we ran the first same-rig comparison against the EXL3+vLLM recipe (twice), and we found the
root cause of a long-prefill crash that affects both stacks' users.

1. fp8 KV cache on GB10 works. The tilelang tree already ships a complete raw-fp8 sparse kernel,
HIP-gated in three plumbing sites. Ported, validated on sm_121 (decode parity with bf16, 4/5
temp-0 exact, 32k recall clean, TTFT@16k 6.6s vs 7.9s). Upstream PR: sglang #36904.

2. The ~55 tok/s "bandwidth plateau" was a silent cap: the mamba state cache limits EVERY DFlash
config to 2 concurrent streams (boot log: "capped to 2 by the mamba state cache"). Fix:
--max-mamba-cache-size 40 --mamba-ssm-dtype bfloat16. Result: c8 = 78 tok/s aggregate (8/8 truly
concurrent), c12 = 83.5 vs 48.8 capped. Filed as sglang #36889. Correctness matrix under load
(fixed arithmetic, temp 0): 44/44 correct at c1/c4/c8.

3. GLM-5.3-Flash is multimodal and the SGLang path serves it: --enable-multimodal. Image input
verified working on the fp8 + 8-stream config, no measurable decode tax.

THE SHOOTOUT (ran it twice, second time after MiaAI-Lab shipped a concurrency update):
Same two Sparks, same prompts, same protocol, their repo/image/defaults.
- Single-user code: EXL3+vLLM wins, 45-52 vs 29 tok/s
- Prose: SGLang fp8 wins, 29.2 vs 25.0
- Fleet c12: SGLang fp8 wins, 83.5 vs 59.9 (their update improved this +12%, still trails)
- Long context: THEIR lane wins — recalled a needle at 54k tokens; ours kills the worker
Solo coder: run theirs. Agents/teams/concurrent: run ours. Both live on our rig; we re-measure
as they ship. Credit to MiaAI-Lab and brandonmusic.

LONG-PREFILL WARNING + ROOT CAUSE (matters for anyone running big prompts on GB10):
Prompts past ~40k tokens exhaust unified memory and can silently kill the worker rank — no
traceback, no OOM record, docker's own state record incoherent. Filed as sglang #36941 with the
mechanism: on models with index_kpool set (GLM-5.3-Flash has it), the DSA indexer materializes a
dense fp32 logits matrix sized q_rows x total_prompt_tokens, and the guard that exists for exactly
this (_should_chunk_mqa_logits) is defined but never called on that code path — the non-kpool
indexer does call it. Mitigation we are testing tonight: --chunked-prefill-size 2048.

Everything above with the full ladder including failed experiments and retractions:
github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark

# FINDINGS — GLM-5.3-Flash + DFlash2 on 2× DGX Spark

Research log, 2026-08-27. Every claim below was verified against the artifact named
(PR diff, Docker Hub API, HF API, forum thread), not against a label. Timestamps UTC.

## The one-line answer

**The missing engine support exists as of 18:23 UTC today.** SGLang PR #36708
("[DFLASH] Support GLM-5.3-Flash hidden-state capture") merged at 2026-08-27T18:23:30Z —
**into the feature branch `xinyuan/glm-5.3-flash-support` (PR #36507's branch), not into
main.** That is why searches of `sgl-project/sglang` main for `glm5_next` come up empty
and will keep coming up empty until #36507 itself merges. `refs/pull/36507/head`
(SHA `c4d5d45e506dcd978a65661a503eda1a272c39a4`) now contains the complete stack.

## Engine support map (all verified via GitHub API today)

### sglang PR #36507 — "GLM-5.3-Flash support" (OPEN, author JustinTong0323, branch `xinyuan/glm-5.3-flash-support`)
- https://github.com/sgl-project/sglang/pull/36507
- Head SHA: `c4d5d45e506dcd978a65661a503eda1a272c39a4` (this IS the #36708 merge commit —
  the stacked PR's merge advanced the branch head).
- Scope (file list read from the PR): `srt/models/glm5_next.py` (+1860),
  `srt/models/glm5_next_nextn.py` (+89, MTP head — an alternative spec-decode lane),
  `srt/configs/glm5_next.py` (+349), mHC kernels (`kernels/ops/layernorm/mhc.py` +214,
  `layers/communicator_mhc.py` +565), DSA kpool indexer (+1767/+1720/+871), KDA backend
  changes, memory-pool/hybrid-cache plumbing. This is the full hybrid
  KDA + deepseek-sparse-attention + mHC + 288-expert MoE implementation — nothing for us
  to write.
- The branch also contains the full DFLASH/DSPARK speculative infrastructure
  (verified via contents API at the head SHA): `srt/speculative/dflash_worker_v2.py`,
  `dflash_info.py`, `dflash_info_v2.py`, `dflash_utils.py`, `dflash_disaggregation.py`.

### sglang PR #36708 — DFLASH capture adapter (MERGED into #36507's branch, author jianc99)
- https://github.com/sgl-project/sglang/pull/36708
- Merged 2026-08-27T18:23:30Z, merge commit `c4d5d45e5`.
- Exactly two files: `srt/models/glm5_next.py` (+31/−5) and a unit test (+53).
- What it does: implements `set_dflash_layers_to_capture` on
  `Glm5NextForConditionalGeneration` and contracts the mHC inter-layer state from
  `hc_mult × hidden_size` down to `hidden_size` (mean over the hc groups) before handing
  it to the DFLASH drafter; disables TBO overlap while capturing.
- PR body: "Run in the GLM-5.3-Flash SGLang serving image based on the #36507 head" —
  i.e. the authors themselves test in the per-model image lineage we plan to use.
- Full diff captured; embedded in IMPLEMENTATION.md.

### The per-model Docker image (Docker Hub API)
- `lmsysorg/sglang:glm-5.3-flash-arm64` — pushed **2026-08-27T05:22:08Z**, arm64,
  14.1 GB compressed, digest `sha256:73f9294b78e38…`.
- Also `glm-5.3-flash` (amd64 alias pushed 05:22:40Z) and `glm-5.3-flash-amd64`.
- **Push time 05:22 predates the #36708 merge (18:23) by 13 h** → the published image has
  the full glm5_next model + DFLASH worker files but NOT the +31/−5 capture adapter.
  One-file patch required (or a later upstream rebuild — check digest before patching).
- Same per-model-image pattern as `qwen38-27b` (the proven DFlash2-on-Spark image) and
  `dev-*-qwen38-27b-dflash2` tags.

### vLLM (the fallback lane)
- vLLM main: `v1/spec_decode/dflash.py` exists, drafter model files are per-model
  (`qwen3_dflash2.py`, `laguna_dflash.py`) — **no GLM target or GLM drafter file**, and
  the sm_121 MLA path asserts `pe_dim == 64` while GLM-5.3-Flash is NoPE
  (`qk_rope_head_dim = 0`) → the "echoes the prompt back" bug named on the LibertAIDAI card.
- tonyd2wild fixed that (and 7 more day-0 bugs) in a patched image and ships a complete
  vLLM TP2 recipe with **MTP** (not DFlash):
  https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark
  Forum post ("43.4 tok/s PEAK [Checkpoint]", posted by tonyd615 2026-08-27):
  https://forums.developer.nvidia.com/t/glm-5-3-flash-on-2x-nvidia-dgx-spark-43-4-tok-s-peak-checkpoint/381429
  — vLLM TP2: 21.8 tok/s decode (peak 22.7), TTFT 0.29 s, 262K ctx, fp8-KV 507K pool,
  4-draft-token MTP, image `vllm/vllm-openai:glm53-flash-arm64-cu130` (their build),
  explicitly "waiting on DFLASH2".
- vLLM tracking PR named on the LibertAIDAI card: https://github.com/vllm-project/vllm/pull/53906

## Model artifacts

### Target quant: `LibertAIDAI/GLM-5.3-Flash-NVFP4`
- https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4
- 181 GiB; NVFP4 (E2M1, g16, weight-only) on routed-expert FFNs (37,152 tensors);
  BF16 for KDA layers, DSA+indexer layers, shared experts, routers, embeddings, lm_head,
  **and the MTP head** (kept — matters for the MTP fallback). 598.5 GiB → 181 GiB,
  round-trip cosine ≈ 0.99665.
- Card's own tested config: **2× GB10 TP=2 on `lmsysorg/sglang:glm-5.3-flash-arm64`**, flags:
  `--attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
  --moe-runner-backend flashinfer_cutlass --kv-cache-dtype bfloat16
  --disable-shared-experts-fusion --reasoning-parser glm45 --tool-call-parser glm47
  --mem-fraction-static 0.84 --context-length 65536 --max-running-requests 2`
- This resolves the "what SGLang did THEY use" tension: the per-model image, stock.

### Drafter: `incoai/GLM-5.3-Flash-DFlash2`
- https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
- HF API today: `gated: manual`, **downloads: 0** (nobody outside inco has pulled it yet),
  files: `config.json`, `model.safetensors` (single shard), LICENSE, README.
- 1B BF16 block-diffusion drafter, block size 8 (7 draft tokens/verify), two-tap dynamic
  convolutions + candidate path selector. Runtime in their card: 4× GB300 — **no GB10
  mention anywhere**.
- Card quickstart installs sglang from **`refs/pull/36708/head`** (== the stacked branch)
  and passes `--speculative-algorithm DFLASH --speculative-draft-model-path
  incoai/GLM-5.3-Flash-DFlash2 --speculative-draft-attention-backend fa4`.
  fa4 is a GB300-era backend; unverified on sm_121 (see Known Issues).
- **License: research and evaluation only.** No commercial/production use, no re-uploads
  or derivatives without written consent; commercial contact `contact@inco.ai`.
- Blog: https://inco.ai/blog/dflash2/ · Method repo: https://github.com/z-lab/dflash

## Proven neighbouring recipes mined for GB10 specifics

### DFlash2 dual-Spark TP2 (different target, Qwen3.8-27B) — the launch-shape donor
- https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-single-dual-dgx-spark-sglang-dflash2-fully-openai-compatible/380732
- `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` (both CX-7 twins — matches `ibv_devices` on our
  spark-1, verified today), `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, MTU 9000
  mandatory (NCCL hangs at warmup otherwise), rank-1 (worker) launched first,
  `--speculative-num-draft-tokens 8` optimal, **no draft-attention-backend flag** (default
  worked on GB10), `--mem-fraction-static 0.90` ceiling — **0.95 hard-reboots the GB10
  during "Capture target verify CUDA graph"**, `--stream-interval 1`, `--sleep-on-idle`.
  DFlash2 incompatible with YaRN on that build. Acceptance ~5/8 on code, ~3/8 on prose.
- Hardened single-Spark variant: https://github.com/hasso5703/dgx-spark-qwen38 —
  `--disable-flashinfer-autotune` for deterministic boots on unified memory; SGLang's
  accounting misses 25–40 GB of transient allocations (autotuner + graph capture) →
  Docker hard memory caps; prefix-cache path hit `CUBLAS_STATUS_INTERNAL_ERROR` on SM_121
  for one target; torch.compile Inductor int64 asserts on SM_121.
- Also: https://github.com/CharmiUwU/Qwen3.8-27B-DFlash2-DGX-Spark,
  https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark

### MiaAI_lab GLM-5.3 status
- https://x.com/MiaAI_lab/status/2092766465191469563 — 2× Spark GLM-5.3 via **vLLM**,
  "Speed is quite slow until dflash2 will be available … might move to sglang later."
  As of tonight no public follow-up shipping GLM+DFlash2; the drafter's download counter
  reads 0. **Nobody has publicly run this combination yet — on any hardware outside inco.**

## Cluster facts verified today (read-only, over ssh)
- spark-1 rails: `enp1s0f1np1` = 10.10.10.1/30, `enP2p1s0f1np1` = 10.10.11.1/30;
  spark-2: 10.10.10.2/30, 10.10.11.2/30. `ibv_devices`: rocep1s0f1, roceP2p1s0f0,
  roceP2p1s0f1.
- Current lane (rollback target): containers `qwen38-flash-next-head` / `-worker`,
  image `qwen38-flashnext-dspark:local` (30.2 GB), up 15 h, serving spark-1:8899.
- Disk: 2.7 TB free on both nodes — image (≈14 GB) + weights (181 GiB shared-split)
  + drafter (~2 GiB) fit trivially.

## Other references
- SGLang cookbook page (GB300-focused, confirms DFLASH in `--speculative-algorithm`
  choices and multi-node flag shapes): https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash
- zai-org/GLM-5.3-Flash: https://huggingface.co/zai-org/GLM-5.3-Flash (glm5_next,
  45 layers = 34 KDA + 11 DSA, 288 routed experts, 1M max positions, MTP head, FP8 main repo)
- llama.cpp support PR (not our lane): https://github.com/ggml-org/llama.cpp/pull/27754
- awesome-dgx-spark index: https://github.com/bidual/awesome-dgx-spark

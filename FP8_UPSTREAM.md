# Upstream PR draft — for operator review before opening (do not auto-post)

**Repo:** sgl-project/sglang · **Base:** the #36507 stack (or main once merged)
**Title:** [DSA] Enable tilelang fp8_e4m3 KV cache on CUDA (raw layout) — verified on sm_121

## Summary
The tilelang DSA backend already ships a complete raw-fp8 sparse attention kernel
(`sparse_mla_fwd_decode_partial_fp8` — generic TileLang, no HIP intrinsics). It is
unreachable on CUDA because three plumbing sites are HIP-gated, and CUDA is always
handed the scaled 528 B/token pool layout the kernel cannot parse. This PR routes the
raw 512 B/token layout on CUDA when both DSA backends are tilelang, relaxes the arg
gate accordingly (SM89+, hard error on mixed backends), and keys the raw fused-quant
write on layout rather than platform (strictly more correct on HIP as well).

## Files (5)
1. `srt/arg_groups/overrides.py` — `_check_tilelang_dsa_fp8_kv`: allow CUDA when
   prefill+decode backends are both tilelang and SM>=89; explicit ValueError on mixed
   backends; log line on enablement (serves as a point-of-effect check)
2. `srt/mem_cache/kv_cache_configurator.py` — `calculate_mla_kv_cache_dim`: CUDA +
   both-tilelang → raw dim → raw fp8 pool
3. `srt/mem_cache/memory_pool.py` — `_write_mla_kv_buffer`: raw-write branch keyed on
   `not dsa_kv_cache_store_fp8` instead of `_is_hip`; None-guard for `cache_k_rope`
   (NoPE models: GLM-5.3 has `qk_rope_head_dim=0`)
4. `kernels/ops/attention/dsa/tilelang_kernel.py` — dispatch to the fp8 partial kernel
   on CUDA under the same condition; (separately: sm_121 needs block_I=32/num_stages=1/
   threads=128 tiles — proposed as arch-conditional selection per our #36507 comment)
5. `srt/layers/attention/dsa/dsa_backend.py` — `supports_mha_one_shot=False` for
   raw-fp8 (moot on GB10; relevant SM90/100)

## Validation (2× DGX Spark GB10 sm_121, GLM-5.3-Flash NVFP4, TP=2, DFlash2 drafter bf16)
- Kernel: one-hot exact rel_err 0.000000; scrambled-index negative control fails 91%;
  spread-case 2.35% = analytic fp8-prob GEMM noise
- Boot-gate negative control (tilelang+flashmla_sparse+fp8) dies with the new error
- Decode n=5 medians: 29.2 code vs 29.8 same-image bf16 (parity); temp-0 4/5 exact;
  32k recall probe pass (re #36390); TTFT@16k warm 6.6 s vs 7.9 s bf16 (~17% faster)
- Known item: raw layout shifts pool accounting; document recommended
  mem-fraction/max-total retune for existing configs

## Credits
Kernel-path analysis and port by the Random Llama Software homelab rig (2× DGX Spark);
validation per the protocol in github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark.
Addresses #36830; relevant to #36390 (ROCm fp8 accuracy reports) and the sm_121
bring-up work in #36845/#36806.

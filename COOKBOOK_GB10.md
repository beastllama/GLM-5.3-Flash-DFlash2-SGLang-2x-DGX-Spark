# GLM-5.3-Flash on 2x DGX Spark (GB10 / sm_121)

GLM-5.3-Flash (NVFP4) with the `incoai/GLM-5.3-Flash-DFlash2` drafter, served with SGLang TP=2
across two DGX Sparks. Every number below was measured on this topology.

## Hardware and topology assumptions

- 2x DGX Spark (GB10, sm_121, aarch64), 121.7 GB unified memory per node.
- Rank 0 serves HTTP; rank 1 is the worker. **Launch the worker first.**
- ConnectX-7 back-to-back DAC. The GB10 QSFP port presents **two** virtual NICs, not one:
  give both an IP on separate /30s at MTU 9000, list both HCAs in `NCCL_IB_HCA`, and set
  `NCCL_IB_MERGE_NICS=1`. `NCCL_SOCKET_IFNAME` carries both rails; `GLOO_SOCKET_IFNAME` and
  `TP_SOCKET_IFNAME` steer the TCP control plane and take the first rail only. Confirm each
  variable inside the container (`docker exec <c> env`), not in the launching shell.
- `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` are required on this platform.
- Stop `earlyoom` on both nodes, or workers are SIGTERM'd with no clear error. Cap container
  memory (`--memory 115g`) so over-allocation fails as a container OOM: a wedged GB10 recovers
  only by power-cycling at the wall.

## Four day-0 fixes

1. **`SGLANG_HOST_IP` per rank.** Without it, multi-node bring-up hangs indefinitely in
   `shm_broadcast`. Set it to that rank's fabric address (`10.10.10.1` / `10.10.10.2`).
2. **DFLASH x hybrid-KDA memory law.** GLM-5.3-Flash has 34 KDA layers; with DFLASH each
   running request consumes `1 + speculative_num_draft_tokens` mamba state slots. Size the
   mamba pool from `max_running_requests x (1 + D)`, not from `max_running_requests`.
3. **DSA tilelang shared-memory retune.** Stock tiles request 169,984 B of dynamic smem at
   the multi-token verify shape; the GB10 ceiling is 101,376 B. Retune to
   `block_I=32, num_stages=1, threads=128`. The `qo_len` multi-token kernel and
   `sparse_attention_fwd_kernel_v1` (verify path) both **require** the 32-wide tile;
   `sparse_mla_fwd_decode_partial` runs either shape at no cost. 64-wide tiles need >= 103.4 KB
   in every stage/thread shape tried — impossible on this chip, not merely untuned.
4. **DFLASH capture adapter.** The `residual=None` crash in mHC hidden-state capture is fixed
   upstream by #36755 (on top of #36708). Use a branch at or past that commit; the model file
   and `communicator_mhc.py` are a coupled pair, so refresh both or neither.

Draft attention: FA3 rejects sm_121 (`FA3 requires SM>=80 and SM<=90`) — use the flashinfer
default, not the `fa4` flag from the drafter card's GB300 quickstart.

## The mamba state cache concurrency cap

Filed upstream as **#36889**. The scheduler derives `max_mamba_cache_size` from the static
memory fraction; when it resolves low, concurrency is capped with an info-level line only:

```
max_running_requests is capped to 2 by the mamba state cache (max_mamba_cache_size=10, 5 state slots per request).
```

Everything else still looks like an 8-stream server: `--max-running-requests 8` is accepted,
`/get_server_info` reports 8, requests are admitted and queue. The curve flattens at ~2 streams
and reads as a hardware bandwidth ceiling. Grep the boot log for this line before believing any
concurrency measurement on a hybrid-KDA model.

Fix as one package — `--max-mamba-cache-size 40` (8 streams x 5 slots),
`--mamba-ssm-dtype bfloat16` (halves per-slot cost: 35 MB vs 75), `--mem-fraction-static 0.90`.
The bf16 state dtype is what makes 40 slots fit; a 48-slot fp32 attempt at 0.88 died at pool
allocation. Aggregate tok/s over 12 code prompts, 400 max_tokens, temp 0, warmed, stock clocks:

| concurrency | capped (10 slots) | fixed (40 slots) |
|---|---|---|
| c1 | 37.3 | 36.9 |
| c8 | 47.3 | **80.3** (8/8 concurrent) |
| c12 | 48.8 | **85.4** |

Cost of the unlock: ~6% single-stream decode (27.4 vs 29.2 code median), consistent with bf16
states shortening accept length (deep-batch accept ~3.1 vs ~4.2 single-stream).

## fp8 KV cache status

`--kv-cache-dtype fp8_e4m3` requires **PR #36904** (open, stacked on #36507; addresses #36830).
Without it, tilelang DSA rejects fp8 KV on CUDA at argument resolution, and the trtllm DSA
backends do not build for sm_121. The PR routes the existing raw 512 B/token fp8 layout on
CUDA when both DSA backends are tilelang, relaxes the gate to SM89+, and keys the fused-quant
write on layout rather than platform. It is validated end-to-end only on sm_121.

Measured on sm_121 against the same image in bf16: decode parity (29.2 vs 29.8 code, n=5
medians), 4/5 temp-0 outputs exactly identical, 32K-depth recall probe pass, and **TTFT at a
16K prompt 6.6 s vs 7.9 s — ~17% faster prefill**. Keep the drafter KV in bf16
(`--speculative-draft-kv-cache-dtype bfloat16`): fp8 on a ~2 GB draft cache costs more in
conversion than it saves in bandwidth.

## Multimodal

GLM-5.3-Flash has a 24-layer vision tower with image and video tokens, the NVFP4 quant ships
all 347 `model.visual.*` tensors, and the #36507 branch implements the path. It is one flag,
`--enable-multimodal`, with no measurable decode tax (29.3 code / 23.8 prose single-stream,
matching the same config without it) and the concurrency curve intact. Image input is
verified by smoke probe; vision quality, video, and vision under concurrency are unbenchmarked.

## Known-good launch command

Run rank 1 first, then rank 0. Requires PR #36904 for the fp8 KV flag; substitute
`--kv-cache-dtype bfloat16` and drop the draft-KV flag to run without it.

```bash
#!/usr/bin/env bash
set -euo pipefail
RANK="${1:?usage: start-glm53-dflash.sh <0|1>}"

docker run -d --name "glm53-dflash-rank${RANK}" \
  --gpus all --network host --ipc host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband --memory 115g --memory-swap 115g \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 \
  -e SGLANG_HOST_IP=10.10.10.$((RANK+1)) \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e NCCL_IB_MERGE_NICS=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e TP_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
  <image> \
  python3 -m sglang.launch_server \
    --model-path LibertAIDAI/GLM-5.3-Flash-NVFP4 \
    --served-model-name glm-5.3-flash-dflash2 --trust-remote-code \
    --tp-size 2 --nnodes 2 --node-rank "$RANK" \
    --dist-init-addr 10.10.10.1:50051 \
    --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --moe-runner-backend flashinfer_cutlass \
    --kv-cache-dtype fp8_e4m3 --speculative-draft-kv-cache-dtype bfloat16 \
    --disable-shared-experts-fusion --reasoning-parser glm45 --tool-call-parser glm47 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path incoai/GLM-5.3-Flash-DFlash2 \
    --speculative-num-draft-tokens 5 \
    --mem-fraction-static 0.92 \
    --context-length 131072 --max-running-requests 8 --max-total-tokens 262144 \
    --mamba-full-memory-ratio 2 --max-mamba-cache-size 40 --mamba-ssm-dtype bfloat16 \
    --enable-multimodal --enable-metrics --stream-interval 1 --sleep-on-idle \
    --host 0.0.0.0 --port 8901
```

`--speculative-num-draft-tokens 5` is the best combined setting here: code throughput peaks at
D=6, prose at D=4, D=5 takes both within spread. D is workload-tunable — high-acceptance
structured workloads pin at the ceiling and may want a larger block. `--mem-fraction-static`
0.92 yields a 244,032-token KV pool with ~8.9 GB headroom; raise it one step at a time.

## Measurement discipline

- **Warm before measuring.** The first generation after boot pays a JIT/cache penalty large
  enough to swamp any effect under test (one 4K-prompt TTFT read 39.6 s cold against 2.3 s
  warm). Issue two real 800-token generations first.
- **Use `stream:false`**, read `usage.completion_tokens`, and time the request wall-clock.
- **Label every number with prompt, max_tokens, and clock state.** All three move results more
  than most effects under test: code and prose prompts differ by 20+ points of acceptance on
  the same engine. Report medians with spread over n>=5 and name the config.
- Parse `/metrics` by metric name, not by position. SGLang exposes `spec_accept*` as gauges —
  sample them during active decode, never at idle.
- Verify the served checkpoint from the engine's own argv (`/proc/1/cmdline`), not from
  `/v1/models`, which echoes whatever `--served-model-name` was configured.
- DFlash2 on this stack is **not** bit-identical to a DFLASH-off boot at temp 0 (1/20 prompts
  exact in a 20-prompt matrix with reasoning captured). Quality is preserved; exactness is not.
  Serve DFLASH-off if bit-exact greedy reproducibility is required.

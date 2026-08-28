# GLM-5.3-Flash (NVFP4) + DFlash2 on 2× NVIDIA DGX Spark — SGLang TP=2

**Status: bring-up recipe, not a victory lap.** Every component below is verified to
exist and is pinned by SHA/digest; the *combination* — GLM-5.3-Flash + DFlash2 on GB10 —
has, as of 2026-08-27, been run by nobody outside inco.ai on any hardware (the drafter
repo's download counter reads 0; the enabling PR merged today at 18:23 UTC). All
performance numbers for the combination are therefore **UNMEASURED** and marked so.
Reference numbers from neighbouring proven deploys are labeled with their source.

Provenance chain (see FINDINGS.md for how each was verified):

| Artifact | Pin | Role |
|---|---|---|
| `lmsysorg/sglang:glm-5.3-flash-arm64` | digest `sha256:73f9294b78e38…`, pushed 2026-08-27T05:22Z | serving image (glm5_next + DFLASH infra) |
| sglang `refs/pull/36507/head` | `c4d5d45e506dcd978a65661a503eda1a272c39a4` | branch the image tracks; head now includes PR #36708 |
| PR #36708 (merged into that branch 18:23Z) | +31/−5 on `models/glm5_next.py` | DFLASH capture adapter — **newer than the image; we patch it in** (IMPLEMENTATION.md) |
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | 181 GiB, 48-ish shards — record actual count at download | target weights; card's own test = this image, 2× GB10 TP=2 |
| `incoai/GLM-5.3-Flash-DFlash2` | 1B BF16, single shard, **gated + research-only license** | drafter, block size 8 |

## 0. Hardware and current state

- 2× DGX Spark (GB10, sm_121, aarch64), 121.7 GB unified memory, 2.7 TB free disk each.
- `spark-1` (rank 0, serves HTTP) / `spark-2` (rank 1) — CX-7 back-to-back DAC, dual-rail
  RoCE: `enp1s0f1np1` (10.10.10.1↔.2/30) + `enP2p1s0f1np1` (10.10.11.1↔.2/30), both MTU
  9000. HCA twins `rocep1s0f1,roceP2p1s0f1` (matches `ibv_devices`; note `roceP2p1s0f0`
  also exists — it is not ours).
- **Production today:** Qwen3.8-Flash-Next-NVFP4, containers `qwen38-flash-next-head` /
  `qwen38-flash-next-worker`, spark-1:8899. It stays up until cutover. Both lanes cannot
  run at once (each wants ~100 GB/node) — bring-up is a maintenance window with the Qwen
  lane stopped, rollback is restarting it (§9).
- GLM lane port: **8901** (deliberately ≠ 8899 so no client or probe can ever confuse
  lanes mid-migration). `--served-model-name glm-5.3-flash-dflash2`.

## 1. Human-required steps — blocking, do these first

1. **Request access to `incoai/GLM-5.3-Flash-DFlash2`** (gate is `manual` — a human at
   inco approves; wall-clock unknown, so file the request before anything else).
2. **Read the license before accepting.** It is *research-and-evaluation only*; no
   commercial/production use without written consent (`contact@inco.ai`). Whether this
   cluster's use qualifies is the operator's call, not this document's.
3. Place the HF token where the download step expects it (`~/.env.glm53` on spark-1,
   `HF_TOKEN=…`). Never paste it into a shell command or this repo.
4. Approve the maintenance window (Qwen lane down for the duration of §5–§8).

## 2. Preflight (both nodes, before the window)

```bash
# hotplug flag must be ABSENT; earlyoom must be STOPPED before launch
test ! -e /etc/nvidia/cx7-hotplug-enabled && echo hotplug-flag OK
sudo systemctl stop earlyoom && systemctl is-active earlyoom

# MTU 9000 end-to-end on BOTH rails (from spark-1):
ping -M do -s 8972 -c 3 10.10.10.2   # rail 1 — expect 0% loss
ping -M do -s 8972 -c 3 10.10.11.2   # rail 2 — expect 0% loss

# free page cache before the big load (GB10 unified-memory ritual):
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

## 3. Image

Follow IMPLEMENTATION.md §Steps 0–3: pull `lmsysorg/sglang:glm-5.3-flash-arm64`, check
whether upstream already rebuilt it past the 18:23Z merge, otherwise overlay the one
patched file → local tag `glm53-flash-dflash:c4d5d45e5`, `docker save | ssh spark-2
docker load`, **verify digests match on both nodes**, and run the patch-presence probe
(with its negative control) on both. Never build independently on the worker.

## 4. Weights (tmux on spark-1; poll, don't block)

```bash
set -a; source ~/.env.glm53; set +a       # HF_TOKEN for the gated drafter
huggingface-cli download LibertAIDAI/GLM-5.3-Flash-NVFP4        # ~181 GiB
huggingface-cli download incoai/GLM-5.3-Flash-DFlash2           # ~2 GiB, gated
```

Then rsync the HF cache to spark-2 (both nodes need both repos), record
`safetensor shard count and missing=0` for BOTH repos on BOTH nodes in the deploy log,
and from then on run containers with `HF_HUB_OFFLINE=1` (a gated repo re-check at boot
fails without the token; offline mode sidesteps it — but only after the cache is
complete). Check cache ownership on both nodes afterward; container writes as root
through the bind mount.

## 5. Launch

One script, `start-glm53-dflash.sh <node-rank>`; run **rank 1 on spark-2 FIRST**, then
rank 0 on spark-1 (worker-first; matches the proven dual-Spark DFlash2 deploy).

```bash
#!/usr/bin/env bash
# start-glm53-dflash.sh <0|1>
set -euo pipefail
RANK="${1:?usage: start-glm53-dflash.sh <0|1>}"

docker run -d --name "glm53-dflash-rank${RANK}" \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband \
  --memory 115g --memory-swap 115g \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_IB_MERGE_NICS=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e TP_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 \
  glm53-flash-dflash:c4d5d45e5 \
  python3 -m sglang.launch_server \
    --model-path LibertAIDAI/GLM-5.3-Flash-NVFP4 \
    --served-model-name glm-5.3-flash-dflash2 \
    --trust-remote-code \
    --tp-size 2 --nnodes 2 --node-rank "$RANK" \
    --dist-init-addr 10.10.10.1:50051 \
    --attention-backend dsa \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --moe-runner-backend flashinfer_cutlass \
    --kv-cache-dtype bfloat16 \
    --disable-shared-experts-fusion \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path incoai/GLM-5.3-Flash-DFlash2 \
    --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.80 \
    --context-length 65536 --max-running-requests 2 \
    --disable-flashinfer-autotune \
    --stream-interval 1 --sleep-on-idle \
    --host 0.0.0.0 --port 8901
```

Boot is minutes-long (weight load + CUDA graph capture). Poll
`curl -s http://10.0.x.x:8901/v1/models` from the Mac mini (over the tailnet — never
loopback) and tail `docker logs -f glm53-dflash-rank0`.

### Why each flag (and which are UNMEASURED)

| Flag | Why | Evidence |
|---|---|---|
| `--attention-backend dsa`, `--dsa-*-backend tilelang` | GLM-5.3's 11 deepseek-sparse-attention layers; tilelang is the backend the quant author ran on GB10 | LibertAIDAI card, tested 2× GB10 TP=2 |
| `--moe-runner-backend flashinfer_cutlass` | NVFP4 routed-expert path on Blackwell | same |
| `--kv-cache-dtype bfloat16` | the GB10-tested config; fp8 KV on this arch on sm_121 is unproven in SGLang (vLLM needed a CTA-tile cap for GB10's 101 KB smem) | same; fp8 KV = UNMEASURED, try later for KV headroom |
| `--disable-shared-experts-fusion`, `--reasoning-parser glm45`, `--tool-call-parser glm47` | quant author's tested config; parsers are GLM-4.x-lineage compatible | same |
| `--speculative-algorithm DFLASH` + drafter path | the point of this recipe | PR #36708; **combination UNMEASURED on GB10** |
| `--speculative-num-draft-tokens 8` | drafter block size is 8 (7 draft + 1); 8 measured optimal for the Qwen3.8 GB10 DFlash2 deploy | drafter card; forum 380732 |
| **no** `--speculative-draft-attention-backend fa4` | incoai's quickstart flag, written for GB300; fa4 on sm_121 unverified — the proven GB10 DFlash2 deploy used the default | risk #1 in IMPLEMENTATION.md |
| `--mem-fraction-static 0.80` | 0.84 was the card's value *without* a drafter; drafter adds ~2.3 GiB + capture buffers. 0.90 is the GB10 ceiling for a 15 GiB model; **0.95 hard-reboots GB10 in graph capture**. With ~91 GiB weights/node we start low. Raise to 0.84 only past all gates, one step, watching host free mem | UNMEASURED for this model; cliffs from forum 380732 |
| `--context-length 65536 --max-running-requests 2` | quant author's tested envelope; also caps contention while the tilelang-collapse risk (G7) is unretired. Model supports 1M; raising ctx is a later, gated experiment | LibertAIDAI card |
| `--disable-flashinfer-autotune` | autotuner's 25–40 GB transient allocations are invisible to SGLang accounting on unified memory; also non-deterministic boots | hasso5703 recipe |
| `--memory 115g` docker cap | fail as container-OOM, not host wedge (wedged GB10 = unplug/replug recovery) | GB10 unified-memory lesson |
| `--stream-interval 1`, `--sleep-on-idle` | client token-count fidelity; idle CPU-spin fix | forum 380732 |
| `NCCL_IB_HCA` both twins, `MERGE_NICS=1`, three socket vars (NCCL both rails, GLOO/TP first rail only) | dual-rail fabric ≈184 Gb/s; Gloo/TP steer TCP control plane | cluster baseline; **confirm inside the container, not the shell — G2** |
| `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0` | required in the proven GB10 dual-Spark DFlash2 deploy | forum 380732 |

## 6. Verification gates (in order; a failed gate is a hard stop)

- **G1 — patch presence** (IMPLEMENTATION.md Step 3), both nodes, WITH the negative
  control against the unpatched image.
- **G2 — env at point of effect:** `docker exec glm53-dflash-rank0 env | grep -E
  'NCCL|GLOO|TP_SOCKET'` on both nodes. A variable you did not confirm arrived is a
  variable you did not set.
- **G3 — boot log:** DFLASH worker init lines present; no silent fallback to
  non-speculative; no NaN/assert warnings from tilelang/DSA init. `NET/IB` (RoCE) in
  NCCL init lines, both HCAs listed.
- **G4 — identity from the artifact:** `docker exec glm53-dflash-rank0 cat /proc/1/cmdline
  | tr '\0' ' '` must show `LibertAIDAI/GLM-5.3-Flash-NVFP4` and the DFLASH flags.
  SGLang will echo whatever served-name you configured — argv is the evidence,
  `/v1/models` is a label.
- **G5 — real generation, cross-tailnet, anti-echo:** from the Mac mini, `stream:false`,
  a prompt whose correct answer shares no 10-gram with the prompt. Assert the completion
  is not a prompt echo (the known sm_121 vLLM failure shape), is coherent, and
  `usage.completion_tokens` > 50. Never probe from loopback.
- **G6 — losslessness spot-check (one-time):** same 5 prompts, temp 0, against a
  DFLASH-off launch (drop the three speculative flags) — outputs must match token-for-token.
  DFlash2 is lossless by construction; this catches a broken capture/verify path, which
  is exactly the part we patched in. Costs one extra boot cycle; worth it once.
- **G7 — acceptance, by NAME, with profile:** scrape `/metrics`, match metric names
  containing `spec_accept` (gauges on SGLang — sample DURING active decode, not idle).
  Expect the code-vs-prose spread (Qwen3.8 reference: ~5/8 code, ~3/8 prose — GLM values
  UNMEASURED). A flat/degenerate accept profile with normal-looking tok/s = broken
  drafter, stop. Then the contention probe: 2 concurrent code generations, scan outputs
  for the token-collapse signature (runs of `!` / token-0 floods) seen once on our Qwen
  lane's tilelang path.
- **G8 — fabric really carrying traffic:**
  `/sys/class/infiniband/rocep1s0f1/ports/1/counters/port_xmit_data` (and the P2 twin)
  advancing by GB-scale deltas during a long decode, both rails. `/sys/class/net`
  statistics stay near zero for RDMA — that is expected, not idle.

Append every gate's numbers + exact commands to the deploy log.

## 7. Benchmark protocol (only after all gates)

Every number is recorded with **prompt name, max_tokens, and clock state** — all three
move results more than most effects being measured.

1. Warm: 2× real 800-token generations (cold penalty ~30% on this cluster's experience;
   returns after idle).
2. `stream:false`, read `usage.completion_tokens`, wall-clock from response timing.
3. Fixed prompt set: `code-1` (implement a nontrivial function, ~200-token prompt) and
   `prose-1` (essay), 800 max_tokens, n≥5 each, report median ± spread.
4. Concurrency: c1 and c2 (the max-running-requests cap). Aggregate and per-stream.
5. Same protocol once with the three DFLASH flags removed → the speedup ratio, measured
   not vibed. Reference points, clearly not ours: same model/hardware on vLLM+MTP =
   21.8 tok/s decode (tonyd2wild, 2026-08-27); Qwen3.8-27B+DFlash2 dual-Spark = ~87 tok/s
   code / ~41 prose (forum 380732). **GLM+DFlash2: UNMEASURED until this step.**

## 8. Cutover

Only after §6+§7: repoint clients (Hermes first) from :8899 to :8901, watch a matching
request appear in spark-1's GLM `/metrics` (provenance by metrics, not by client UI),
then decommission the Qwen containers in a later, separate decision. Keep the Qwen
image + cache untouched for rollback regardless.

## 9. Rollback (to the Qwen lane)

```bash
# both nodes:
docker stop glm53-dflash-rank0 glm53-dflash-rank1 2>/dev/null || docker ps  # stop GLM lane
# then relaunch the Qwen lane with its EXISTING start script (worker first),
# containers qwen38-flash-next-worker / -head, image qwen38-flashnext-dspark:local
```

Verify rollback the same way as deploy: argv (G4), cross-tailnet generation (G5).
Weights and caches for both lanes coexist on disk (2.7 TB free) — rollback is a restart,
never a re-download.

## 10. Known issues — the honest list

| Issue | Status |
|---|---|
| **Nobody has run GLM-5.3+DFlash2 anywhere public** — drafter downloads: 0; enabling PR merged 2026-08-27 18:23Z | every combined number UNMEASURED; expect at least one gate to fail on first bring-up |
| Published image predates the DFLASH adapter by 13 h | one-file overlay, IMPLEMENTATION.md; check for upstream rebuild first |
| `fa4` draft attention backend (incoai quickstart) on sm_121 | unverified; omitted — default used, per the proven GB10 DFlash2 deploy |
| Drafter license | research/eval only — production use needs inco's written consent; operator decision |
| Drafter gate is manual | blocking human step; 0 downloads implies no queue history to estimate approval latency |
| mem-fraction-static with drafter on 91 GiB-weight model | UNMEASURED; start 0.80, ceiling 0.84; GB10 hard-reboots near 0.95 in graph capture (recovery: unplug/replug, power button is dead when wedged) |
| DFlash2 × YaRN | incompatible (forum 380732 build). We run native 65536 ctx — do not bolt YaRN on later to extend; the model is natively 1M, extend via `--context-length` + KV budget instead, as a gated experiment |
| tilelang collapse-under-contention (`!`-floods) seen on our Qwen lane | mitigation: max-running-requests 2 + G7 contention probe; not yet observed on glm5_next path |
| Prefix/radix cache on SM_121 hit `CUBLAS_STATUS_INTERNAL_ERROR` on one other model | not disabling preemptively; if hit, add `--disable-radix-cache` and log it |
| fp8 draft KV / fp8 target KV / ctx > 65536 / max-running > 2 / TP4 | all later, gated experiments; each currently UNMEASURED on this stack |

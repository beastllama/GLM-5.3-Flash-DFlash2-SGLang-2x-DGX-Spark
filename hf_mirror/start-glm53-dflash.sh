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
  -e SGLANG_HOST_IP=10.10.10.$((RANK+1)) \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_IB_MERGE_NICS=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e TP_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 \
  glm53-flash-dflash:c4d5d45e5-gb10tile \
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
    --mem-fraction-static 0.88 \
    --context-length 65536 --max-running-requests 2 --max-total-tokens 131072 --mamba-full-memory-ratio 2 \
    --disable-flashinfer-autotune \
    --stream-interval 1 --sleep-on-idle \
    --host 0.0.0.0 --port 8901

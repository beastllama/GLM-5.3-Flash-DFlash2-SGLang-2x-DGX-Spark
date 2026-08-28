DRAFT — numbers marked [] filled after G6; final sign-off before posting.

---
🔥 GLM-5.3-Flash (320B MoE) + DFlash2 speculative decoding — now running on SGLang, 2× NVIDIA DGX Spark

Hours behind @Tech2Wild's vLLM run 🫡 — but this is the FIRST on the upstream SGLang path (PR #36507), and it took 4 new day-0 fixes nobody had hit:

⚡ 27.6 tok/s code decode (1.88× vs same stack no-spec; prose 1.41×)
🎯 accept len up to 5.6 of 8-token blocks
🧨 found: SGLang multi-node hang on GB10 (SGLANG_HOST_IP)
🧨 found: DFLASH×hybrid-KDA memory law (mamba_ratio=5!)
🧨 found: GB10 smem overflow at verify shape (166KB>101KB)
🧨 found: residual=None crash in the 7-hour-old capture adapter

Losslessness: 1 exact match, 1 near-tie flip at temp 0 — NOT claiming bit-identical yet, full matrix queued. Honest numbers only.

Full recipe, patches, probes, honest numbers:
github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark

Improvements coming: 262K ctx, fp8 KV, concurrency ladder, --enable-metrics 🧵

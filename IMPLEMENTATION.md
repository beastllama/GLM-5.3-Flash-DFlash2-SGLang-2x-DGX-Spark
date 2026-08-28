# IMPLEMENTATION — the engine work that remains (and it is small)

## Situation

The published serving image `lmsysorg/sglang:glm-5.3-flash-arm64`
(digest `sha256:73f9294b78e38…`, pushed 2026-08-27T05:22Z) was built from the
`xinyuan/glm-5.3-flash-support` branch **13 hours before** PR #36708 merged the DFLASH
capture adapter into that branch (2026-08-27T18:23Z). Everything else DFLASH needs is
already in the image lineage: `dflash_worker_v2.py`, `dflash_info*.py`, the
`--speculative-algorithm DFLASH` arg path, and the whole glm5_next model.

**The only engine work is bringing one file, `sglang/srt/models/glm5_next.py`, up to
branch head `c4d5d45e506dcd978a65661a503eda1a272c39a4`.** No kernels, no new ops —
`hc_contract` (the one new import) already ships in the image's
`sglang/kernels/ops/layernorm/mhc.py` (added by #36507 itself, +214 lines, in the
05:22 build).

## Step 0 — check whether the work already evaporated

Upstream rebuilds these per-model tags. Before patching:

```bash
curl -s "https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/glm-5.3-flash-arm64" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['tag_last_pushed'], d['images'][0]['digest'])"
```

If `tag_last_pushed` > 2026-08-27T18:23Z, pull the fresh tag and run the **patch-presence
probe** (Step 3) — if it passes, skip Steps 1–2 entirely. Also watch #36507 itself: once
it merges to main, nightly `dev` images supersede this whole dance.

## Step 1 — fetch the authoritative file (not the diff — the whole file, from the SHA)

We overwrite the file with the branch-head version rather than applying a patch, so the
result is bit-identical to what #36708's author tested, and re-running is idempotent:

```bash
# on spark-1
cd ~/glm53-dflash2
curl -fL -o glm5_next.py \
  "https://raw.githubusercontent.com/sgl-project/sglang/c4d5d45e506dcd978a65661a503eda1a272c39a4/python/sglang/srt/models/glm5_next.py"
grep -c "set_dflash_layers_to_capture" glm5_next.py   # must be >= 1
grep -c "hc_contract" glm5_next.py                    # must be >= 2 (import + use)
```

## Step 2 — build the patched image (Dockerfile, not docker-commit — reproducible)

First locate the in-image module path (do not assume it):

```bash
docker run --rm lmsysorg/sglang:glm-5.3-flash-arm64 \
  python3 -c "import sglang.srt.models.glm5_next as m; print(m.__file__)"
```

Then:

```dockerfile
# Dockerfile.dflash
FROM lmsysorg/sglang:glm-5.3-flash-arm64
# PR 36708 (merged into refs/pull/36507/head at c4d5d45e5): DFLASH mHC capture adapter
COPY glm5_next.py <PATH-PRINTED-ABOVE>
```

```bash
docker build -f Dockerfile.dflash -t glm53-flash-dflash:c4d5d45e5 .
# ship to worker — never rebuild there (WORKER_BUILD=0 discipline):
docker save glm53-flash-dflash:c4d5d45e5 | ssh spark-2 docker load
# verify digest identity on both nodes:
docker images --digests glm53-flash-dflash:c4d5d45e5   # run on both; must match
```

## Step 3 — patch-presence probe (execute, don't read)

Run on BOTH nodes; this is the gate that the adapter is really in the serving path:

```bash
docker run --rm glm53-flash-dflash:c4d5d45e5 python3 - <<'PY'
import inspect
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration, Glm5NextModel
assert hasattr(Glm5NextForConditionalGeneration, "set_dflash_layers_to_capture"), "36708 MISSING"
src = inspect.getsource(Glm5NextModel._prepare_aux_hidden_state)
assert "hc_contract" in src, "mHC contraction MISSING"
print("PATCH OK: DFLASH capture adapter present")
PY
```

Negative control (the harness must be able to fail): run the same probe once against the
UNPATCHED `lmsysorg/sglang:glm-5.3-flash-arm64` — it must print `36708 MISSING`. If it
doesn't, the probe is theater and the pulled image was already rebuilt (see Step 0).

## The upstream diff being carried (verbatim, PR #36708)

Only the model-file hunk matters in-container; the unit test stays upstream.
Semantics: (1) new hook `set_dflash_layers_to_capture(layer_ids)` — sets
`capture_aux_hidden_states`, flips `model.dflash_capture`, captures at `layer_id + 1`;
(2) `_prepare_aux_hidden_state` = `hidden_states + residual`, then, when
`config.mhc`, `hc_contract(x, hc_mult)` (mean over the `hc_mult` groups) so the drafter
sees `hidden_size`, not `hc_mult × hidden_size`; (3) TBO layer-overlap disabled while
capturing (`can_run_tbo and not self.dflash_capture`); (4) EAGLE3 capture path untouched.

```diff
--- a/python/sglang/srt/models/glm5_next.py
+++ b/python/sglang/srt/models/glm5_next.py
@@ -8,6 +8,7 @@
 from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
+from sglang.kernels.ops.layernorm.mhc import hc_contract
 from sglang.kernels.ops.layernorm.mhc import hc_post as _hc_post_fn
 from sglang.kernels.ops.layernorm.mhc import hc_pre as _hc_pre_fn
@@ -959,6 +960,7 @@ def __init__(
         super().__init__()
+        self.config = config
         self.padding_id = config.pad_token_id
@@ -1064,6 +1066,7 @@
         self.layers_to_capture = []
+        self.dflash_capture = False
@@ -1072,6 +1075,14 @@
     def get_input_embeddings(self) -> torch.Tensor:
         return self.embed_tokens

+    def _prepare_aux_hidden_state(
+        self, hidden_states: torch.Tensor, residual: torch.Tensor
+    ) -> torch.Tensor:
+        aux_hidden_state = hidden_states + residual
+        if self.dflash_capture and self.config.mhc:
+            aux_hidden_state = hc_contract(aux_hidden_state, self.config.hc_mult)
+        return aux_hidden_state
+
@@ -1122,7 +1133,7 @@
-        if forward_batch.can_run_tbo:
+        if forward_batch.can_run_tbo and not self.dflash_capture:
@@ -1141,13 +1152,14 @@
             with ctx:
                 if i in self.layers_to_capture:
+                    aux_hidden_state = self._prepare_aux_hidden_state(
+                        hidden_states, residual
+                    )
                     if self.enable_a2a_moe and i > self.first_k_dense_replace:
                         aux_hidden_state = get_parallel().attn_tp_group.all_gather(
-                            hidden_states + residual, dim=0
+                            aux_hidden_state, dim=0
                         )
-                        aux_hidden_states.append(aux_hidden_state)
-                    else:
-                        aux_hidden_states.append(hidden_states + residual)
+                    aux_hidden_states.append(aux_hidden_state)
@@ -1388,6 +1400,20 @@
     def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
         ...unchanged...
+    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
+        if not self.pp_group.is_last_rank:
+            return
+        if layer_ids is None:
+            raise ValueError(
+                "DFLASH requires explicit layer_ids for aux hidden capture."
+            )
+        self.capture_aux_hidden_states = True
+        self.model.dflash_capture = True
+        # Capturing before layer k + 1 gives the completed output of layer k.
+        self.model.layers_to_capture = [val + 1 for val in layer_ids]
```

(Abridged for context; Step 1 uses the full file from the SHA, never this excerpt.)

## Risk register for the DFLASH bring-up, with planned reactions

| # | Risk | Signal | Reaction |
|---|------|--------|----------|
| 1 | `--speculative-draft-attention-backend fa4` (from the incoai card, written for GB300) unsupported on sm_121 | server arg error or crash at draft init | omit the flag (the proven Qwen3.8 GB10 DFlash2 deploy used the default); if default also fails, try `triton` |
| 2 | DFLASH × DSA-tilelang interaction never run on sm_121 | crash in verify step, or garbage accepts (flat per-position profile) | fall back to MTP lane (below) and file upstream issue with repro |
| 3 | Drafter arch not loadable by the image's generic `models/dflash.py` | load error naming the drafter's `architectures` | diff drafter `config.json` (readable only post-gate) against `z-lab/Qwen3.8-27B-DFlash2`'s; port deltas or hold for upstream |
| 4 | GB10 hard reboot during "Capture target verify CUDA graph" | node power-cycles (recovery = unplug/replug, NOT the power button) | we start at mem-fraction 0.80, ceiling 0.84 — never approach the 0.90/0.95 cliff with a 90 GiB-weight model |
| 5 | TileLang logits data race / token-`!` collapse under contention (seen on our Qwen lane) | repeated `!` or token-0 flood in concurrent streams | `--max-running-requests 2` at first; run the contention quality probe (README gate G7) before raising it |

## Fallback ladder (pre-decided, so a failed gate doesn't improvise)

1. **v1-fallback — same image, MTP instead of DFLASH.** The branch ships
   `glm5_next_nextn.py` and the NVFP4 quant retains the BF16 MTP head. tonyd2wild's vLLM
   numbers suggest MTP accept-len 2.5–2.9 on this model. Flags: swap the three DFLASH
   flags for the image's MTP/EAGLE-family equivalents (`--speculative-algorithm` per
   cookbook; exact GLM MTP flag set UNMEASURED on GB10 — one experiment).
2. **v1-alternative — tonyd2wild's vLLM+MTP recipe wholesale** (21.8 tok/s decode, 262K
   ctx, proven on identical hardware 2026-08-27). Trades DFlash upside for zero novelty.
3. **v0 — stay on the Qwen lane** (current production, untouched throughout).

## Upstream watch (retire local patch ASAP)

- PR #36507 merge to main → glm5_next in nightlies → repin to
  `lmsysorg/sglang:nightly-dev-…` or a rebuilt `glm-5.3-flash-arm64`, drop our overlay.
- Watch for a `glm-5.3-flash-dflash2` per-model tag (upstream did exactly this for
  qwen38-27b: `dev-*-qwen38-27b-dflash2`, 2026-08-22).
- If risk #3 materializes, the drafter-loader delta is a candidate upstream PR from us.

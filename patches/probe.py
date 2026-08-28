import inspect, sys
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
ok = hasattr(Glm5NextForConditionalGeneration, "set_dflash_layers_to_capture")
print("dflash capture adapter present:", ok)
sys.exit(0 if ok else 1)

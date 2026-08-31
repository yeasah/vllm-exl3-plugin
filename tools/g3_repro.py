import sys
from vllm import LLM, SamplingParams
# gemma-3-1b-it: 26 layers, sliding_window_pattern 6 -> full attention at 5,11,17,23.
# TurboQuant cannot serve a sliding window, so the sliding layers keep a native cache.
sliding = [str(i) for i in range(26) if (i + 1) % 6 != 0]
extra = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] else []
llm = LLM(model="unsloth/gemma-3-1b-it", kv_cache_dtype="turboquant_4bit_nc",
          kv_cache_dtype_skip_layers=sliding + extra,
          max_model_len=2048, gpu_memory_utilization=0.60, enforce_eager=True)
cc = llm.llm_engine.vllm_config.cache_config
print("EFFECTIVE SKIPS:", len(cc.kv_cache_dtype_skip_layers), "layers; block", cc.block_size)
o = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
print("OK:", o[0].outputs[0].text[:60])
import os; sys.stdout.flush(); os._exit(0)

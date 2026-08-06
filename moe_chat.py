import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams
model, rev, dtype = sys.argv[1], sys.argv[2], sys.argv[3]
llm = LLM(model=model, revision=rev, dtype=dtype, enforce_eager=True,
          gpu_memory_utilization=0.92, max_model_len=2048)
convs = [[{"role":"user","content":"What is the capital of France? Answer in one sentence."}],
         [{"role":"user","content":"What is 2 + 2?"}]]
for c, o in zip(convs, llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=48))):
    print(f"\nUSER: {c[0]['content']}\nMODEL: {o.outputs[0].text!r}", flush=True)

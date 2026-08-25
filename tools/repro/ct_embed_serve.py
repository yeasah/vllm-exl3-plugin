"""Serve llm-compressor's quantized-embedding output in vLLM, per its README's
claim that such checkpoints are "ready to be loaded into vLLM"."""
import os, sys


def main():
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM
    llm = LLM(model="/home/ypell/ckpt/pythia-160m-embedding-W4A16-G64",
              max_model_len=512, gpu_memory_utilization=0.35, enforce_eager=True)
    out = llm.generate(["The capital of France is"], use_tqdm=False)
    print("LOADED_OK", repr(out[0].outputs[0].text[:40]))
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()

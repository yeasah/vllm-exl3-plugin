"""llm-compressor's own documented embedding recipe, on its own example family.

The README validates embedding quantization on `pythia-1.4b` and states the result
is "ready to be loaded into vLLM". Pythia is GPTNeoXForCausalLM, which vLLM serves
from gpt_neox.py -- a file that constructs its VocabParallelEmbedding without
passing quant_config. This produces the checkpoint that tests that claim.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "EleutherAI/pythia-160m"
SAVE_DIR = "/home/ypell/ckpt/pythia-160m-embedding-W4A16-G64"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    config_groups={
        "embedding": {
            "targets": ["Embedding"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True,
                        "strategy": "group", "group_size": 64},
        }
    }
)
oneshot(model=model, recipe=recipe)          # weight-only -> data-free
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print("SAVED", SAVE_DIR)

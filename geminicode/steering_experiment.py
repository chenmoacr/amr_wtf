import torch
import json
import os
import sys
import time
from pathlib import Path
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

# Configuration
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DATA_PATH = "J:/amr/amr_wtf/data/gemma_TargetedDrugs/gemma_code_GB01.json"
GEMINI_DIR = Path("J:/amr/amr_wtf/geminicode")
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

def load_input():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['A'][0]['query']

def apply_steering(model, steering_config):
    """
    steering_config: { layer_idx: { neuron_idx: alpha } }
    """
    hooks = []

    def create_hook(layer_idx, config):
        def hook_fn(module, args):
            # args is a tuple of (input_tensor, ...)
            # We modify the first element of args
            x = args[0]
            for ni, alpha in config.items():
                x[:, :, ni] += alpha
            return (x,) + args[1:]
        return hook_fn

    layers = model.model.language_model.layers
    for li, config in steering_config.items():
        # Hooking the down_proj's input which is the latent activation
        h = layers[li].mlp.down_proj.register_forward_pre_hook(create_hook(li, config))
        hooks.append(h)

    return hooks

def run_gen(model, tokenizer, prompt, name):
    print(f"\n[Run] Generating for: {name}")
    log_file = GEMINI_DIR / f"run_{name}.log"

    msgs = [{"role": "user", "content": prompt}]
    # Thinking mode is usually triggered by specific prompt or system instruction in Gemma 4
    # Here we assume the base prompt from Case A already includes necessary context.

    inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) or hasattr(inputs, "data") else inputs

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=32768, # Adjusted to a safe high value for stable generation
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
    duration = time.time() - start_time

    output_content = f"Experiment: {name}\nDuration: {duration:.2f}s\n\n{gen_text}"
    log_file.write_text(output_content, encoding="utf-8")
    print(f"Done. Output saved to {log_file}")
    return gen_text

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE
    )

    query = load_input()

    # V2: Only Mid/Late - Focusing on high-quality code generation
    config_v2 = {
        28: {2033: -2.0},
        33: {9054: 2.0}
    }

    # Run V2 with high max_tokens
    hooks_v2 = apply_steering(model, config_v2)
    # Using 16384 as a practical high limit for Gemma 4 context window stability,
    # though 65535 is requested, the model's KV cache and context usually caps lower.
    final_output = run_gen(model, tokenizer, query, "steer_v2_max_tokens")
    for h in hooks_v2: h.remove()

    # Extract code and save as a standalone file
    with open(GEMINI_DIR / "steer_v2_final_code.cpp", "w", encoding="utf-8") as f:
        f.write(final_output)
    print(f"\nFinal C++ code saved to {GEMINI_DIR / 'steer_v2_final_code.cpp'}")

if __name__ == "__main__":
    main()

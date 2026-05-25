"""
Phase 1: Verify Modal setup + load Qwen + install a no-op attention hook.
The hook does nothing yet. We just want to confirm the monkey-patch
mechanism works before we put real ADHD logic into it.

Run with: modal run phase1.py
"""

import modal

app = modal.App("adhd-attention")

# Container image with pinned versions (avoids surprise breakage)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "hf_transfer==0.1.8",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Persistent volume for cached model weights, so we only download once
cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": cache_vol},
    timeout=60 * 30,
)
def run_baseline_and_hooked(prompt: str = "In two sentences, explain why the sky is blue."):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/cache")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir="/cache",
        attn_implementation="eager",  # plain Python attention so we can monkey-patch it
    )
    model.eval()
    print("Model loaded.\n")

    def generate(label: str):
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"=== {label} ===")
        print(response)
        print()
        return response

    # 1. Baseline run
    baseline = generate("BASELINE")

    # 2. Install a no-op hook on the attention forward pass
    original_forward = Qwen2Attention.forward

    def hooked_forward(self, hidden_states, *args, **kwargs):
        # Phase 3 will insert the ADHD mask logic HERE.
        # For Phase 1, just pass through and confirm nothing breaks.
        return original_forward(self, hidden_states, *args, **kwargs)

    Qwen2Attention.forward = hooked_forward

    # 3. Run again with the hook installed
    hooked = generate("HOOKED (no-op)")

    # 4. Restore so we don't leave a global monkey-patch lying around
    Qwen2Attention.forward = original_forward

    match = "MATCH" if baseline == hooked else "MISMATCH (something's off)"
    print(f"Baseline vs hooked: {match}")
    return baseline, hooked


@app.local_entrypoint()
def main():
    run_baseline_and_hooked.remote()
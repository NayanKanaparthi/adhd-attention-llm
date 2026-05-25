"""
Phase 2: Permute-CoT experiment.

For each test prompt:
1. Generate a chain-of-thought response (sampling, seeded for reproducibility).
2. Parse the CoT into discrete reasoning steps.
3. Build three orderings: original, reversed, randomly shuffled.
4. For each ordering, feed the steps back and ask for a final answer (greedy).
5. Print everything so we can compare side by side.

Run with: modal run phase2.py
"""

import modal
import re
import random

app = modal.App("adhd-attention-phase2")

# Same image as phase1, Modal will reuse the cached build.
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

# Same cache volume, weights already downloaded.
cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TEST_PROMPTS = [
    ("math",     "If 3 apples cost $4.50, how much do 7 apples cost?"),
    ("logic",    "Alice is taller than Bob. Bob is taller than Charlie. Carol is taller than Alice. Who is the shortest and who is the tallest?"),
    ("creative", "Invent a new use for a paperclip that nobody has thought of before. Explain why it would work."),
    ("open",     "What would change about human society if humans could photosynthesize and didn't need to eat?"),
]


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": cache_vol},
    timeout=60 * 30,
)
def run_phase2(seed: int = 42):
    import torch
    import warnings
    from transformers import AutoModelForCausalLM, AutoTokenizer

    warnings.filterwarnings("ignore", category=UserWarning)

    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/cache")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir="/cache",
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded.\n")

    # ---------- helpers ----------
    def chat_generate(prompt: str, max_tokens: int = 400, sample: bool = False) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=sample,
                temperature=0.7 if sample else 1.0,
                top_p=0.9 if sample else 1.0,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def generate_cot(prompt: str) -> str:
        cot_request = (
            "Think through this problem step by step. Format your response EXACTLY like this:\n\n"
            "Step 1: [first reasoning step]\n"
            "Step 2: [second reasoning step]\n"
            "Step 3: [third reasoning step]\n"
            "(continue with as many steps as needed, usually 3 to 6)\n"
            "Answer: [your final answer in one sentence]\n\n"
            f"Problem: {prompt}"
        )
        return chat_generate(cot_request, max_tokens=500, sample=True)

    def parse_steps(cot_text: str):
        """Return (list_of_step_strings, final_answer_string)."""
        # Split on "Step N:" allowing optional markdown bold (**) around the
        # label only. We intentionally do NOT strip all asterisks, because the
        # reasoning content can contain "*" as multiplication (e.g. "1.50 * 7").
        parts = re.split(r"(?:^|\n)\s*\*{0,2}\s*Step\s*\d+\s*:\s*\*{0,2}\s*", cot_text)
        if len(parts) < 2:
            return [], "[no steps parsed]"
        step_chunks = parts[1:]
        last = step_chunks[-1]
        ans_split = re.split(r"\n\s*\*{0,2}\s*Answer\s*:\s*\*{0,2}\s*", last, maxsplit=1)
        if len(ans_split) == 2:
            step_chunks[-1] = ans_split[0]
            answer = ans_split[1].strip()
        else:
            answer = "[no answer line parsed]"
        steps = [f"Step {i+1}: {s.strip()}" for i, s in enumerate(step_chunks)]
        return steps, answer

    def answer_from_ordering(prompt: str, ordered_steps: list) -> str:
        joined = "\n".join(ordered_steps)
        re_prompt = (
            "Below are reasoning steps for a problem. Based ONLY on these steps, "
            "give the final answer in one or two sentences. Do not introduce new reasoning.\n\n"
            f"Problem: {prompt}\n\n"
            f"Reasoning steps:\n{joined}\n\n"
            "Final answer:"
        )
        return chat_generate(re_prompt, max_tokens=200, sample=False)

    # ---------- experiment ----------
    for category, prompt in TEST_PROMPTS:
        print("=" * 80)
        print(f"[{category.upper()}] {prompt}")
        print("=" * 80)

        cot = generate_cot(prompt)
        print("\n--- ORIGINAL CoT ---")
        print(cot)

        steps, original_answer = parse_steps(cot)

        if len(steps) < 2:
            print(f"\n!! Could not parse 2+ steps (got {len(steps)}). Skipping.\n")
            continue

        n = len(steps)
        original_order = list(range(n))
        reverse_order = list(reversed(range(n)))
        shuffled_order = original_order.copy()
        random.shuffle(shuffled_order)
        if shuffled_order == original_order:
            shuffled_order = reverse_order

        orderings = [
            ("ORIGINAL ORDER", original_order),
            ("REVERSED",       reverse_order),
            ("SHUFFLED",       shuffled_order),
        ]

        print(f"\n--- ANSWER stated in original CoT ---")
        print(original_answer)

        for label, order in orderings:
            ordered_steps = [steps[i] for i in order]
            ans = answer_from_ordering(prompt, ordered_steps)
            print(f"\n--- DERIVED ANSWER ({label}: order={order}) ---")
            print(ans)

        print()


@app.local_entrypoint()
def main():
    run_phase2.remote()
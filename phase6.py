"""
Phase 6: Wanderer with attention temperature flattening.

Same Wanderer/Critic/Synthesis loop as Phase 5, BUT:
- Replace random per-weight attention dropout (Phase 5) with attention temperature
  flattening BEFORE softmax.
- Random dropout punches binary holes (some weights to 0). Phase 5 showed this causes
  catastrophic degeneracy at 7B over 200-token generations (33% failure rate).
- Temperature flattening is smoother: divide pre-softmax logits by T>1, making the
  resulting distribution less peaked. Nothing is zeroed. More peripheral tokens
  contribute, but none are randomly silenced.
- This is the framework's primary "loose attention" mechanism. Phase 5 tested the
  secondary mechanism (dropout) and it failed. Phase 6 tests the primary one.

Critic and Synthesis still run with stock attention (T=1.0).

Run with: modal run phase6.py
"""

import modal
import math
import re
import warnings

app = modal.App("adhd-attention-phase6")

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

cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Same prompts as Phase 4 and Phase 5 for direct comparison
TEST_PROMPTS = [
    "Invent a new use for a paperclip that nobody has thought of before. Explain why it would work.",
    "What would change about human society if humans could photosynthesize and didn't need to eat?",
    "How could we make online learning genuinely engaging instead of feeling like homework?",
]

# Global controller for attention temperature.
# 1.0 = stock attention (peaked, normal).
# >1.0 = flatter attention (more peripheral tokens contribute).
# <1.0 = sharper attention (only top tokens dominate). Not what we want here.
ATTENTION_TEMPERATURE = 1.0


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache": cache_vol},
    timeout=60 * 60,
)
def run_phase6(seed: int = 42, n_candidates: int = 6, wanderer_temperature: float = 2.0):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention, apply_rotary_pos_emb, repeat_kv,
    )

    global ATTENTION_TEMPERATURE
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.manual_seed(seed)

    print(f"Loading {MODEL_NAME} with eager attention...")
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

    # ---------- patched attention forward with controllable temperature ----------
    original_forward = Qwen2Attention.forward

    def patched_forward(
        self, hidden_states, position_embeddings=None, attention_mask=None,
        past_key_value=None, cache_position=None, **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention logits (pre-softmax scores)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # ---- ADHD intervention: temperature flattening BEFORE softmax ----
        # T > 1 -> smoother distribution -> more peripheral tokens contribute.
        # T = 1 -> stock attention (this branch skipped entirely).
        if ATTENTION_TEMPERATURE != 1.0:
            attn_weights = attn_weights / ATTENTION_TEMPERATURE
        # -----------------------------------------------------------------

        # Softmax (upcast to fp32 for stability)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_weights = attn_weights.to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value

    # ---------- verify patched forward at T=1.0 matches stock ----------
    print("Verifying patched forward at ATTENTION_TEMPERATURE=1.0 matches stock attention...")
    ATTENTION_TEMPERATURE = 1.0
    test_prompt = "The capital of France is"
    test_inputs = tok(test_prompt, return_tensors="pt").to("cuda")

    # First, generate with stock forward
    torch.manual_seed(seed)
    with torch.no_grad():
        stock_out = model.generate(
            **test_inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id
        )
    stock_text = tok.decode(stock_out[0], skip_special_tokens=True)

    # Install patch
    Qwen2Attention.forward = patched_forward

    # Now generate with patched forward (T=1.0, should be identical)
    torch.manual_seed(seed)
    with torch.no_grad():
        patched_out = model.generate(
            **test_inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id
        )
    patched_text = tok.decode(patched_out[0], skip_special_tokens=True)

    if stock_text == patched_text:
        print(f"  MATCH: patched forward at T=1.0 is byte-identical to stock.\n")
    else:
        print(f"  MISMATCH!")
        print(f"  Stock:   {stock_text!r}")
        print(f"  Patched: {patched_text!r}")
        print(f"  Aborting. The patch is not a clean superset of stock.\n")
        return

    # ---------- generation helpers ----------
    def generate(prompt: str, max_tokens: int = 400, temperature: float = 0.7, sample: bool = True) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=sample,
                temperature=temperature if sample else 1.0,
                top_p=0.9 if sample else 1.0,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ---------- WANDERER: flatter attention, minimal nudge, no domain hints ----------
    def wanderer(problem: str, idx: int) -> str:
        global ATTENTION_TEMPERATURE
        prompt = (
            f"Find an unusual angle on this problem. Propose ONE specific, concrete idea "
            f"that most people wouldn't think of first. Be brief, 2 to 3 sentences. "
            f"Do not give a list. Just one strong, unexpected idea.\n\n"
            f"Problem: {problem}\n\n"
            f"Unexpected idea:"
        )
        torch.manual_seed(seed + (idx + 1) * 1000)
        ATTENTION_TEMPERATURE = wanderer_temperature  # flatter attention ON
        result = generate(prompt, max_tokens=200, temperature=1.0, sample=True)
        ATTENTION_TEMPERATURE = 1.0  # stock attention restored
        return result

    # ---------- CRITIC: stock attention ----------
    def critic(problem: str, candidate: str) -> dict:
        global ATTENTION_TEMPERATURE
        prompt = (
            f"Problem: {problem}\n\n"
            f"Proposed idea: {candidate}\n\n"
            f"Score this idea on three axes from 1 (lowest) to 10 (highest):\n"
            f"- Novelty: how unusual is this compared to obvious or generic answers?\n"
            f"- Usefulness: would this actually help address the problem?\n"
            f"- Coherence: does this make internal sense and hold together?\n\n"
            f"Respond ONLY with three numbers separated by commas, no other text. "
            f"Example: 8,6,7\n\n"
            f"Scores:"
        )
        ATTENTION_TEMPERATURE = 1.0  # tight attention for evaluation
        response = generate(prompt, max_tokens=20, sample=False)
        match = re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", response)
        if match:
            n, u, c = int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            n, u, c = 5, 5, 5
            print(f"  [critic parse failed, using 5,5,5 fallback. raw response: {response!r}]")
        n, u, c = max(1, min(10, n)), max(1, min(10, u)), max(1, min(10, c))
        combined = n * 0.4 + u * 0.4 + c * 0.2
        return {"novelty": n, "usefulness": u, "coherence": c, "combined": combined}

    # ---------- SYNTHESIS: stock attention ----------
    def synthesize(problem: str, top_ideas: list) -> str:
        global ATTENTION_TEMPERATURE
        numbered = "\n\n".join(f"Idea {i+1}: {idea}" for i, idea in enumerate(top_ideas))
        prompt = (
            f"Problem: {problem}\n\n"
            f"You generated several unusual ideas for this problem:\n\n{numbered}\n\n"
            f"Synthesize the strongest elements into a single, coherent, novel proposal. "
            f"Preserve the unexpected combinations rather than averaging into something generic. "
            f"Be specific and concrete.\n\n"
            f"Synthesized answer:"
        )
        ATTENTION_TEMPERATURE = 1.0
        torch.manual_seed(seed + 999)
        return generate(prompt, max_tokens=500, temperature=0.5, sample=True)

    # ---------- experiment loop ----------
    for prompt in TEST_PROMPTS:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)

        # 1. Baseline: stock attention, single shot
        print("\n--- BASELINE (stock attention T=1.0, single shot) ---")
        ATTENTION_TEMPERATURE = 1.0
        torch.manual_seed(seed)
        baseline = generate(prompt, max_tokens=400, temperature=0.7, sample=True)
        print(baseline)
        baseline_score = critic(prompt, baseline)
        print(f"  Baseline score: N={baseline_score['novelty']} U={baseline_score['usefulness']} "
              f"C={baseline_score['coherence']} combined={baseline_score['combined']:.1f}")

        # 2. Wanderer: flatter attention, no domain hints
        print(f"\n--- WANDERER ({n_candidates} candidates, attention temperature={wanderer_temperature}, no domain hints) ---")
        candidates = []
        for i in range(n_candidates):
            print(f"\n  [{i+1}]")
            cand = wanderer(prompt, i)
            candidates.append(cand)
            print(f"  {cand}")

        # 3. Critic
        print(f"\n--- CRITIC scoring each candidate (stock attention) ---")
        scored = []
        for i, cand in enumerate(candidates):
            scores = critic(prompt, cand)
            scored.append((i, cand, scores))
            print(f"  [{i+1}] N={scores['novelty']} U={scores['usefulness']} "
                  f"C={scores['coherence']} -> combined={scores['combined']:.1f}")

        # 4. Top 3
        scored.sort(key=lambda x: x[2]["combined"], reverse=True)
        top_3 = scored[:3]
        print(f"\n--- TOP 3 SELECTED ---")
        for i, cand, scores in top_3:
            print(f"  [#{i+1}] combined={scores['combined']:.1f}")

        # 5. Synthesis
        print(f"\n--- FINAL SYNTHESIS (combining top 3) ---")
        final = synthesize(prompt, [cand for _, cand, _ in top_3])
        print(final)
        final_score = critic(prompt, final)
        print(f"  Synthesis score: N={final_score['novelty']} U={final_score['usefulness']} "
              f"C={final_score['coherence']} combined={final_score['combined']:.1f}")

        # Direct comparison
        delta = final_score['combined'] - baseline_score['combined']
        print(f"\n--- COMPARISON ---")
        print(f"  Baseline combined:  {baseline_score['combined']:.1f}")
        print(f"  Synthesis combined: {final_score['combined']:.1f}")
        print(f"  Delta:              {delta:+.1f}")

        print()


@app.local_entrypoint()
def main():
    run_phase6.remote()
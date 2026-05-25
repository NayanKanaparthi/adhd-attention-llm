"""
Phase 3: Attention mask intervention.

Monkey-patches Qwen2Attention.forward to apply a configurable "ADHD mask"
to the attention weight matrix RIGHT AFTER softmax. The mask is random dropout:
each attention weight has probability `dropout_rate` of being zeroed.
After zeroing, we renormalize so each row still sums to 1 (so it remains a
valid probability distribution over keys).

We test the same four prompts as Phase 2 at three dropout rates plus baseline,
using greedy decoding so any output differences come from the mask, not from
sampling noise.

Run with: modal run phase3.py
"""

import modal
import random
import math

app = modal.App("adhd-attention-phase3")

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
def run_phase3(seed: int = 42):
    import torch
    import torch.nn as nn
    import warnings
    from typing import Optional, Tuple
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        apply_rotary_pos_emb,
        repeat_kv,
    )
    from transformers.cache_utils import Cache

    warnings.filterwarnings("ignore", category=UserWarning)
    random.seed(seed)
    torch.manual_seed(seed)

    # Mutable config the custom forward reads from. We change this between runs
    # to toggle the ADHD mask on/off and set the dropout rate.
    adhd_state = {"mode": "none", "dropout_rate": 0.0}

    # ---------- custom attention forward with ADHD mask ----------
    def adhd_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Project hidden states to Q, K, V and reshape for multi-head attention.
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings.
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update KV cache if generation is using one.
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat K and V for grouped-query attention (Qwen 2.5 uses GQA).
        key_states   = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores and apply causal mask.
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax (in fp32 for numerical stability, then cast back).
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # ====================== ADHD MASK INSERTION ======================
        if adhd_state["mode"] == "dropout" and adhd_state["dropout_rate"] > 0:
            # Promote to fp32 for the masking + renormalization to avoid bf16 precision loss.
            w = attn_weights.to(torch.float32)
            keep = (torch.rand_like(w) > adhd_state["dropout_rate"]).to(torch.float32)
            w = w * keep
            # Renormalize so each row still sums to 1.
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
            attn_weights = w.to(query_states.dtype)
        # =================================================================

        # Model's own training-time dropout (no-op in eval mode).
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output  = torch.matmul(attn_weights, value_states)

        # Reshape and project back to hidden size.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    # Capture the stock implementation BEFORE patching so we can verify that
    # our reimplementation, with the mask inactive, is faithful to it.
    original_forward = Qwen2Attention.forward

    # Install the patch globally on the Qwen2Attention class.
    Qwen2Attention.forward = adhd_forward

    # ---------- load model ----------
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
    print("Model loaded with ADHD-capable attention.\n")

    # ---------- generation helper ----------
    def generate(prompt: str, max_tokens: int = 250) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    # ---------- verify patched-but-inactive == un-patched (spec requirement) ----------
    # Greedy decoding is deterministic and the inactive mask path adds nothing,
    # so a faithful reimplementation must reproduce the stock model token-for-token.
    def verify_baseline(prompt: str, max_tokens: int = 64):
        adhd_state["mode"], adhd_state["dropout_rate"] = "none", 0.0
        Qwen2Attention.forward = original_forward      # stock attention
        ref = generate(prompt, max_tokens=max_tokens)
        Qwen2Attention.forward = adhd_forward          # our patch, mask inactive
        test = generate(prompt, max_tokens=max_tokens)
        ok = ref == test
        print(f"[VERIFY] patched-inactive == un-patched baseline: "
              f"{'MATCH (patch is faithful)' if ok else 'MISMATCH — patch diverges from stock!'}")
        if not ok:
            print("  un-patched   :", repr(ref[:200]))
            print("  patched(none):", repr(test[:200]))
        print()

    verify_baseline(TEST_PROMPTS[0][1])

    # ---------- conditions ----------
    conditions = [
        ("BASELINE (no mask)", "none",    0.0),
        ("DROPOUT 10%",        "dropout", 0.10),
        ("DROPOUT 30%",        "dropout", 0.30),
        ("DROPOUT 60%",        "dropout", 0.60),
    ]

    # ---------- experiment ----------
    for category, prompt in TEST_PROMPTS:
        print("=" * 80)
        print(f"[{category.upper()}] {prompt}")
        print("=" * 80)

        for label, mode, rate in conditions:
            adhd_state["mode"] = mode
            adhd_state["dropout_rate"] = rate
            # Reseed before each generation so dropout patterns are reproducible.
            torch.manual_seed(seed)
            response = generate(prompt, max_tokens=250)
            print(f"\n--- {label} ---")
            print(response)

        print()
        # Reset to baseline between prompts to keep state tidy.
        adhd_state["mode"] = "none"
        adhd_state["dropout_rate"] = 0.0


@app.local_entrypoint()
def main():
    run_phase3.remote()
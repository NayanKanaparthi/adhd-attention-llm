"""
Phase 4: Wanderer/Architect Loop (controlled disinhibition).

Stock Qwen 7B, no model modification. Pure orchestration on top:
1. WANDERER: generate N candidates, each forced through a different
   cross-domain frame to encourage associative leaps.
2. CRITIC: the same model scores each candidate on novelty, usefulness,
   coherence.
3. SYNTHESIS: combine top-scored candidates into a final novel answer.

Compare against single-shot baseline to see whether the orchestration produces
more novel and useful ideas than just asking once.

No attention modification this phase. If the orchestration works, we add
ADHD attention into the Wanderer stage in Phase 5 as an enhancement.

Run with: modal run phase4.py
"""

import modal
import random
import re

app = modal.App("adhd-attention-phase4")

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

# Domains the Wanderer forces each candidate through. Each is deliberately
# unusual relative to the natural framing of most user problems.
DOMAINS = [
    "biological systems, including how cells, immune systems, or evolution solve problems",
    "MMORPG game design, with guilds, quests, achievements, and player economies",
    "religious ritual and ceremony, with repetition, symbols, and community meaning",
    "jazz improvisation, with structured freedom and call-and-response patterns",
    "architectural and urban planning principles, with flow, density, and shared space",
    "military strategy and tactics, with reconnaissance, deception, and force economy",
    "apprenticeship traditions where masters and learners co-create through doing",
    "fitness training programs with progressive overload, recovery, and measurable gains",
]

TEST_PROMPTS = [
    "Invent a new use for a paperclip that nobody has thought of before. Explain why it would work.",
    "What would change about human society if humans could photosynthesize and didn't need to eat?",
    "How could we make online learning genuinely engaging instead of feeling like homework?",
]


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/cache": cache_vol},
    timeout=60 * 60,
)
def run_phase4(seed: int = 42, n_candidates: int = 6):
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
    )
    model.eval()
    print("Model loaded.\n")

    def generate(prompt: str, max_tokens: int = 400,
                 temperature: float = 0.7, sample: bool = True) -> str:
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
        return tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    # ---------- WANDERER: generate one candidate through one domain lens ----------
    def wanderer(problem: str, domain: str, idx: int) -> str:
        prompt = (
            f"Think about this problem through the lens of {domain}.\n\n"
            f"What surprising frame from that domain could apply to the problem? "
            f"Propose ONE specific, concrete idea drawn from that domain's thinking. "
            f"Do not give a list. Give one strong, unusual idea in 2 to 3 sentences. "
            f"The idea should clearly carry the fingerprint of the domain, not retreat "
            f"to a generic answer.\n\n"
            f"Problem: {problem}\n\n"
            f"Idea drawn from {domain}:"
        )
        # Per-candidate seed for variety + reproducibility
        torch.manual_seed(seed + idx * 1000)
        return generate(prompt, max_tokens=200, temperature=1.0, sample=True)

    # ---------- CRITIC: score a candidate on three axes ----------
    def critic(problem: str, candidate: str) -> dict:
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
        response = generate(prompt, max_tokens=20, sample=False)
        match = re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", response)
        if match:
            n, u, c = int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            n, u, c = 5, 5, 5  # fallback if parsing fails
            print(f"  [critic parse failed, using 5,5,5 fallback. raw response: {response!r}]")
        n, u, c = max(1, min(10, n)), max(1, min(10, u)), max(1, min(10, c))
        # Combined: weight novelty + usefulness equally, coherence less.
        # This biases selection toward "novel and useful" over "safe and tidy".
        combined = n * 0.4 + u * 0.4 + c * 0.2
        return {"novelty": n, "usefulness": u, "coherence": c, "combined": combined}

    # ---------- SYNTHESIS: combine top candidates into a final answer ----------
    def synthesize(problem: str, top_ideas: list) -> str:
        numbered = "\n\n".join(f"Framing {i+1}: {idea}" for i, idea in enumerate(top_ideas))
        prompt = (
            f"Problem: {problem}\n\n"
            f"You have generated several different framings for this problem, "
            f"each drawn from a different domain:\n\n"
            f"{numbered}\n\n"
            f"Synthesize the strongest elements of these framings into a single, "
            f"coherent, novel proposal. Preserve the unexpected combinations rather "
            f"than averaging them into something generic. The final answer should "
            f"clearly carry the fingerprint of the unusual frames, not retreat to a "
            f"conventional response. Be specific and concrete.\n\n"
            f"Synthesized answer:"
        )
        torch.manual_seed(seed + 999)
        return generate(prompt, max_tokens=500, temperature=0.5, sample=True)

    # ---------- experiment ----------
    for prompt in TEST_PROMPTS:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)

        # 1. Baseline (the control)
        print("\n--- BASELINE (single shot, no orchestration) ---")
        torch.manual_seed(seed)
        baseline = generate(prompt, max_tokens=400, temperature=0.7, sample=True)
        print(baseline)
        baseline_score = critic(prompt, baseline)
        print(f"  Baseline score: N={baseline_score['novelty']} U={baseline_score['usefulness']} "
              f"C={baseline_score['coherence']} combined={baseline_score['combined']:.1f}")

        # 2. Wanderer
        print(f"\n--- WANDERER ({n_candidates} candidates through different domain lenses) ---")
        chosen_domains = DOMAINS[:n_candidates]
        candidates = []
        for i, domain in enumerate(chosen_domains):
            short_domain = domain.split(",")[0]
            print(f"\n  [{i+1}] Lens: {short_domain}")
            cand = wanderer(prompt, domain, i)
            candidates.append(cand)
            print(f"  {cand}")

        # 3. Critic
        print(f"\n--- CRITIC scoring each candidate ---")
        scored = []
        for i, cand in enumerate(candidates):
            scores = critic(prompt, cand)
            scored.append((i, cand, scores))
            short_domain = chosen_domains[i].split(",")[0]
            print(f"  [{i+1}] {short_domain:50s}  "
                  f"N={scores['novelty']} U={scores['usefulness']} "
                  f"C={scores['coherence']} -> combined={scores['combined']:.1f}")

        # 4. Top 3
        scored.sort(key=lambda x: x[2]["combined"], reverse=True)
        top_3 = scored[:3]
        print(f"\n--- TOP 3 SELECTED ---")
        for i, cand, scores in top_3:
            short_domain = chosen_domains[i].split(",")[0]
            print(f"  [#{i+1}] {short_domain}  combined={scores['combined']:.1f}")

        # 5. Synthesis
        print(f"\n--- FINAL SYNTHESIS (combining top 3 framings) ---")
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
    run_phase4.remote()
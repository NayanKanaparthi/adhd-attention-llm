# Giving an LLM ADHD

> I tried to give a language model ADHD by reaching into its attention mechanism and loosening it on purpose. Three different ways of doing that. Three different ways it fell apart. And, anticlimactically, the thing that actually produced creative output turned out to have nothing to do with attention at all.

This is a lab notebook, not a library. There's nothing to `pip install`. It's the write-up of a hunch I chased across seven experiments, what broke, and what I think it means.

---

## The hunch

The ADHD literature has this recurring, slightly romantic idea: the same loose attentional control that makes linear, convergent tasks hard (sit still, follow steps, don't get distracted) is the thing that helps with divergent, associative, creative tasks. Jumpy attention is bad at long division and occasionally great at connecting things nobody asked you to connect. The brain-network version of the story is the default mode network wandering off-leash while the task-positive network is supposed to be in charge.

A transformer's attention is, very literally, a learned filter over context. Each token decides how much to "attend" to every previous token. That's a softmax — a probability distribution that says *where to look*.

So the hunch wrote itself: **what if I mess with that filter the way ADHD messes with attentional control? Loosen it. Make the model look in places it normally wouldn't. Would it get more creative on open-ended prompts while falling apart on logic and math?**

If "loose attention → creative wandering" is real in brains, maybe a crude computational analog shows *some* signal in a model.

That was the bet. Here's what happened.

---

## The setup

- **Model:** Qwen 2.5 — 3B for the early cheap probes, 7B once things got serious.
- **Infra:** [Modal](https://modal.com) for serverless GPU (A10G for 3B, A100 for 7B), Hugging Face Transformers on PyTorch, weights cached in a Modal volume so I only download once.
- **The one non-negotiable detail:** every model is loaded with `attn_implementation="eager"`. Eager attention is the slow, pure-PyTorch path — which means I can monkey-patch `Qwen2Attention.forward` and rewrite what happens between the query·key product and the value aggregation. SDPA and FlashAttention use fused CUDA kernels you can't reach into. Slow but surgical.

**The honesty check.** Every intervention reimplements the entire attention forward pass and inserts a knob. Before any experiment runs, the script generates text with the *stock* forward, then with the *patched* forward at knob=0, and asserts the two are byte-identical. If they aren't, it aborts. This is how I know any effect I see is the intervention and not a bug I introduced rewriting attention. Patch is a clean superset of stock, or the run doesn't happen.

---

## The journey

I went in phases, each one cheaper to reason about than the last was expensive to be wrong about.

### Phase 1 — Can I even do this?
Plumbing. Load Qwen, install a no-op patch on the attention forward, confirm the model still produces identical output. Pure "does the monkey-patch mechanism work before I put real logic in it." It did.

### Phase 2 — The cheapest possible probe (don't touch the model)
Before perturbing anything internal, test the hypothesis at the prompt level. Ask the model for a chain-of-thought in a strict `Step 1 / Step 2 / … / Answer` format, then **reorder its own reasoning steps** — original, reversed, shuffled — and ask it to answer from the scrambled steps. The idea: if non-linear order of thought breaks logic but creative prompts shrug it off, that's first blood for the hypothesis. (v2 tightened the prompt so each step carried real computation instead of vacuous narration like "calculate the cost," and isolated the steps from the problem statement.) This is the black-box version. No attention surgery yet.

### Phase 3 — The real intervention: attention dropout
Now the white-box test. Right after softmax produces the attention weights, randomly **zero out a fraction of them**, then renormalize so each row still sums to 1 (so I'm changing *where* the model attends, not *how much* in total). Sweep the dropout rate: 0%, 10%, 30%, 60%, on a single prompt.

The result was genuinely encouraging, and it's where the hunch felt alive: **10% was a sweet spot.** Coherent, but meaningfully *different* — the boring "use a paperclip as a wire organizer" answer became a "use it as an improvised fishing hook" answer. **30% mostly degenerated. 60% was word salad.** A clean little degradation curve. (Caveat that matters later: this was n=1, on the 3B model, over short generations.)

### Phase 4 — Orchestration that worked (the plot twist hides here)
Switch to 7B and build a loop, no attention modification at all:
- **Wanderer** generates several candidate ideas, each forced through a deliberately weird domain lens — *think about this like biology, like MMORPG design, like religious ritual, like jazz, like military strategy.*
- **Critic** scores each on novelty / usefulness / coherence.
- **Synthesis** fuses the top candidates into one answer.

And it *worked*. The outputs were diverse and often genuinely interesting. But look closely at why: **I hand-fed the diversity.** The interesting framings came from me hardcoding "think like jazz." The model just executed the lenses I handed it.

### Phase 5 — The test that called the bluff
So: was Phase 4's creativity the model's, or mine? Rip out the hardcoded domains. Replace them with a bare "find an unusual angle." Make the **attention dropout from Phase 3 the only source of wandering.** If loose attention is doing real associative work, the candidates should be just as varied as Phase 4's were.

They weren't. At 7B over 200-token generations, the same 10% dropout that was the sweet spot on 3B short outputs caused **catastrophic degeneracy — roughly a third of the candidates collapsed** into broken text. The intervention that worked small didn't survive scale and length. And without the scaffolding, what didn't degenerate mostly converged on the obvious answer. Phase 4, it turns out, was **creativity theater**: the hardcoded frames did the creative work; the loose attention did not.

### Phase 6 — A smoother knob: temperature flattening
Maybe per-weight dropout is just too violent — punching random binary holes in the attention is bound to destroy signal. So try the gentler, "primary" version of loose attention from the framework: **before softmax, divide the logits by a temperature T > 1.** This flattens the distribution — more peripheral tokens contribute — but *nothing is zeroed*. Smooth, not punched.

It was **stable** — no degeneracy this time. But the candidates **converged.** Flattening everything uniformly didn't make the model wander to interesting places; it made it mushy and same-y. Loosening attention globally is not the same as redirecting it.

### Phase 7 — A surgical knob: head dropout
The third intervention. Instead of perturbing individual weights or the whole distribution, **zero out entire random attention heads** each forward pass — a fresh mask per token, each layer rolling its own. This is the lightweight cousin of "head specialization": force the model to route around whichever heads happen to be suppressed, without needing to know what each head does. The third swing at making attention itself produce the wandering.

---

## What I found

The repo description says it in one breath: **three interventions, three failure modes, one orchestration pattern that actually works.**

- **Per-weight dropout (Phase 5)** breaks at scale — fine on a small model and short outputs, catastrophic on a real one over real lengths.
- **Temperature flattening (Phase 6)** is stable but blunt — it converges instead of diverging. Uniform looseness ≠ creative redirection.
- **Head dropout (Phase 7)** — the third attention-level swing — likewise didn't deliver self-directed creative wandering.
- **The orchestration loop (Phase 4)** — Wanderer/Critic/Synthesis with explicit, diverse framings — is the only thing that reliably produced novel *and* useful output. And it works at the level of *process and prompting*, not by touching the model's insides at all.

The deflating, useful conclusion: **the creativity lived in the structure I built around the model, not in the attention substrate I tried to perturb.**

---

## What it means (to me, at least)

**The brain analogy is seductive and leaky.** "Loose attention helps creativity" is a statement about human attentional *control* — an executive process. A transformer's attention weights are not that knob. Randomly perturbing the softmax is far more like adding noise to a signal than like the default mode network going for a walk. Noise mostly destroys; it doesn't reliably diversify.

**Diversity in LLM ideation seems to come from where you point the model, not from jitter in the forward pass.** The thing that moved the needle was giving it genuinely different *places to look* (domains, framings, a search structure) — not loosening *how* it looked. That's a slightly humbling result for anyone (me) who wanted the elegant neuro-inspired mechanism to be the answer.

**Negative results are still results.** This rules out the *simplest* version of attention-as-ADHD: random per-weight dropout, uniform temperature flattening, random head dropout. It does **not** rule out smarter versions — targeted ablation of specific heads, a learned "looseness" adapter, novelty-pressure during decoding, or retrieval-driven wandering. The crude knobs failed; the targeted ones are untested.

---

## Big honest caveats

Read the findings through these:

- **The critic is a broken measuring instrument.** Its novelty scores pin near 9 for almost everything, regardless of how novel an idea actually is — it has no calibration on that axis. It's also the *same model* grading its own work, with a known bias toward its longer, more structured synthesis. So the numeric deltas in the output are mood lighting, not measurement. **Every real conclusion here is from eyeballing the text**, not from the scores.
- **n is tiny.** The Phase 3 sweet spot was a single example. Single seed end-to-end, three prompts, no aggregate statistics. This is hunch-chasing, not a paper.
- **Model size is a confound.** Phases 1–3 ran on 3B; Phases 4–7 on 7B. The clean little degradation curve from Phase 3 and the degeneracy in Phase 5 are not strictly comparable across that line.
- **One intervention at a time.** I never tried stacking the orchestration *and* an attention intervention, which is the obvious next thing.

---

## The phases at a glance

| Phase | File | What it does | Outcome |
|------:|------|--------------|---------|
| 1 | `phase1.py` | Modal + Qwen plumbing, no-op attention patch | Mechanism works |
| 2 | `phase2.py`, `phase2_v2.py` | Permute chain-of-thought steps (prompt-level, black box) | Cheap probe of the hypothesis |
| 3 | `phase3.py` | Post-softmax attention dropout, sweep 0/10/30/60% (3B) | 10% sweet spot, degrades after |
| 4 | `phase4.py` | Wanderer/Critic/Synthesis with **hardcoded domain lenses** (7B, stock attention) | **Works** — but diversity is hand-fed |
| 5 | `phase5.py` | Same loop, domains removed, **attention dropout** as the only wandering | Degenerates at scale (~⅓ collapse) |
| 6 | `phase6.py` | Same loop, **pre-softmax temperature flattening** | Stable but converges |
| 7 | `phase7.py` | Same loop, **random head dropout** | Third attention swing, same disappointment |

---

## Poke at it yourself

Each phase is a standalone Modal script:

```bash
modal run phaseN.py
```

You'll need a Modal account and the Hugging Face weights (they cache into a Modal volume named `hf-cache` on first run). Every attention-modifying phase self-verifies its patch against stock attention before doing anything, and aborts on mismatch — so if it runs, the intervention is the only thing that changed.

The attention-intervention knob is a module-level global (`DROPOUT_RATE`, `ATTENTION_TEMPERATURE`, `HEAD_DROPOUT_RATE`) flipped between generations: **loose** during the Wanderer stage, **tight** (stock) for the Critic and Synthesis. One model, two cognitive modes, toggled with a variable.

---

*Built with Qwen 2.5, Modal, and a hypothesis that didn't survive contact with a 7B model. Worth it anyway.*

# 03 — Post-Training Methods

*v4 post-training summaries, generated solely from the full-text files in [`../texts/`](../texts/) (the existing `summaries/`, `summaries-v2/`, and `summaries-v3/` folders were **not** used as source material). Papers are sorted by arXiv-ID year (2022→2026), then by corpus order within each year. Each entry synthesizes **one lens only** for that paper; the chronological cross-lens narrative — older trends, how the field evolved, and why the newer methods were proposed — lives in [`EVOLUTION_OVERVIEW.md`](EVOLUTION_OVERVIEW.md).*

**Lens:** the core post-training method/algorithm — preference optimization, RL algorithms (PPO/GRPO/DAPO/…), process reward models, distillation, self-play, agent loops, and infra: the objective, key tricks, and the problem solved.

**Coverage:** 99 of the 99 papers contribute substantive content on this lens; papers for which this lens was not a focus are omitted here and appear under their relevant topic files.

---

## 2022

### CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning
*2022 · code · `code_CodeRL_2207.01780.txt` · arXiv [2207.01780](https://arxiv.org/abs/2207.01780)*

CodeRL casts program synthesis as actor-critic RL: the pretrained LM is the policy, and a smaller Transformer critic is trained as a 4-class error predictor (CompileError, RuntimeError, FailedTest, PassedTest) supplying dense per-token feedback q̂_φ. Heuristic unit-test returns (−1.0/−0.6/−0.3/+1.0) are made relative to a greedy-decoding baseline to cut variance, yielding the policy-gradient update in Eq. (10). At inference, Critic Sampling runs two strategies: refining reseeds from critic-scored subsequences of example-test-passing programs, and repairing routes top-M critic-selected failures plus compiler-error types to a seq2seq repair model trained on synthetic buggy/correct pairs — one round each, nucleus sampling at batch N=200.

- Actor = pretrained LM (policy); critic = 4-class error predictor giving per-token dense reward
- Heuristic returns −1.0/−0.6/−0.3/+1.0, variance-reduced against a greedy baseline
- Critic Sampling: refining (critic-scored subsequence seeds) + repairing (seq2seq repair model), one round each
- Targets neglect of unit-test signals and limits of NTP-only SFT

**Key results:** CodeRL+CodeT5-770M sets APPS SOTA at 2.69% pass@1, 6.81% pass@5, 20.98% pass@1000; zero-shot MBPP reaches 63.0% pass@80 and 81.8% pass@1000, beating GPT-137B (61.4%).

*Evolution:* Extends REINFORCE/actor-critic sequence generation and CodeT5 pretraining into code, anticipating execution-feedback and reward-model-driven post-training for code.

### STaR: Bootstrapping Reasoning With Reasoning
*2022 · data · `data_STaR_2203.14465.txt` · arXiv [2203.14465](https://arxiv.org/abs/2203.14465)*

STaR (Self-Taught Reasoner) bootstraps chain-of-thought rationale generation without a large human rationale set. The loop: few-shot prompt GPT-J (6B) to emit a rationale r then answer y for each question; keep only rationales where ŷ=y_true (indicator reward); fine-tune M on the filtered set; iterate. Rationalization feeds the correct answer as a hint for failed items and trains on the backward rationale as if unhinted, exposing the model to hard problems. The authors cast this as an approximate RL policy-gradient on a latent-variable model p(y|x)=Σ_r p(r|x)p(y|x,r) with reward 1(ŷ=y): greedy decoding reduces variance (biased exploration), multiple gradient steps mimic PG, and the answer-match filter is the reward. No separate value function or verifier (unlike Expert Iteration/GPT-f); GPT-2 fails since few-shot accuracy must exceed chance to bootstrap.

- Iterative self-bootstrap: prompt → filter by answer correctness → SFT → repeat
- Rationalization: hint-driven backward rationales for failed items, trained as if unhinted
- Approximate PG on latent rationales; greedy decoding = biased, low-variance exploration
- No verifier/value function; needs base few-shot accuracy above chance (GPT-J 6B, not GPT-2)

**Key results:** GPT-J (6B) + STaR with rationalization reaches 72.5% on CommonsenseQA dev, matching a 30× larger GPT-3 direct-finetuned (73.0%) and beating GPT-J direct-finetuning (60.0%) by +12.5%; arithmetic reaches 89.5% after 16 iterations.

*Evolution:* Turns few-shot CoT into an iterative self-bootstrapping SFT loop using answer-correctness as reward, helping launch the self-improving-reasoning line (STaR*, Quiet-STaR, V-STaR/RFT, ReST).

### SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions
*2022 · data · `data_SelfInstruct_2212.10560.txt` · arXiv [2212.10560](https://arxiv.org/abs/2212.10560)*

SELF-INSTRUCT is a nearly annotation-free instruction-tuning framework using the LM's own generations in an iterative bootstrapping loop over a task pool: seed tasks → generate new instructions (in-context prompted) → classify classification/non-classification → generate input-output instances → filter → add survivors → repeat. The key trick is output-first instance generation for classification tasks: emit class labels first, then condition input generation on each label, countering the bias where input-first over-produces one class (e.g., mostly grammatical inputs for grammar-error detection). Filtering combines ROUGE-L<0.7 de-duplication, keyword exclusion, and heuristic validity checks. It is framed as self-distillation where source and target are the same model and the distilled content is instruction tasks; training itself is standard supervised SFT.

- Bootstrapping loop over a growing task pool with in-context-prompted instruction generation
- Output-first instance generation for classification tasks counteracts label skew
- ROUGE-L<0.7 de-dup + keyword/validity filtering; standard SFT training
- Targets limited diversity/creativity of human-written instruction data (PROMPTSOURCE/SUPERNI)

**Key results:** GPT3_SELF-INST (GPT3 175B finetuned on 52K self-generated instructions) improves over vanilla GPT3 by +33.1 absolute ROUGE-L on SUPERNI (39.9 vs 6.8), nearly matching InstructGPT001 (40.8).

*Evolution:* Pioneers self-bootstrapped synthetic instruction data, directly enabling later open self-instruct-style models (Alpaca, Baize) and motivating distillation/reward-based refinement of synthetic instruction corpora.

## 2023

### AgentTuning: Enabling Generalized Agent Abilities for LLMs
*2023 · code · `code_AgentTuning_2310.12823.txt` · arXiv [2310.12823](https://arxiv.org/abs/2310.12823)*

AgentTuning is a hybrid instruction-tuning method that activates latent agent abilities in already-aligned LLMs without harming general skills, via a weighted log-likelihood J(θ)=η·E_agent[log π_θ(y|x)] + (1−η)·E_general[log π_θ(y|x)]. The agent data consists of ReAct-style trajectories where each step pairs a thought with an action, so the model learns the reasoning leading to tool/environment actions rather than just final answers. The central finding is that Llama-2-chat already has agent capability but misfires on elementary errors (invalid actions, repetition, refusals); AgentTuning largely removes these, suggesting activation rather than overfitting. It is framed as the first end-to-end attempt to instruction-tune LLMs on interaction trajectories spanning multiple agent tasks, producing the open AgentLM-7B/13B/70B.

- Weighted SFT objective combining agent-trajectory and general-instruction data (η mix)
- ReAct-style thought+action trajectories teach reasoning-to-action, not just final answers
- Activates latent agent skill in Llama-2-chat; removes elementary action errors (activation, not overfit)
- First multi-task agent-trajectory SFT; outputs open AgentLM-7B/13B/70B

**Key results:** AgentLM-70B achieves held-out overall 1.40 (+176% over Llama-2-70B), roughly matching GPT-3.5 (1.49); general ability is preserved (overall general 0.96 vs 0.95).

*Evolution:* One of the earliest multi-task agent-trajectory SFT works, anticipating later agentic post-training and tool-use RL that scales trajectory data and moves from SFT toward reinforcement over environment rewards.

### RLTF: Reinforcement Learning from Unit Test Feedback
*2023 · code · `code_RLTF_2307.04349.txt` · arXiv [2307.04349](https://arxiv.org/abs/2307.04349)*

RLTF is an online RL framework for code LLMs using multi-granularity unit-test feedback. Two weight-shared LLMs run in parallel: one generates programs and interacts with the compiler to fill an online buffer; the other computes the loss and updates weights. The RL loss L_rl=−R(Ŵ)log P(Ŵ|D,θ), with start/end positions (S,E) of the penalized span chosen per feedback type. Three granularities combine: coarse (whole-episode ±1.0/−0.3/−0.6/−1.0); fine-grained, parsing compiler error types into U_line (penalize the offending line), U_global (all code for logic errors), U_ignore, with adaptive weight α=(E−S)/T; and adaptive reward −0.3+1.3·N_pass/(N_pass+N_fail). Coarse/adaptive use a dynamic historical-best baseline. Total loss L_sl+L_coarse+L_fine+L_adaptive. It targets offline-RL distribution shift and coarse feedback's inability to locate errors.

- Online RL: two weight-shared LLMs, one generating/compiling, one updating from an online buffer
- Three granularities: coarse episode reward, fine-grained compiler-error line targeting (U_line/U_global/U_ignore), adaptive pass-rate reward
- Adaptive weight α=(E−S)/T matches magnitudes; dynamic historical-best baseline
- Solves offline-RL distribution shift and coarse feedback's poor error localization

**Key results:** CodeT5-770M + RLTF on APPS sets SOTA among CodeT5-based RL methods (pass@1000 19.92%), beating CodeRL and PPOCoder; on MBPP zero-shot pass@1 71.3 vs CodeRL 68.1; fine-grained feedback contributes the largest single gain.

*Evolution:* Moves code-LLM RL from offline coarse-reward to an online loop with compiler-parsed line-level reward shaping — a 2023 precursor to later execution-feedback/RLVR-style code training.

### What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
*2023 · data · `data_DEITA_2312.15685.txt` · arXiv [2312.15685](https://arxiv.org/abs/2312.15685)*

The core method is Score-First, Diversity-Aware data selection (Algorithm 1): compute a combined evol score s=c·q (complexity×quality, summed over dialogue turns), sort the pool by s, then iteratively add a sample to the selected set S only if its nearest-neighbor embedding distance to S exceeds threshold τ=0.9 (discarding redundant samples) until budget m is reached — coupling score-based ranking with embedding-distance de-duplication. The complexity and quality scorers are distilled from ChatGPT rank-and-score judgments over evolution-generated graded variants, yielding finer scores than direct single-sample scoring (which collapses scores). The problem solved: automatically curating a small but effective SFT set so a model matches SOTA aligned models trained on 10× or more data.

- Score-First, Diversity-Aware selection: evol score s=c·q, sort, add if NN-embedding distance > τ=0.9
- Complexity/quality scorers distilled from ChatGPT rank-and-score over evolved graded variants
- Coupling score ranking with embedding-distance de-duplication
- Auto-curates a small SFT set matching models trained on 10×+ data

**Key results:** DEITA-Mistral-7B-6K+DPO reaches 7.55 MT-Bench and 90.06% AlpacaEval, comparable to Zephyr-beta trained on ~30× more data; 3K DEITA-selected samples match full 300K training (100× reduction).

*Evolution:* Formalizes automatic data selection across complexity, quality, and diversity for SFT, anticipating the data-centric turn in post-training where curated SFT/DPO data rather than raw scale drives alignment.

### Direct Preference Optimization: Your Language Model is Secretly a Reward Model
*2023 · data · `data_DPO_2305.18290.txt` · arXiv [2305.18290](https://arxiv.org/abs/2305.18290)*

Direct Preference Optimization replaces the reward-model-plus-RL stages of RLHF with one classification loss. Starting from the KL-constrained reward objective max E[r(x,y)]−β·KL(π‖π_ref), the optimal policy has closed form π*(y|x)∝π_ref(y|x)exp(r(x,y)/β); rearranging gives r(x,y)=β·log(π*(y|x)/π_ref(y|x))+β·log Z(x). Under the Bradley-Terry (or Plackett-Luce) preference model the partition function cancels in reward differences, so preference probability depends only on the policy. The DPO loss L=−E[log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)))] is a BCE increasing y_w and decreasing y_l likelihood, with an implicit per-example reward-based weight that prevents the degeneration a naive ratio objective suffers. No LM sampling, no explicit reward model, no RL loop; Theorem 1 shows the reparameterization loses no reward-class generality.

- Replaces RM+RL with a single BCE loss on static preference pairs
- Closed-form reparameterization: r=β·log(π*/π_ref)+β·log Z; Z cancels under BT/PL
- Implicit per-example reward-based weight prevents naive-ratio degeneration
- Theorem 1: no loss of reward-class generality; no on-policy sampling needed

**Key results:** TL;DR summarization DPO ~61% GPT-4 win rate vs reference, beating PPO's 57%; IMDb sentiment DPO achieves the best reward-KL frontier, dominating PPO and oracle PPO-GT; humans prefer DPO over PPO 58%.

*Evolution:* Replaces the reward-model-plus-PPO recipe with a single cross-entropy loss, enabling a family of simpler preference-optimization methods (IPO, KTO, SLiC) and becoming a default alignment stage.

### A General Theoretical Paradigm to Understand Learning from Human Preferences
*2023 · data · `data_IPO_2310.12036.txt` · arXiv [2310.12036](https://arxiv.org/abs/2310.12036)*

The core contribution is ΨPO, a general KL-regularized objective max_π E[Ψ(p*(y≻y'|x))]−τ·D_KL(π‖π_ref) for an arbitrary non-decreasing Ψ. The authors prove RLHF and DPO are special cases recovered when Ψ(q)=log(q/(1−q)) under Bradley-Terry. They show DPO overfits when empirical preferences are {0,1}-valued: the unbounded logit drives the optimal policy to deterministic regardless of τ, neutralizing KL regularization (RLHF's underfit reward model implicitly regularizes, which DPO loses). The fix is IPO: set Ψ to the identity, yielding max p*_ρ(π≻μ)−τ·D_KL(π‖π_ref), bounded and bypassing BT. They derive a convex offline sampled loss L(π)=E[(h_π(y_w,y_l)−τ⁻¹/2)²] with h_π=log(π(y_w)π_ref(y_l)/(π(y_l)π_ref(y_w))), regressing the log-ratio gap to τ⁻¹/2, and prove a unique global optimum (Theorem 2).

- ΨPO unifies RLHF and DPO (special cases under Ψ=log(q/(1−q)) and Bradley-Terry)
- Diagnoses DPO overfitting under deterministic {0,1} preferences: unbounded logit kills KL
- IPO: Ψ=identity, bounded, BT-free; convex loss regresses log-ratio gap to τ⁻¹/2
- Theorem 2 guarantees a unique global optimum

**Key results:** Qualitative contributions only; in 2-action asymptotic sets with p*=1 and uniform π_ref, IPO yields π*(y1)=σ(0.5τ⁻¹) (τ-controlled) while DPO yields π*(y1)=1 for all τ, and on 3-action sets DPO collapses to deterministic while IPO stays τ-controlled near π_ref.

*Evolution:* Unifies RLHF and DPO under ΨPO and diagnoses DPO's overfitting under deterministic preferences, motivating later BT-free direct alignment variants (e.g., KTO and IPO follow-ups).

### LIMA: Less Is More for Alignment
*2023 · data · `data_LIMA_2305.11206.txt` · arXiv [2305.11206](https://arxiv.org/abs/2305.11206)*

The core contribution is the Superficial Alignment Hypothesis: knowledge and capabilities are learned during pretraining, and alignment merely teaches which output subdistribution/format to use — so a tiny high-quality SFT set suffices. The method is vanilla supervised next-token loss on 1,000 examples with no RL or preference objective; speaker roles are delimited by a special end-of-turn (EOT) token rather than reusing EOS. Training: AdamW (β1=0.9, β2=0.95, weight decay 0.1), 15 epochs, lr 1e-5 linearly decayed to 1e-6, no warmup, batch 32 (64 for 7B/30B), sequences trimmed at 2,048 tokens. Residual dropout increases linearly from p=0.0 at the bottom layer to p=0.3 at the last (0.2 for smaller models), following Ouyang et al. Because validation perplexity anticorrelates with generation quality, checkpoints are manually picked between epochs 5–10 on a 50-example dev set.

- Superficial Alignment Hypothesis: 1,000 high-quality SFT examples suffice, no RL
- Vanilla next-token loss; EOT token delimits speaker roles (not EOS)
- AdamW, 15 epochs, lr 1e-5→1e-6 linear decay; residual dropout 0.0→0.3 across layers
- Manual checkpoint picking (epochs 5–10) since val perplexity anticorrelates with quality

**Key results:** LIMA (65B LLaMa, SFT on 1,000 examples, no RLHF) is equal-or-preferred to GPT-4 43%, Claude 46%, Bard 58%, DaVinci003 65%, and beats Alpaca 65B (52K examples); 50% of outputs rated excellent.

*Evolution:* Crystallizes the data-quality-over-quantity movement, arguing alignment is "superficial" (mostly style) so a curated 1,000-example SFT set rivals RLHF products, informing debates on how much alignment truly requires RLHF.

### Magicoder: Empowering Code Generation with OSS-Instruct
*2023 · data · `data_Magicoder-OSS-Instruct_2312.02120.txt` · arXiv [2312.02120](https://arxiv.org/abs/2312.02120)*

OSS-Instruct is a data-generation method (not a new training algorithm) mitigating the system bias of LLM-generated synthetic instructions. Unlike Self-Instruct (21 fixed seed tasks, identical template) and Evol-Instruct (5 heuristics evolving Code Alpaca), OSS-Instruct draws on the effectively infinite supply of real open-source code snippets as per-sample inspiration, yielding diverse, realistic, controllable problem–solution pairs in one shot. Ablations justify the design: directly finetuning on 75K semantically relevant comment–function pairs mined from the same corpus actually degrades the base model (HumanEval+ stays 34.1, MultiPL-E drops to 24.1), whereas OSS-Instruct yields large gains — showing data factuality, not format, drives improvement. A weaker-teacher experiment (Mixtral-8x7B generating 20K samples) produces a 7B model beating the Mixtral teacher on HumanEval+/MBPP+, indicating OSS-Instruct is not mere distillation but activates pretrained knowledge via seed-snippet context.

- OSS-Instruct: real open-source code snippets as per-sample inspiration for synthetic instructions
- One-shot generation; no fixed seed tasks or fixed evolution heuristics
- Ablation: mining 75K real comment–function pairs degrades the model; snippet-inspired data helps
- Weaker-teacher experiment shows it activates base knowledge, not mere distillation

**Key results:** MagicoderS-CL-7B surpasses ChatGPT on HumanEval+ pass@1 (66.5 vs 65.9) with only 7B parameters; MagicoderS-DS-6.7B beats DeepSeek-Coder-Instruct-6.7B on HumanEval(+)/MBPP(+) using 8× fewer finetuning tokens.

*Evolution:* Extends Self-Instruct and reacts against Evol-Instruct/WizardCoder's narrow seed-task sets, grounding synthetic instruction generation in real open-source code and demonstrating real-code inspiration plus orthogonal complexity evolution.

### Scaling Relationship on Learning Mathematical Reasoning with Large Language Models
*2023 · data · `data_RFT-rejection-sampling_2308.01825.txt` · arXiv [2308.01825](https://arxiv.org/abs/2308.01825)*

Rejection sampling Fine-Tuning (RFT) is a verifier-free alternative to STaR/CoRE/MCTS augmentation. Algorithm 1 (Reasoning Path Selection): given k sampled paths Rq for question q, keep the first path for each new equation-list key; for a duplicate key, replace the stored path if the candidate is more Levenshtein-dissimilar from already-selected paths, maximizing diversity. The objective is plain SFT loss on the augmented set D' (no reward model, no PPO). Key empirical insight: distinct-reasoning-path count, not raw sample count, drives RFT gains (k=100 dedup matches k=100 no-dedup in accuracy but trains faster), and doubling k gives sub-log-linear returns since no new questions are added. Aggregating paths from multiple SFT models (D'U13B, D'U33B) yields more diverse calculation orders/forms, further boosting accuracy; larger models (33B+) overfit training paths and contribute fewer unique paths.

- Verifier-free RFT: sample k CoT paths, dedup by equation-list key, maximize Levenshtein diversity
- Plain SFT loss on augmented set — no reward model, no PPO
- Distinct-path count (not raw samples) drives gains; doubling k is sub-log-linear
- Aggregating paths across SFT models boosts diversity; 33B+ models overfit and add fewer unique paths

**Key results:** LLaMA-7B RFT-U13B reaches 49.3% maj1@1 on GSM8K vs 35.9% SFT (+13.4); RFT helps weaker (higher pre-training-loss) base models most and adds nothing for 33B/65B/70B, which overfit training paths.

*Evolution:* Replaces trained verifiers/MCTS with simple rejection-sampling dedup of correct CoT paths, prefiguring the later rejection-sampling/RLAIF data-augmentation wave (e.g., LLaMA2 alignment) and correctness-reward RLVR for math.

### Statistical Rejection Sampling Improves Preference Optimization
*2023 · data · `data_RSO_2309.06657.txt` · arXiv [2309.06657](https://arxiv.org/abs/2309.06657)*

RSO (Statistical Rejection Sampling Optimization) addresses that DPO's MLE requires preference pairs drawn from the optimal policy π*, yet DPO uses pairs from mixed unknown policies and SLiC can only sample from the SFT policy. RSO uses Neal (2003) statistical rejection sampling with the SFT policy as proposal and a pointwise reward rψ from a pairwise T5-XXL reward model: accept an SFT sample y with probability exp((rψ(y)−r_max)/β), giving samples from π_{rψ}. β trades reward exploitation against SFT regularization: β→0 reduces to top-k-over-N (prone to reward hacking), β→∞ keeps the SFT policy. The paper unifies losses: DPO is logistic regression on preferences (sigmoid-norm), SLiC is essentially an SVM with hinge loss; it introduces hinge-norm (SVM counterpart of DPO) and decouples loss temperature γ from β. Theorem 1 gives the expected acceptance rate of the sampling-without-replacement algorithm.

- Neal rejection sampling: SFT policy as proposal, accept with prob exp((rψ(y)−r_max)/β) → samples from π_{rψ}
- β trades exploitation vs SFT regularization; β→0 = best-of-N (reward hacking), β→∞ = SFT
- Unifies DPO (logistic regression/sigmoid-norm) and SLiC (SVM/hinge); introduces hinge-norm
- Decouples loss temperature γ from β; Theorem 1 gives expected acceptance rate

**Key results:** RSO (T5-large, sigmoid-norm, rso-sample-rank) reaches 84.40% Gold Reward and 71.86% AutoSxS on Reddit TL;DR vs DPO 76.09/58.65; at T5-XXL scale RSO improves AutoSxS over DPO by +1.1% (TL;DR) and +33.1% (AnthropicHH); human raters choose RSO >2× more often than DPO.

*Evolution:* Bridges offline preference optimization and online RLHF via reward-guided statistical rejection sampling to approximate on-policy data, foreshadowing later iterative-DPO and best-of-N/on-policy preference methods.

### Enhancing Chat Language Models by Scaling High-quality Instructional Conversations
*2023 · data · `data_UltraChat_2305.14233.txt` · arXiv [2305.14233](https://arxiv.org/abs/2305.14233)*

The contribution is a synthetic data-generation pipeline rather than a novel post-training algorithm. Two ChatGPT Turbo APIs play user and assistant roles and are called iteratively to build multi-turn dialogues; the user model is prompted with explicit personas and reminders of the conversation's purpose to prevent the "role exchange" failure where it drifts into answering. Opening-line diversity is engineered via meta-information (topics, Wikidata entities), in-context expansion, and iterative prompting. The motivation is that directly ChatGPT-generated dialogues are shallow because they lack RLHF-style alignment; iterative user–AI simulation yields more informative turns. Training is conventional SFT with response-only cross-entropy loss over context-broken sequences. The method solves how to scale diverse, realistic instruction-tuning data without human annotation.

- Two ChatGPT Turbo APIs role-play user+assistant iteratively for multi-turn dialogues
- User model prompted with personas + purpose reminders to prevent role-exchange drift
- Opening-line diversity via topics/Wikidata entities, in-context expansion, iterative prompting
- Conventional response-only SFT; targets scaling diverse instruction data without humans

**Key results:** UltraLLaMA-13B scores 9.02 overall (ChatGPT-judged, 1–10) vs Vicuna-13B 8.96 and ChatGPT 9.12; pairwise win rate up to 85% over open-source baselines, 13% above Vicuna.

*Evolution:* Builds on the Self-Instruct/Alpaca/Vicuna distillation wave, reacting to small curated sets (e.g., LIMA) stalling below Vicuna; its large-scale synthetic multi-turn SFT corpus later seeded open alignment pipelines such as UltraFeedback.

### UltraFeedback: Boosting Language Models with Scaled AI Feedback
*2023 · data · `data_UltraFeedback_2310.01377.txt` · arXiv [2310.01377](https://arxiv.org/abs/2310.01377)*

UltraRM is a LLaMA2-13B reward model trained with a margin-aware binary ranking loss L=−log(σ(r(x,y_c)−r(x,y_r)−m(r))), where m(r) is the annotated reward gap (0 for ranking-only data, normalized to (0,1] to stabilize cross-dataset scale mismatches); trained 1 epoch, batch 512 pairs, lr 1e-5, cosine decay, 3% warmup. Three variants compare data mixes and fine-grained vs overall scores. Best-of-n samples n completions (temperature 1, top-p 1) and picks the highest-reward one, improving the policy without weight updates. RLAIF via PPO aligns UltraLM-13B into UltraLM-13B-PPO: 80 iterations, 512 samples/iter, mini-batch 64, lr 1e-6. UltraCM, a LLaMA2-13B critique model, is trained on 255,864 textual critiques (2 epochs, batch 256, lr 2e-5) to generate judge-and-suggest feedback. The key trick is GPT-4 fine-grained multi-aspect annotation with simultaneous scoring, which beats overall single-shot scoring.

- UltraRM: LLaMA2-13B RM, margin-aware ranking loss with annotated reward gap m(r) normalized to (0,1]
- Best-of-n (temp 1, top-p 1) picks highest-reward completion — no weight updates
- RLAIF-PPO: 80 iters, 512 samples/iter, mini-batch 64, lr 1e-6 → UltraLM-13B-PPO
- GPT-4 fine-grained multi-aspect annotation beats overall single-shot scoring; UltraCM critique model

**Key results:** UltraRM beats open-source reward models by >6.3% avg preference-prediction accuracy; UltraLM-13B-PPO attains the highest average win rate (69.7%), +16.8% over base and surpassing LLaMA2-70B-Chat; Best-of-16 lifts AlpacaEval vs text-davinci-003 from 76.53% to 91.54%.

*Evolution:* Extends RLAIF by scaling GPT-4 AI feedback into a large, diverse, fine-grained open preference dataset; its released dataset and reward model became a staple substrate for later 2023–2024 preference work (e.g., Zephyr/DPO-style alignment).

### WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions
*2023 · data · `data_WizardLM-EvolInstruct_2304.12244.txt` · arXiv [2304.12244](https://arxiv.org/abs/2304.12244)*

Evol-Instruct is an evolutionary data-generation algorithm with two components: an Instruction Evolver (an LLM prompted to rewrite or mutate instructions) and an Instruction Eliminator (filters failed evolutions). In-Depth Evolving makes prompts harder via five operations, each capped at adding only 10–20 words to keep difficulty gradual; In-Breadth Evolving generates a brand-new, rarer, same-domain instruction for diversity. After evolving, the same LLM generates responses at temperature 1, top-p 0.9, max 2048 tokens. Fine-tuning initializes from pretrained LLaMA-13B, uses Adam at LR 2e-5, batch size 4/GPU, DeepSpeed Zero-3 on 8 V100 GPUs for 140 hours over 3 epochs, with Vicuna's chat prompt format and greedy decoding at inference. The contribution is showing LLM-evolved, complexity-controlled instructions beat human- and Self-Instruct-created data for SFT.

- Evol-Instruct: Instruction Evolver (mutate/rewrite) + Instruction Eliminator (filter failures)
- In-Depth (5 ops, +10–20 words each) vs In-Breadth (new rare same-domain instruction)
- Responses at temp 1, top-p 0.9; SFT LLaMA-13B, Adam lr 2e-5, DeepSpeed Zero-3, 8×V100, 3 epochs
- Shows LLM-evolved complexity-controlled SFT data beats human/Self-Instruct data

**Key results:** WizardLM-13b beats Vicuna-13b and Alpaca-13b on average (58.96 vs 54.60 vs 43.44) with HumanEval 24.0 and GSM8k 37.15; WizardLM-70b reaches 71.33 avg (approaching ChatGPT-3.5's 76.15); quality rises monotonically with evolved-instruction complexity across rounds C0–C4.

*Evolution:* Reacts to the difficulty skew in human-created ShareGPT, anticipating the later synthetic-data-scaling and difficulty-aware-curriculum trend (e.g., WizardMath/Evol series) by showing LLM-evolved, complexity-controlled SFT data can surpass human and Self-Instruct data without RLHF.

### ZEPHYR: Direct Distillation of LM Alignment
*2023 · data · `data_Zephyr_2310.16944.txt` · arXiv [2310.16944](https://arxiv.org/abs/2310.16944)*

The contribution is distilled Direct Preference Optimization (dDPO): distill alignment from a teacher onto a small student using AI feedback, without PPO, rejection sampling, or human labels. Following Rafailov et al., the optimal reward is reparameterized through the policy, r*(x,y)=β·log(π*/π_dSFT)+β·log Z(x), giving a closed-form loss over static preference triples: maximize E[σ(β·log(π(y_w|x)/π_dSFT(y_w|x)) − β·log(π(y_l|x)/π_dSFT(y_l|x)))]. Training requires only forward probabilities from the frozen dSFT reference and the current model, then a backprop step — no on-policy sampling. AIF generation has four models respond per prompt and GPT-4 score them; the top score and a random lower one form each pair. Implementation uses TRL with DeepSpeed ZeRO3 and FlashAttention-2, full bfloat16 fine-tuning (no LoRA), replacing the reward-modeling-plus-PPO loop with a static-data objective.

- dDPO: distill alignment via DPO over GPT-4-scored AI feedback, no PPO/rejection/human labels
- Closed-form loss over static preference triples; only forward probs from frozen dSFT ref + current model
- AIF: 4 models respond per prompt, GPT-4 scores; top + random-lower form each pair
- TRL + DeepSpeed ZeRO3 + FlashAttention-2, full bfloat16 (no LoRA); no on-policy sampling

**Key results:** Zephyr-7B (Mistral-7B + dSFT on UltraChat + dDPO on UltraFeedback) reaches MT-Bench 7.34, surpassing Llama2-Chat-70B (6.86), with AlpacaEval 90.60% win rate; trained in 2–4 hours on 16 A100s with no human annotation.

*Evolution:* Popularizes the distilled SFT-then-DPO-on-AIF pipeline, replacing the costly human-feedback-plus-PPO stage (as in Llama2-Chat) with DPO over GPT-4-scored AI feedback and showing a 7B model can match 70B RLHF chat models.

### Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations
*2023 · rl · `rl_Math-Shepherd_2312.08935.txt` · arXiv [2312.08935](https://arxiv.org/abs/2312.08935)*

Math-Shepherd is a process reward model (PRM) scoring every reasoning step, trained without human labels via MCTS-inspired "completion + estimation": a completer decodes N continuations from a step and the step's quality is its potential to reach the golden answer (HE: any-correct; SE: fraction-correct). The PRM is trained with per-step binary cross-entropy (two special tokens for "has/no potential", no architecture change); the solution score is the minimum step score, optionally combined with self-consistency grouping. For RL, the PRM drives step-by-step PPO, delivering reward at the end of each reasoning step rather than only at sequence end (as in ORM-PPO). PPO uses KL coefficient 0.04, cosine schedule (min lr 1e-8), lr 4e-7 (LLaMA2-7B)/1e-7 (Mistral-7B), max seq len 512, 3D parallelism. This solves the cost/scalability problem of human-annotated PRMs while keeping PRM's finer-grained feedback over ORM.

- PRM scoring every step, auto-labeled via MCTS-style completion (HE any-correct / SE fraction-correct)
- Per-step BCE with two special tokens (has/no potential); solution score = min step score
- Step-by-step PPO: reward at end of each step, not just sequence end (vs ORM-PPO)
- PPO: KL 0.04, cosine to 1e-8, lr 4e-7/1e-7, 3D parallelism; solves human-PRM cost/scalability

**Key results:** Mistral-7B + step-by-step PPO with Math-Shepherd: GSM8K 77.9%→84.1%, MATH 28.6%→33.0% (greedy); with verification, 89.1% GSM8K and 43.5% MATH; DeepSeek-67B-MetaMath + Math-Shepherd verification reaches 93.3% GSM8K and 48.1% MATH — SOTA for open-source models without tools; the auto-annotated PRM dataset (~440k solutions, ~4× PRM800K) yields a verifier surpassing human-annotated PRM800K on MATH.

*Evolution:* Makes process supervision scalable by automating step labels via MCTS-style completion, then plugs that PRM into step-by-step PPO, anticipating the 2024 wave of RLVR/process-reward work and iterative RLHF where verifiers are bootstrapped from the generator itself.

## 2024

### AgentGym: Evolving Large Language Model-based Agents across Diverse Environments
*2024 · code · `code_AgentGym_2406.04151.txt` · arXiv [2406.04151](https://arxiv.org/abs/2406.04151)*

AGENTEVOL reframes RL as probabilistic inference (Dayan-Hinton/MAPPO), deriving a variational lower bound on log P(O=1) that decouples data collection from policy update. The frozen previous policy explores trajectories across multiple agent environments, which emit binary rewards (r<1 mapped to 0); the learning step then maximizes a reward-weighted log-likelihood E[r(e,u,τ) log πθ(τ|e,u)], up-weighting high-reward trajectories. This is essentially ReST/STaR-style self-improvement generalized across environments rather than a single task. DPO on failed trajectories was tried but underperformed reward-weighted SFT in the multi-task setting.

- Iterative self-improvement loop across heterogeneous agent environments, not one task
- Exploration (frozen prev-policy sampling) and learning (reward-weighted NLL) decoupled
- Binary reward r∈{0,1} via environment; reward-weighted SFT beats DPO-on-failures
- 4 iterations, ~20h on 8×A100-80GB, K=1 sample per instruction

**Key results:** AGENTEVOL on Llama-2-Chat-7B beats the behavioral-cloning upper bound BClarge on WebShop (76.5 vs 73.5), ALFWorld (88.0 vs 83.0), BabyAI (82.7 vs 74.19), and TextCraft (64.0 vs 60.0), and matches or surpasses GPT-4-Turbo on several environments.

*Evolution:* Building on the RL-as-inference tradition (Dayan-Hinton, MAPPO) and single-task self-improvement lines (ReST, STaR, Trial-and-Error), AgentGym is an early 2024 attempt to push LLM agent self-evolution beyond isolated environments toward multi-environment generalists, complementing behavioral-cloning agent-tuning works like AgentTuning/AgentOhana.

### AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials
*2024 · code · `code_AgentTrek_2412.09605.txt` · arXiv [2412.09605](https://arxiv.org/abs/2412.09605)*

The method is "guided replay with web tutorials": instead of a bare goal, the GUI agent receives a standardized tutorial (target URL + step-by-step instructions) and executes it in live BrowserGym environments, emitting an inner-thought reasoning chain before each Playwright action. Tutorial guidance lifts replay success 15.78%→52% (400-task controlled study) by supplying direct URLs, human-style plans, and menu-navigation help. Quality control uses a GPT-4o VLM evaluator scoring task adherence and component completion at trajectory, step, and earliest-failure levels. Vision agents map Playwright actions to pyautogui pixel commands; text agents use AXTree observations. Qwen2-VL's NaViT encoder handles dynamic-resolution screenshots.

- Tutorial-guided replay (+23 pts over goal-only) is the central empirical trick
- VLM-evaluator (GPT-4o) for trajectory, step, and earliest-failure quality gating
- Playwright→pyautogui mapping for vision; AXTree with bid IDs for text agents
- NaViT dynamic-resolution screenshots (720p in ~1,200 tokens vs ~4,000 HTML)

**Key results:** Qwen2.5-32B-Instruct fine-tuned on AgentTrek scores 22.40 success rate on WebArena, surpassing GPT-4 (14.41). Qwen2-VL-7B w/ AgentTrek reaches 67.4 average on ScreenSpot Web grounding vs 30.7 baseline.

*Evolution:* AgentTrek (2024, ICLR 2025) builds on the synthetic-trajectory trend for GUI agents (BAGEL, NNetNav, Synatra) and reacts against the cost and scalability limits of human-annotated datasets like Mind2Web, WebArena, and WebLINX.

### Agentless: Demystifying LLM-based Software Engineering Agents
*2024 · code · `code_Agentless_2407.01489.txt` · arXiv [2407.01489](https://arxiv.org/abs/2407.01489)*

Agentless resolves repo-level GitHub issues via a three-phase prompting pipeline with no autonomous agent loop. Phase 1 hierarchical localization builds a tree-style repo structure, prompts GPT-4o for top-N suspicious files, adds embedding retrieval (text-embedding-3-small, chunk 512), keeps top-3 files, then localizes to classes/functions via signature-only skeletons and to edit locations (4 samples). Phase 2 repair samples 10 patches per location set (1 greedy + 9 at T=0.8) in Aider Search/Replace diff format (40 candidates/bug). Phase 3 validation samples 40 reproduction tests, keeps "Issue reproduced" ones, runs LLM-filtered regression tests, requires "Issue resolved," then majority-votes over AST-normalized patches. Default greedy decoding with gpt-4o-2024-05-13.

- Three-phase prompt pipeline: localize → repair → validate, no agent autonomy
- Hierarchical localization (repo tree → files → skeletons → edit locations)
- 40 candidate patches/bug via Aider S/R diffs; AST-normalized majority vote
- Reproduction-test validation with regression filtering; no weight updates

**Key results:** Agentless (GPT-4o) resolves 96/300 (32.00%) on SWE-bench Lite at $0.70, the highest among open-source approaches; 84/249 (33.73%) on SWE-bench Lite-S; and 194/500 (38.80%) on SWE-bench Verified, best among GPT-4o-based tools.

*Evolution:* Building on LLM-based automated program repair and IR-based fault localization, Agentless reacts against the 2024 surge of complex autonomous coding agents (Devin, SWE-agent, AutoCodeRover), showing a simple prompting pipeline can match or beat them at a fraction of the cost.

### Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions
*2024 · code · `code_Marco-o1_2411.14405.txt` · arXiv [2411.14405](https://arxiv.org/abs/2411.14405)*

Marco-o1 couples CoT fine-tuning with inference-time MCTS and a reasoning-action strategy. Each MCTS node is a reasoning state; actions are LLM-generated full steps or 32/64-token mini-steps; rollouts run to terminal and are scored by a confidence-based reward. Per-token confidence c_i is the softmax of that token's log-prob against the top-5 alternatives; rollout reward v is the mean c_i over all tokens. Action granularity trades breadth for resolution. A reflection mechanism appends "Wait! Maybe I made some mistakes..." to trigger self-critique, fixing ~half of initially-incorrect hard problems. It targets open-ended tasks where rewards are hard to quantify, using model confidence in lieu of an external reward model.

- CoT SFT + AlphaZero-style MCTS over reasoning states with confidence reward
- Confidence = softmax of token log-prob vs top-5 alternatives; reward = mean over tokens
- Variable action granularity: full step, 64- or 32-token mini-steps
- Self-reflection trigger fixes ~half of initially-incorrect hard problems

**Key results:** Marco-o1-MCTS (step): 90.40% on MGSM-En vs 84.00% Qwen2-7B-Instruct (+6.17%). Marco-o1-MCTS (mini-step 32): 82.40% on MGSM-Zh vs 76.80% (+5.60%). Test@32 reaches 99.60% (En) / 96.80% (Zh).

*Evolution:* Marco-o1 is an early open attempt (Nov 2024) to demystify OpenAI o1's reasoning by combining CoT SFT with AlphaZero-style MCTS and self-reflection, explicitly extending the reasoning-model trend to open-ended and multilingual/translation tasks.

### SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
*2024 · code · `code_SWE-agent_2405.15793.txt` · arXiv [2405.15793](https://arxiv.org/abs/2405.15793)*

The contribution is the Agent-Computer Interface (ACI): an LM-friendly abstraction over a Linux shell for software engineering, run with GPT-4 Turbo/Claude 3 Opus on the ReAct thought-action loop. Custom commands include a 100-line file viewer (open/goto/scroll), an edit command that replaces a line range and re-displays the file, and capped search/navigation (find_file/search_file/search_dir ≤50 results). A flake8 linter guardrail rejects edits introducing E999/E111-113/F821-822-831/E902 errors, reverting them and showing before/after snippets to stymie cascading failures. Observations older than the last five collapse to one line; format errors trigger retry then de-noising. The ACI is tuned by manual inspection and grid search on dev examples—no LM weights change.

- ACI: LM-friendly shell abstraction co-designed with the environment, no weight updates
- Custom commands: 100-line file viewer, line-range edit+redisplay, capped search/nav
- flake8 linter guardrail reverts syntax/undefined-name edits to prevent cascades
- Observation truncation (last 5) + format-error retry/de-noising

**Key results:** SWE-agent w/ GPT-4 Turbo resolves 12.47% (286/2,294) of full SWE-bench and 18.00% of SWE-bench Lite, far above the 3.8% prior best RAG baseline, and scores 87.7% pass@1 on HumanEvalFix-Python.

*Evolution:* SWE-agent builds on the ReAct agent paradigm and the SWE-bench benchmark (Jimenez et al., 2024), transposing HCI-style interface-design principles to LM agents and arguing that tooling/interaction design, not weight changes, drives coding-agent gains.

### ToolACE: Winning the Points of LLM Function Calling
*2024 · code · `code_ToolACE_2409.00920.txt` · arXiv [2409.00920](https://arxiv.org/abs/2409.00920)*

ToolACE is an automated agentic data-generation pipeline (TSS, SDG, DLV) producing accurate, diverse, complex function-calling SFT data tailored to the target LLM. SDG's complexity evaluator measures difficulty as the model M's negative log-likelihood H_M(x,y) over response tokens; this loss correlates positively with candidate-API count, used-API count, and query/API-description dissimilarity, validating it as a difficulty signal. Multi-agent generation repeats each assistant action and keeps only cross-instance-consistent decisions plus a structured thinking process; self-guided complication re-prompts the user agent to add/remove APIs or shift dissimilarity to hit the target loss band. Downstream training uses LoRA (rank 16, alpha 32, all modules), lr 1e-4, cosine schedule, warmup 0.1, batch 48, 3 epochs, on LLaMA-3.1-8B-Instruct (and Qwen backbones).

- Capability-aware complexity targeting via target-model NLL as difficulty signal
- Multi-agent consistency filtering + structured thinking for tool-calling
- Self-guided complication re-prompts user to hit target loss band
- LoRA SFT: rank 16/alpha 32, lr 1e-4 cosine, batch 48, 3 epochs, all modules

**Key results:** ToolACE-8B (LLaMA-3.1-8B-Instruct + LoRA) ranks 3rd on BFCL-v3 with 59.13 overall (Non-live AST 89.27, Non-live Exec 90.07), competitive with GPT-4-turbo (59.49) and GPT-4o (59.29).

*Evolution:* ToolACE (ICLR 2025) extends the synthetic-data-for-tool-use line of ToolLLM, ToolAlpaca, Gorilla, and xLAM, which relied on public APIs with limited diversity/complexity, by adding evolutionary API synthesis, capability-aware loss-based complexity targeting (cf. MODS, Du et al. 2023), and rigorous dual-layer verification.

### WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
*2024 · code · `code_WebRL_2411.02337.txt` · arXiv [2411.02337](https://arxiv.org/abs/2411.02337)*

WebRL is an off-policy actor-critic RL algorithm for web agents with binary terminal reward, targeting task scarcity, sparse feedback, and policy drift. An outcome-supervised reward model (ORM)—an LLM emitting YES/NO over (instruction, action history, final-state HTML)—supplies 0/1 rewards for newly generated tasks. The policy objective is maximum-entropy RL with a KL anchor to the previous phase's policy: max E[r + β log πref] + βH(πθ). Derived as a Q*-to-policy mapping (Rafailov 2024), the loss L = Eν[β log(πθ/πref) − A*]² both raises good-action and lowers bad-action probability. Advantage A comes from a value network trained by cross-entropy classification (Farebrother 2024) with GAE (λ=0.5, γ=0.9); a replay buffer with actor-confidence (perplexity) filtering prevents forgetting. Contrasted with DPO (no pairwise data needed), PPO (off- vs on-policy), and AWR.

- Off-policy actor-critic with KL-anchored max-entropy objective and binary terminal reward
- LLM ORM (YES/NO) supplies 0/1 reward for reward-less generated tasks
- Loss derived as Q*-to-policy mapping: E[β log(πθ/πref) − A*]²
- GAE (λ=0.5, γ=0.9) value net via cross-entropy classification; perplexity-filtered replay

**Key results:** Llama-3.1-8B + WebRL: 4.8% to 42.4% SR on WebArena-Lite (vs GPT-4-Turbo 17.6%); Llama-3.1-70B reaches 49.1% and GLM-4-9B reaches 43%. The trained 8B ORM verifies success at ~80% accuracy, exceeding GPT-4-based verifiers (~70-73%).

*Evolution:* WebRL builds on DigiRL/AWR actor-critic online RL and WizardLM-style evol-instruct task generation, reacting to DigiRL's fixed task set by adding a self-evolving curriculum and KL-anchored off-policy updates.

### HybridFlow: A Flexible and Efficient RLHF Framework
*2024 · code · `code_veRL-HybridFlow_2409.19256.txt` · arXiv [2409.19256](https://arxiv.org/abs/2409.19256)*

HybridFlow is an RLHF infrastructure framework with a hierarchical hybrid programming model: a single-controller (Ray RPC) coordinates few-node inter-node dataflow, while each model node runs distributed intra-node computation under a multi-controller via ParallelWorker classes (3DParallelWorker, FSDPWorker, ZeROWorker). It decouples computation from data resharding through ~8 transfer protocols (collect/distribute) and a ResourcePool abstraction enabling flexible colocation/split/standalone placement. A 3D-HybridEngine runs actor training and generation with different 3D parallel configs (p-t-d for training; p_g-t_g-d_g-d for generation) with a novel parallel grouping achieving zero memory redundancy and minimal all-gather during train↔gen resharding. An auto-mapping algorithm enumerates placement plans via simulators to minimize per-iteration latency. It supports PPO, ReMax, and Safe-RLHF in ~8 lines, integrating Megatron-LM, FSDP, DeepSpeed, and vLLM (distributed KVCache manager).

- Hierarchical hybrid programming: single-controller dataflow + multi-controller intra-node compute
- ~8 transfer protocols + ResourcePool decouple compute from resharding
- 3D-HybridEngine: distinct train/gen 3D parallelism, zero-redundancy resharding
- Auto-mapping algorithm enumerates placements via simulators to cut iteration latency

**Key results:** PPO on 7B-70B Llama models: HybridFlow beats DeepSpeed-Chat 3.67x (up to 7.84x), OpenRLHF 3.25x (up to 5.93x), NeMo-Aligner 12.52x (up to 20.57x); overall 1.53x-20.57x across PPO/ReMax/Safe-RLHF.

*Evolution:* Builds on the single-controller dataflow tradition of RLlib/RLlib Flow and the multi-controller efficiency of Megatron-LM/vLLM/DeepSpeed, reacting to the inflexibility and redundancy of first-generation RLHF systems (DeepSpeed-Chat, OpenRLHF, NeMo-Aligner) that hardcoded one placement and resharding scheme.

### DataComp-LM: In search of the next generation of training sets for language models
*2024 · data · `data_DataCompLM_2406.11794.txt` · arXiv [2406.11794](https://arxiv.org/abs/2406.11794)*

The method is curation-as-algorithm: standard decoder-only Transformers (GPT-2/Llama-style) trained with next-token prediction + z-loss, LayerNorm-without-bias, qk-LayerNorm, SwiGLU, depth-scaled init in OpenLM/FSDP at 2048 context, with the empirical finding that model-based quality filtering dominates other curation choices. Across seven filters at 1B-1x (PageRank, SemDedup, BGE-embedding linear classifier, AskLLM, CCNet-style perplexity, top-k average logits, fastText), a simple fastText bigram classifier wins (CORE 30.2 vs 27.5 RefinedWeb reproduction). Flagship tricks include cooldown-distribution reweighting, model souping of two cooldowns, and continual long-context adaptation. There is no novel RL/preference algorithm.

- Curation-as-algorithm: filter choice swings 7B/280B MMLU from 35% to 44%
- fastText bigram (OH-2.5+ELI5) beats six other filters at 1B-1x
- Standard decoder-only training: z-loss, qk-LayerNorm, SwiGLU, depth-scaled init, 2048 ctx
- Cooldown reweighting + model souping of two cooldowns + continual long-context adaptation

**Key results:** DCLM-BASELINE 7B trained on 2.6T tokens reaches 63.7% MMLU 5-shot, +6.6pp over MAP-Neo with 40% less compute and within ~2-3pp of Llama3-8B at 6.6x less compute.

*Evolution:* Building on the DataComp vision benchmark and the RefinedWeb/CCNet/MinHash curation lineage (C4, The Pile, Dolma, FineWeb), DCLM systematizes data-centric LM research at 240T-token/7B scale as a reaction to closed-data models like Llama, Mistral, and Gemma.

### KTO: Model Alignment as Prospect Theoretic Optimization
*2024 · data · `data_KTO_2402.01306.txt` · arXiv [2402.01306](https://arxiv.org/abs/2402.01306)*

KTO (Kahneman-Tversky Optimization) is a human-aware loss (HALO) derived from prospect theory that directly maximizes human utility of generations rather than preference log-likelihood. The paper defines HALOs—losses expressible as E[v(rθ(x,y)−EQ[rθ])]+C with non-decreasing value function v concave in gains—and proves DPO and PPO-Clip are HALOs. KTO swaps the unstable Kahneman-Tversky power value function for a logistic σ, uses KL(πθ‖πref) as the reference point z0 (biased microbatch mismatched-output estimator, not backpropagated), and applies separate loss-aversion weights λD,λU plus risk-aversion β: L_KTO=E[λ_y−v(x,y)], with v=λDσ(β(rθ−z0)) for desirable and λUσ(β(z0−rθ)) for undesirable outputs. It needs only binary desirable/undesirable signal. A reference-model-free variant assumes uniform πref. Theory shows preference-likelihood maximization need not maximize utility.

- HALO family: DPO and PPO-Clip proven to be HALOs; KTO directly optimizes utility
- Logistic value function (not KT power), KL reference point z0 not backpropagated
- Separate λD/λU loss-aversion weights + risk-aversion β; binary desirable/undesirable signal only
- Reference-free variant uses uniform πref → rθ−z0 reduces to logπθ−entropy

**Key results:** KTO matches or exceeds DPO across 1B–30B models; on Zephyr-β-SFT aligned on UltraFeedback, GSM8K rises 40.0→53.5 (+13.5 pts) over DPO. Human-eval winrate vs SFT targets: KTO 72.9% vs DPO 62.1% (p<0.05).

*Evolution:* Building on DPO (Rafailov et al., 2023) and PPO-RLHF, KTO reframes alignment through prospect theory and the HALO family, arguing the inductive bias of the loss—not just data or reward modeling—drives success.

### MAGPIE: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing
*2024 · data · `data_Magpie_2406.08464.txt` · arXiv [2406.08464](https://arxiv.org/abs/2406.08464)*

MAGPIE's method is seed-free self-synthesis: because aligned models were fine-tuned on the pre-query template, feeding only that template triggers the model to sample a real user instruction before EOS (Step 1); the instruction is wrapped with pre+post-query templates and the same LLM generates the response, optionally greedily to favor high-probability training-derived tokens (Step 2). This avoids the diversity collapse and seed dependence of Self-Instruct/Evol-Instruct. Preference data is built by sampling k=5 responses at T=0.8 and labeling highest/lowest ArmoRM score as chosen/rejected for DPO. Multi-turn appends the pre-query template with a role-reinforcing system prompt; domain/multilingual control uses tailored system prompts (math, code, Chinese) or specialized models like DeepSeek-Coder-V2 and Qwen2-Math.

- Seed-free: pre-query template alone elicits a real instruction then a response
- Greedy generation favors high-probability, training-derived tokens
- Preference pairs via k=5 samples at T=0.8 + ArmoRM chosen/rejected for DPO
- Domain/multilingual control via tailored system prompts or specialized models

**Key results:** Llama-3-8B-Base + MAGPIE-Pro-300K-Filtered SFT + MAGPIE-Pro-DPO reaches AlpacaEval2 LC 50.10% vs GPT-4-Turbo and WR 53.53% vs official Llama-3-8B-Instruct, surpassing both while using only ~400K data versus >10M.

*Evolution:* MAGPIE (2024) extends synthetic-instruction lines such as Self-Instruct, Evol-Instruct, and UltraChat by removing their seed-question and prompt-engineering dependence, instead extracting an aligned model's own instruction distribution via its chat template.

### ORPO: Monolithic Preference Optimization without Reference Model
*2024 · data · `data_ORPO_2403.07691.txt` · arXiv [2403.07691](https://arxiv.org/abs/2403.07691)*

ORPO (Odds Ratio Preference Optimization) is a reference-free, monolithic preference alignment algorithm. Its objective is L_ORPO=E[L_SFT + λ·L_OR], where L_SFT is standard causal-LM NLL on chosen responses and L_OR=−log σ(log(odds(yw|x)/odds(yl|x))) with odds(y|x)=P(y|x)/(1−P(y|x)). The log-sigmoid-wrapped log-odds-ratio penalizes disfavored relative to favored responses during SFT. The gradient factors into δ(d) (penalty accelerating updates when the model favors the rejected response) and h(d) (weighted contrast amplifying gradients when likelihood is low). No reference model means two forward passes per batch instead of four and no SFT warm-up. The paper argues odds ratio is milder than probability ratio and avoids over-suppressing rejected-token logits before domain adaptation. λ tuned per model (0.1 Mistral, 0.2 Llama-2, 0.25 Phi-2).

- Monolithic SFT+preference objective, no reference model, no SFT warm-up
- L = L_SFT + λ·L_OR; L_OR is log-sigmoid-wrapped log-odds ratio of favored vs disfavored
- Gradient factors: δ(d) penalty + h(d) amplification when likelihood is low
- Halves forward passes/batch (2 vs 4); λ tuned per backbone (0.1/0.2/0.25)

**Key results:** Mistral-ORPO-β (7B): 12.20% AlpacaEval2.0, 7.32 MT-Bench, 66.19% IFEval instruction-level loose—surpassing Zephyr-β and Llama-2-Chat (13B) with single-epoch training on UltraFeedback alone.

*Evolution:* ORPO builds on DPO's reference-free spirit (Rafailov 2023) and the unlikelihood-training tradition, reacting against the unstable multi-stage SFT->RLHF/PPO pipeline and the cost of keeping a frozen reference model.

### Scaling Laws for Data Filtering—Data Curation cannot be Compute Agnostic
*2024 · data · `data_ScalingLawsFiltering_2404.07177.txt` · arXiv [2404.07177](https://arxiv.org/abs/2404.07177)*

Introduces scaling laws for heterogeneous, limited web data extending y = a·n^b + d with two additions. (1) Utility decays exponentially with repetition, b_{k+1}=b·(1/2)^(k/τ)=b·δ^k, with half-life τ capturing pool diversity, giving closed-form loss y_k = a·n_1^{b_1}·∏_{j=2}^k (n_j/n_{j-1})^{b_j} + d. (2) Theorem 1 combines p pools without jointly training: merged half-life τ_hat=p·τ and effective utility b_eff is a weighted mean of per-pool utilities. For contrastive CLIP, where N samples yield O(N²) comparisons, τ scales with pool size: τ_hat=(N_hat/N)·τ. Parameters (utility b, half-life τ, shared normalizer a, irreducible d) are fit by grid search since gradient/SciPy optimizers were unstable. Three axiom formulations are compared; only the utility-decay version handles heterogeneous mixtures.

- Extends y=a·n^b+d with exponential utility decay b·δ^k, half-life τ captures diversity
- Closed-form multi-epoch loss y_k via per-epoch utility exponents
- Theorem 1 merges p pools without joint training: τ_hat=p·τ, b_eff weighted mean
- CLIP contrastive scaling: τ_hat=(N_hat/N)·τ; parameters fit by grid search

**Key results:** At 128M compute, LAION filtering gives +7.5% avg zero-shot accuracy over 18 tasks versus no-filtering, but beyond ~450M samples seen, unfiltered common crawl outperforms LAION-filtered data.

*Evolution:* Builds on Kaplan (2020) and Hoffmann/Chinchilla (2022) scaling laws and on Muennighoff et al. (2023)'s observation that >4 epochs yield negligible gains, but extends these to heterogeneous web data and contrastive CLIP training, where prior fits (Cherti et al. 2023) failed for repeated data.

### Self-Rewarding Language Models
*2024 · data · `data_SelfRewarding_2401.10020.txt` · arXiv [2401.10020](https://arxiv.org/abs/2401.10020)*

Self-Rewarding LMs make a single LLM serve simultaneously as instruction follower and as its own reward model via LLM-as-a-Judge prompting, replacing the frozen reward model of RLHF/RLAIF with one that updates each iteration. Self-instruction creation: generate a new prompt (Self-Instruct few-shot), sample N=4 candidate responses, and have the same model score them 0–5 with an additive five-criteria rubric (relevance, coverage, usefulness, clarity, expertise) plus CoT justification. Preference pairs (highest vs lowest scored candidate; ties discarded) train the next model with DPO (β=0.1). Key tricks: seeding the judge with EFT data so scoring doesn't collapse to one bucket; an additive rather than multiple-choice judge prompt (65.1% vs 26.6% pairwise accuracy); averaging 3 sampled judgments; and shared judge/follower parameters for cross-task transfer.

- One LLM is both policy and reward model, updated each DPO iteration (β=0.1)
- LLM-as-Judge with additive 5-criteria rubric + CoT, 3 averaged judgments
- EFT-seeded judge avoids score collapse; additive prompt beats multiple-choice (65.1% vs 26.6%)
- Preference pairs = highest vs lowest of N=4 candidates; ties discarded

**Key results:** Llama 2 70B Self-Rewarding M3 reaches 20.44% AlpacaEval 2.0 win rate over GPT-4 Turbo, beating Claude 2 (17.19%), Gemini Pro (16.85%), and GPT-4 0613 (15.76%). Reward-modeling pairwise accuracy rises each iteration (65.1% SFT -> 78.7% M1 -> 80.4% M2 -> 81.7% M3).

*Evolution:* Builds on Iterative DPO / Pairwise Cringe Optimization (Xu et al.), Self-Instruct (Wang et al.), and LLM-as-a-Judge / RLAIF (Constitutional AI), but folds the reward model into the policy so it co-improves across DPO iterations rather than staying frozen.

### SimPO: Simple Preference Optimization with a Reference-Free Reward
*2024 · data · `data_SimPO_2405.14734.txt` · arXiv [2405.14734](https://arxiv.org/abs/2405.14734)*

SimPO is a reference-free offline preference optimization algorithm with two key designs. (1) A length-normalized implicit reward equal to the average log probability of a response: r(x,y)=(β/|y|)·∑ log πθ(y_i|x,y<i), matching the likelihood metric used at inference (beam search/multiple-choice scoring) and removing DPO's train/inference mismatch; summed log-prob is avoided because it biases toward longer sequences. (2) A target reward margin γ>0 inserted into the Bradley-Terry objective so the winning reward must exceed the losing one by at least γ. Loss: L=−E log σ((β/|y_w|)log π(y_w|x) − (β/|y_l|)log π(y_l|x) − γ). No reference model cuts runtime ~20% and GPU memory ~10% vs DPO; no explicit KL term (low KL maintained empirically via small lr, diverse data, LLM robustness). Good defaults: β∈[2.0,2.5], γ∈[0.5,1.5].

- Reference-free: length-normalized implicit reward = average log prob matches inference scoring
- Target reward margin γ in Bradley-Terry so winner must exceed loser by ≥γ
- No reference model → ~20% faster, ~10% less GPU memory vs DPO
- No explicit KL; low KL maintained via small lr, diverse data, model robustness

**Key results:** Gemma-2-9B-it-SimPO reaches 72.4% LC win rate on AlpacaEval 2, 59.1% WR on Arena-Hard, and ranks 1st among <10B models on Chatbot Arena (lifting Gemma-2-9B-it from 36th to 25th overall).

*Evolution:* SimPO builds directly on DPO (2023) and contemporaneous reference-free work such as ORPO, reacting to DPO's mismatch between its log-ratio reward and the average-log-likelihood used at generation, plus the cost of carrying a reference model.

### Tülu 3: Pushing Frontiers in Open Language Model Post-Training
*2024 · data · `data_Tulu3_2411.15124.txt` · arXiv [2411.15124](https://arxiv.org/abs/2411.15124)*

Tülu 3's headline method contribution is RLVR: it keeps the standard KL-constrained RLHF objective but swaps the learned reward model for a deterministic verifier v(x,y) (α=10 on verified-correct, 0 otherwise), optimized via PPO over math (exact-match answer extraction) and verifiable instruction following (constraint checkers). The fully open recipe chains persona-driven synthetic SFT, then length-normalized DPO (à la SimPO), then RLVR, deliberately favoring verifiable rewards over a learned RM for the final stage. Infrastructure scales PPO to 405B using ZeRO-3 with asynchronous RL on vLLM/PagedAttention inference GPUs.

- PPO tricks: value model init from a general UltraFeedback RM (EOS-token reward), dropout disabled, non-EOS penalty −10, advantage whitening, multi-epoch shuffled prompts.
- Preference stage uses length-normalized DPO, chosen over SimPO, vanilla DPO, and PPO.
- RLVR replaces the learned RM with a rule verifier; the recipe is fully open data + code.

**Key results:** Tülu 3 70B averages 76.2 on Tülu 3 Eval, surpassing Llama 3.1 70B Instruct (74.1), GPT-4o-mini (69.6), and Claude 3.5 Haiku (75.3); 405B reaches 80.7.

*Evolution:* Tülu 3 extends the open Tülu 2/Zephyr-β line toward closed-style multi-stage recipes and formalizes verifiable-reward RL (RLVR), anticipating the 2024–25 GRPO/DeepSeek-R1 wave and reproducible open post-training.

### DeepSeek-V3 Technical Report
*2024 · report · `report_DeepSeek-V3_2412.19437.txt` · arXiv [2412.19437](https://arxiv.org/abs/2412.19437)*

DeepSeek-V3's post-training core is Group Relative Policy Optimization (GRPO), inherited from V2: the critic is dropped and advantage is estimated from group rewards as A_i = (r_i − mean)/std, with a clipped policy-ratio objective plus KL-to-reference penalty (β) and clip ε. A hybrid reward combines a rule-based RM (boxed math answers, LeetCode compiler/test-case checks) and a model-based RM (free-form ground-truth matching, creative-writing feedback) trained from the V3 SFT checkpoint. The headline algorithmic move is distilling long-CoT reasoning from a DeepSeek-R1 model into V3 via RL-trained expert generators and rejection sampling, transferring reflection/verification patterns while controlling length; self-rewarding via constitutional AI fills gaps where rules fail.

- GRPO: critic-free, group-normalized advantages, clipped ratio + KL(π‖πref).
- Hybrid RM: rule-based (verifiable) + model-based RM trained from the SFT checkpoint.
- Long-CoT distillation from R1 via high-temperature RL generation + rejection sampling.

**Key results:** DeepSeek-V3 (671B-total/37B-active MoE) reaches AIME 2024 39.2 Pass@1, MATH-500 90.2 EM, 85.5 Arena-Hard win rate, matching GPT-4o/Claude-3.5-Sonnet at ~$5.6M total cost.

*Evolution:* Building on DeepSeek-V2's MLA/DeepSeekMoE and GRPO, V3 pioneers distilling long-CoT reasoning from the R1 series into a standard aligned MoE model, anticipating the 2025 wave of distilling strong reasoners into efficient bases.

### Gemma 2: Improving Open Language Models at a Practical Size
*2024 · report · `report_Gemma2_2408.00118.txt` · arXiv [2408.00118](https://arxiv.org/abs/2408.00118)*

Gemma 2's core method contribution is using knowledge distillation as the pre-training objective for the 2B and 9B models, replacing one-hot next-token prediction with the teacher's full next-token distribution (−Σ_x P_T log P_S). Students train on >50× the compute-optimal token count to simulate training beyond available data. Architecture combines interleaved local sliding-window (4096) and global (8192) attention, Grouped-Query Attention, logit soft-capping (50.0 attention, 30.0 final), and RMSNorm pre/post-norm. Post-training stacks SFT with on-policy distillation, RLHF against a large multi-turn reward model, and weight averaging of phase checkpoints.

- Pre-training objective: cross-entropy to teacher's full next-token distribution, not one-hot labels.
- Students over-trained >50× compute-optimal tokens; 2B/500B from a 7B teacher hits 67.7 vs 60.3 from scratch.
- Post-training: SFT + on-policy distillation + multi-turn-RM RLHF + checkpoint weight averaging.

**Key results:** Gemma 2 27B-IT reaches LMSYS Elo 1218 (beating Llama-3 70B-IT's 1206); 9B-IT (1187) matches GPT-4-0314, and 2.6B-IT beats GPT-3.5-Turbo-0613.

*Evolution:* Builds on Hinton (2015) and Gemini 1.5 distillation, repurposing it as a pre-training substitute for next-token prediction to push small models past compute-optimal token counts, motivating later teacher-distilled pre-training and weight-averaged RLHF recipes.

### InternLM2 Technical Report
*2024 · report · `report_InternLM2_2403.17297.txt` · arXiv [2403.17297](https://arxiv.org/abs/2403.17297)*

InternLM2's method core is COOL RLHF (Conditional Online RLHF). Instead of LLaMA2-style separate reward models for conflicting domains, a single Conditional Reward Model is conditioned on domain-specific system prompts; its loss is a focal ranking loss with a difficulty-decay coefficient (γ=2) plus a log-barrier penalty confining scores to [−5,5] (λ=0.02). To fight reward hacking, Online RLHF runs three rounds with a Fast Path (20–100 preference patches per round contrasting early/late PPO responses) and a Slow Path (human-labeled pairs across SFT/early/late PPO to raise the reward upper bound). PPO uses four same-size models with critic initialized from the RM (50-iter warmup), a pre-train gradient term (coeff 0.5), KL=0.01, PPO λ=0.99, top-p=0.9, no value clipping or advantage normalization.

- Conditional RM: one system-prompt-conditioned model replaces multiple domain RMs; focal ranking loss + log-barrier.
- Online RLHF: 3 rounds with Fast Path (patched early/late contrasts) + Slow Path (human labels across stages).
- PPO: critic init from RM, pre-train-gradient term, KL=0.01, no value clipping/advantage norm, ~200k queries/~400 iters.

**Key results:** InternLM2-Chat-20B hits AlpacaEval win rate 21.8, GSM8K 79.6, MATH 32.4, HumanEval 67.7, MTBench 7.9, beating GPT-3.5 on reasoning.

*Evolution:* COOL RLHF refines LLaMA2's separate helpful/harmless RMs into a single system-prompt-conditioned RM with multi-round online patching against reward hacking, anticipating later iterative/online RLHF and process-reward trends.

### The Llama 3 Herd of Models
*2024 · report · `report_Llama3_2407.21783.txt` · arXiv [2407.21783](https://arxiv.org/abs/2407.21783)*

Llama 3's post-training method is SFT → rejection sampling (RS) → Direct Preference Optimization, deliberately chosen over PPO for lower compute and better IFEval. The reward model reuses the Llama-2 objective minus the margin term, concatenating prompt with multiple ranked responses (edited>chosen>rejected, shuffled) per row. RS samples K=10–30 outputs from the latest policy and keeps the RM-best, accelerated with PagedAttention (~2× throughput). DPO adds two tricks: masking special header/termination formatting tokens in the loss (prevents tail repetition and abrupt termination from the contrastive objective), and an NLL regularization term scaled 0.2 on chosen sequences to stabilize training. Model averaging merges checkpoints across data/hyperparameter variants; tool use, math, and factuality are taught via synthetic self-generated trajectories with execution/verifier feedback.

- Pipeline: SFT → rejection sampling (K=10–30, PagedAttention) → DPO, explicitly preferred over PPO.
- DPO tricks: mask formatting tokens in the loss; 0.2-scaled NLL reg on chosen sequences.
- Synthetic self-generated trajectories with execution/verifier feedback teach tools, math, factuality; checkpoint averaging across variants.

**Key results:** Llama 3 405B matches GPT-4 on human pairwise win rate, with HumanEval 89.0, MGSM 91.6, and 100% Needle-in-a-Haystack to 128K; 8B/70B are best-in-class for their sizes.

*Evolution:* Refines the Llama-2 RM+SFT+RS recipe and the 2023 DPO trend, favoring stable SFT/RS/DPO over PPO while scaling human preferences to millions and leaning on self-generated synthetic data, foreshadowing the 2024-25 shift to synthetic-data-driven post-training and RLVR.

### MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies
*2024 · report · `report_MiniCPM_2404.06395.txt` · arXiv [2404.06395](https://arxiv.org/abs/2404.06395)*

MiniCPM's method core is the Warmup-Stable-Decay (WSD) learning-rate scheduler: linear warmup to η, a constant stable phase, then a decay phase f(s−T)·η that splits training into high-LR exploration and a distinct annealing phase. This enables continuous training without a predefined token budget and reuse of stable-stage checkpoints; loss drops dramatically during decay and ~10% of tokens suffices for annealing, matching a Cosine schedule trained to that step. WSD makes scaling-law measurement linear-cost (O(mC)): training 6 model sizes (0.04B–2B) and decaying from 10N–60N data fits L(N,D)=C_N·N^−α + C_D·D^−β + L0 (α=0.29, β=0.23; compute-optimal data/model ratio ~192× vs Chinchilla's 20×). Supporting methods include Model Wind Tunnel Experiments (μP width/depth scaling, optimal batch law bs=1.21e9/L^6.24, stable LR ~0.01), DPO alignment, Sparse Upcycling MoE (2/8 experts, load-balancing loss ×0.01), and Int4 GPTQ quantization.

- WSD scheduler separates exploration (constant high LR) from a decay/anneal phase; ~10% of tokens for annealing.
- Linear-cost scaling-law fitting: 6 sizes × multiple data counts → α=0.29, β=0.23, compute-optimal ~192× data/model ratio.
- Supporting: μP wind-tunnel transfer, two-stage pre-training, DPO, Sparse Upcycling MoE, Int4 GPTQ.

**Key results:** MiniCPM-2.4B surpasses Mistral-7B and Llama2-13B on average across C-Eval/CMMLU/MMLU/HumanEval/MBPP/GSM8K/MATH/BBH; MiniCPM-2.4B-DPO reaches MTBench 7.25, beating Llama2-70B-Chat.

*Evolution:* Building on Kaplan/Chinchilla scaling laws and μP/Tensor Program transfer, MiniCPM recovers a much higher (~192×) compute-optimal ratio under modern overtraining; its WSD scheduler and anneal-on-curated-data recipe presage later 2024 annealing and continuous-training practices.

### LLM Pruning and Distillation in Practice: The Minitron Approach
*2024 · report · `report_Minitron_2408.11796.txt` · arXiv [2408.11796](https://arxiv.org/abs/2408.11796)*

Minitron's contribution is an updated compression recipe for when the original pretraining data is inaccessible. It adds a teacher-correction phase adapting the unpruned teacher to the new distillation dataset, then structured pruning via joint width pruning (hidden size, MLP intermediate dim, embedding channels; attention heads retained) and depth pruning. A new downstream-task-based saliency criterion ranks contiguous layer blocks by Winogrande accuracy rather than LM validation loss/PPL, since the latter fails to predict downstream quality (dropping layers 16–31 yields 0.595 vs 0.5 for importance-based non-contiguous selection). Single-shot pruning uses l2-norm/mean aggregation on forward-pass activations from 1024 calibration samples. Distillation is logit-only, minimizing forward KL divergence between teacher and student logits and discarding the LM cross-entropy loss; NAS is skipped in favor of manual architectures. Training runs on 32 DGX H100 nodes.

- Teacher-correction phase adapts the unpruned teacher to the new distillation dataset before pruning.
- Depth-pruning saliency uses Winogrande accuracy on contiguous layer blocks, not PPL.
- Logit-only forward-KL distillation with no LM cross-entropy loss; single-shot pruning on 1024 calibration samples.

**Key results:** MN-Minitron-8B matches Llama 3.1 8B accuracy using 40× fewer tokens (380B vs 15T); Llama-3.1-Minitron-4B reaches near-teacher quality with 150× fewer tokens (94B).

*Evolution:* Extends the original Minitron (2407.14679) and Gromov et al. contiguous-layer depth pruning to the private-pretraining-data case via teacher correction and downstream-task depth saliency, exemplifying the 2024 shift to cheap SLM derivation from frontier models.

### Mixtral of Experts
*2024 · report · `report_Mixtral_2401.04088.txt` · arXiv [2401.04088](https://arxiv.org/abs/2401.04088)*

Mixtral's core contribution is architectural: a decoder-only Sparse Mixture-of-Experts transformer where each of 32 layers replaces the FFN sub-block with 8 SwiGLU experts and a top-2 router G(x)=Softmax(TopK(x·W_g)) picks 2 experts per token, additively combining outputs y=Σ Softmax(Top2(x·W_g))_i·SwiGLU_i(x), yielding 47B sparse / 13B active parameters. It resembles GShard but replaces all FFN blocks (vs every other) and uses simpler top-2 gating; efficient inference uses Megablocks sparse-matrix kernels and Expert Parallelism. The instruction-following post-training method is SFT then DPO (Rafailov et al. 2023), optimizing the policy directly from pairwise preferences without a separate reward model or PPO—the practical trick producing Mixtral 8x7B – Instruct.

- SMoE: 32 layers, 8 SwiGLU experts each, top-2 router, 47B sparse / 13B active.
- All FFN blocks replaced (vs GShard's every-other); Megablocks kernels + Expert Parallelism for inference.
- Instruction tuning: SFT → DPO, no separate RM or PPO.

**Key results:** Mixtral 8x7B matches or beats Llama 2 70B and GPT-3.5 with 5× fewer active params (MMLU 70.6%, GSM8K 74.4% maj@8); Instruct reaches MT-Bench 8.30 and Arena Elo 1121, best open-weights as of Dec 2023.

*Evolution:* Builds on the GShard/Shazeer sparse-MoE line and Mistral 7B, adopting DPO over PPO-based RLHF; as one of the first open-weights sparse-MoE LLMs matching dense 70B at 5× lower active compute, it validated open MoE and motivated the wave that followed.

### Phi-4 Technical Report
*2024 · report · `report_Phi-4_2412.08905.txt` · arXiv [2412.08905](https://arxiv.org/abs/2412.08905)*

Phi-4's headline post-training contribution is Pivotal Token Search (PTS), which generates token-level DPO pairs. PTS observes that a completion's success probability often hinges on a few "pivotal" tokens; full-length DPO dilutes their signal and wrongly rewards low-probability non-pivotal tokens. The Subdivide procedure recursively splits the token sequence at the cumulative log-probability midpoint until per-segment change in p(success) falls below threshold p_gap or the segment is one token; tokens with a sharp p(success) change (estimated by sampling completions and checking an oracle) are kept. Each becomes a DPO pair with the prefix as query and accepted/rejected tokens those that raise/lower p(success); targets are filtered to 0.2≤p(success)≤0.8. Unlike contrastive-estimation proxies, PTS directly estimates p(success). A second judge-guided DPO round uses GPT-4o to label ~850k pairs; targeted SFT/DPO refusal data mitigates hallucination.

- PTS: token-level DPO pairs from a few "pivotal" tokens that swing p(success).
- Subdivide recursively splits at the cumulative log-prob midpoint; targets filtered to 0.2≤p(success)≤0.8.
- Second DPO round: ~850k GPT-4o-judged pairs; refusal data curbs hallucination.

**Key results:** phi-4 (14B) scores MATH 80.4 and GPQA 56.1, surpassing its teacher GPT-4o (74.6/~50.6) and beating Qwen-2.5-14B-Instruct on 9/12 benchmarks.

*Evolution:* Extends the Phi "textbooks are all you need" thesis beyond GPT-4 distillation, showing curated synthetic data plus token-level preference optimization (PTS) lets a 14B model surpass its teacher on STEM reasoning at ~10× lower cost than long-CoT models.

### Qwen2.5 Technical Report
*2024 · report · `report_Qwen2.5_2412.15115.txt` · arXiv [2412.15115](https://arxiv.org/abs/2412.15115)*

Qwen2.5's core contribution is a large-scale post-training recipe: intricate SFT (>1M examples) followed by two-stage RL—offline DPO (Rafailov 2023) then online GRPO (Shao 2024). For DPO, the SFT model resamples responses to new queries; execution-feedback/answer-matching checks label pass/fail as positive/negative preference pairs (~150K), trained with the Online Merging Optimizer. GRPO samples 8 responses per query and uses group-relative advantage under a reward model trained on six criteria (truthfulness, helpfulness, conciseness, relevance, harmlessness, debiasing), with queries scheduled by reward variance. Supporting methods include ABF/YARN/DCA length extrapolation (4× extension; Turbo to 1M tokens) and Minference-based sparse attention (12.5× attention compute reduction). A key empirical finding: RM benchmark scores (Reward Bench, RMB, PPE) do not predict downstream RL quality, motivating multi-benchmark RM evaluation.

- Pipeline: >1M-example SFT → offline DPO (~150K execution/answer-labeled pairs) → online GRPO (8 samples/query).
- GRPO reward model trained on six criteria; queries scheduled by reward variance.
- Finding: RM-benchmark scores don't predict downstream RL quality—over-optimizing one triggers Goodhart's law.

**Key results:** Qwen2.5-72B-Instruct matches or exceeds Llama-3.1-405B-Instruct (~6× larger) on MMLU-redux 86.8 vs 81.6, plus MATH, MBPP, MultiPL-E, LiveCodeBench, Arena-Hard, MTBench; Qwen2.5-RM-72B leads on PPE.

*Evolution:* Extends the Qwen2/Math/Coder lineage with a deliberate offline-DPO-then-online-GRPO two-stage RL design and 18T-token data scaling; its finding that RM-benchmark scores fail to predict downstream RL quality motivates better reward-model evaluation and later specialized/reasoning models.

### Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models — The Story Goes On
*2024 · report · `report_Skywork-Math_2407.08348.txt` · arXiv [2407.08348](https://arxiv.org/abs/2407.08348)*

Skywork-Math's method is SFT-only alignment on common 7B pre-trained LLMs (LLaMA2-7B, Mistral-7B, DeepSeekMath-Base-7B) using standard auto-regressive cross-entropy loss over query-response pairs. The novelty is a two-stage data-synthesis-and-SFT pipeline: GPT-4 synthesizes diverse normal problems (stage 1) then hard problems (stage 2), with a core-set seed selector and three combined augmentation styles to maximize diversity. Training uses 3 epochs, global batch 32, AdamW (no weight decay), lr 2e-5 (LLaMA2) or 2e-6 (Mistral/DeepSeekMath), warmup 0.03, 8× A800-80G, max length 2048, with vLLM inference reusing the SFT prompt (prefix "\nThe answer is"). The key empirical claim is that data quantity dominates data quality before saturation and that synthetic SFT alone matches or beats 120B-token continual pre-training.

- SFT-only on common 7B bases; standard next-token cross-entropy over query-response pairs.
- Two-stage GPT-4 synthesis: diverse normal problems then hard problems, with core-set seeds + 3 augmentation styles.
- Training: 3 epochs, AdamW, base-dependent lr (2e-5 vs 2e-6), vLLM inference with shared SFT prompt.

**Key results:** Skywork-Math-Mistral-7B (SFT only on 2.5M synthetic GPT-4 data) hits 51.2% MATH and 83.9% GSM8K, SOTA among <10B models and surpassing an early GPT-4 on MATH.

*Evolution:* Builds on 2023-24 synthetic-SFT trends (MetaMath, WizardMath/Evol-Instruct, Xwin-Math) and pushes back against LIMA "less is more", showing scaled synthetic SFT rivals 120B-token continual pre-training and motivating larger, harder synthetic-data scaling for math.

### Improve Mathematical Reasoning in Language Models by Automated Process Supervision
*2024 · rl · `rl_OmegaPRM_2406.06592.txt` · arXiv [2406.06592](https://arxiv.org/abs/2406.06592)*

OmegaPRM is a divide-and-conquer MCTS algorithm (inspired by AlphaGo Zero) that automates process-supervision data collection. Its first key idea is binary-search Monte Carlo: split a candidate solution at its midpoint and run k=8 rollouts; if any reaches the gold answer the prefix is correct (error lies after), else the error lies before, recursing in O(k log M) instead of O(kM). Second, MCTS over a state-action tree reuses all rollouts: Select uses a PUCT-style rule with Q(s,r)=α^(1−MC(s))·β^(len(r)/L) (α=0.5, β=0.9, L=500, c_puct=0.125) favoring "supposed-to-be-correct wrong-answer" rollouts; Binary Search locates the first error and adds edges; Maintain updates N, MC, Q. The PRM is trained with pointwise soft-label BCE (ŷ=MC(s)), pointwise hard label, or Bradley-Terry pairwise loss; soft label wins. Inference uses PRM-weighted majority voting (product of step scores).

- Binary-search Monte Carlo localizes the first error in O(k log M) instead of O(kM).
- MCTS with PUCT selection (α=0.5, β=0.9, c_puct=0.125) reuses all rollouts; favors "supposed-correct but wrong" paths.
- PRM trained with soft-label BCE (ŷ=MC(s)) beats hard-label and Bradley-Terry; PRM-weighted majority voting at inference.

**Key results:** Gemini Pro 51%→69.4% on MATH500 and 86.4%→93.6% on GSM8K; Gemma2 27B 42.3%→58.2% (MATH500) and 74.0%→92.2% (GSM8K) via OmegaPRM-weighted voting; 1.5M auto-annotations at 75× brute-force efficiency.

*Evolution:* Builds on PRM800K's human process supervision and Math-Shepherd/MiPS's per-step Monte Carlo automation, importing AlphaGo Zero's MCTS to make step-level reward data cheap and large-scale, motivating the later PRM and RLVR wave.

### ReFT: Reasoning with Reinforced Fine-Tuning
*2024 · rl · `rl_ReFT_2401.08967.txt` · arXiv [2401.08967](https://arxiv.org/abs/2401.08967)*

ReFT augments SFT with online PPO. After warm-up, the policy repeatedly samples CoTs for a question, extracts the answer, and updates via PPO's clipped objective. Reward comes directly from ground-truth answers (no trained reward model): r=1 if the extracted answer matches, 0.1 partial credit if numeric-but-wrong, 0 otherwise; total reward adds a KL penalty −β·KL(π_θ‖π_θ^(0)) against the warm-up policy (β=0.01 P-CoT, 0.05 N-CoT). The value model is a linear head on the policy's last hidden states (shared torso, initialized from warm-up). GAE uses λ=1, γ=0.95, clip ε=0.2, U=2 PPO updates/step, value-loss weight α=5. This addresses the SFT weakness of learning from a single CoT path: PPO explores many paths and uses negative (incorrect) samples. Preliminary DPO/IPO was only on par with offline self-training because it cannot explore online.

- Online PPO after SFT warm-up; reward is ground-truth answer match (1 / 0.1 numeric-wrong / 0), no trained RM.
- KL penalty against the warm-up policy (β=0.01 P-CoT, 0.05 N-CoT); shared-torso linear value head.
- GAE λ=1, γ=0.95, clip ε=0.2, 2 updates/step; DPO/IPO underperformed because they can't explore online.

**Key results:** CodeLLAMA-7B + ReFT reaches 75.28 P-CoT accuracy on GSM8K vs 63.68 for SFT (+11.6), and 81.2 with reward-model reranking, surpassing GPT-3.5-turbo (78.0) using only a 7B model.

*Evolution:* An early-2024 demonstration that outcome-only RL (PPO with answer-derived rewards, no trained RM) on the same SFT data substantially beats SFT for math, anticipating the GRPO/RLVR direction later popularized by DeepSeekMath and DeepSeek-R1.

### Technical Report on Slow Thinking with LLMs: II — Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems
*2024 · rl · `rl_STILL-2_2412.09413.txt` · arXiv [2412.09413](https://arxiv.org/abs/2412.09413)*

STILL-2 is a reproduction recipe for o1-like reasoning built on Qwen2.5-32B-Instruct. Imitation uses SFT (lr=1e-5, batch 96) on a few thousand distilled long-form thought examples with special format tokens, eliciting a single-pass "thought then solution" output. Exploration replaces an explicit reward model with answer-matching: multiple rollouts per problem, keep trajectories whose final answer matches ground truth. Self-improvement offers two objectives—rejection-sampling SFT and DPO; for DPO the key tricks are pairing high-perplexity correct trajectories against low-perplexity incorrect ones (harder discrimination), aligning only the thought part (since the solution is near-deterministic given a good thought), and adding an SFT loss for stability. The contribution is showing complex reasoning can emerge from small distilled demonstrations plus self-generated correct trajectories without process reward models or test-time tree search.

- Imitation: SFT (lr=1e-5, batch 96) on ~few-thousand distilled long-CoT examples with format tokens.
- Exploration: answer-matching rollouts replace an explicit reward model; keep ground-truth-matching trajectories.
- Self-improve via rejection-sampling SFT or DPO; DPO pairs high-PPL correct vs low-PPL incorrect, aligns only the thought, with an SFT loss for stability.

**Key results:** STILL-2 (Qwen2.5-32B-Instruct + 3.9k distilled SFT) reaches 90.2% MATH-OAI and 46.7% AIME2024 (vs 80.0/13.3 backbone), approaching o1-preview's 85.5/44.6.

*Evolution:* Rejects the authors' own Nov-2024 reward-guided tree-search report in favor of distillation from R1/QwQ plus self-improvement, evidencing that small distilled long-CoT data plus rejection-sampling/DPO can elicit cross-domain slow thinking and motivating later RL-based and scaled self-improvement recipes.

### Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
*2024 · rl · `rl_TestTimeCompute_2408.03314.txt` · arXiv [2408.03314](https://arxiv.org/abs/2408.03314)*

The contribution is a "compute-optimal" test-time scaling strategy that adaptively allocates inference compute per prompt. Two mechanisms are studied. (1) Search against a dense process reward model: the PRM is trained without human labels using Monte-Carlo rollout reward-to-go estimates (Math-Shepherd approach), used in best-of-N weighted (marginalizing verifier scores by final answer), beam search (BFS-V style, beam width M), and lookahead search (k-step rollouts, a special case of MCTS with stochastic exploration removed); step-wise aggregation uses the last-step PRM score. (2) Refining the proposal distribution via an iterative revision model (SFT on multi-turn incorrect→correct trajectories, building on Recursive Introspection), then sampling sequential revisions combined with parallel samples. Optimal hyperparameters (search type, sequential/parallel ratio, budget) are selected per difficulty bin (5 quantiles of base-model pass@1 from 2048 samples); easy prompts favor sequential revisions, hard prompts favor parallel/beam search.

- PRM trained unsupervised via MC rollout reward-to-go (Math-Shepherd); used for weighted best-of-N, beam search, and lookahead (MCTS without stochastic exploration).
- Iterative revision model: SFT on incorrect→correct multi-turn trajectories; sequential revisions combined with parallel samples.
- Per-difficulty-bin budget allocation (5 pass@1 quantiles from 2048 samples): easy→revisions, hard→parallel/beam search.

**Key results:** Compute-optimal test-time scaling outperforms best-of-N by up to 4× on MATH for both PRM search (16 vs 64 generations) and revisions (64 vs 256); FLOPs-matched, it beats a ~14× larger pretrained model on easy/medium MATH.

*Evolution:* Building on PRM800k/Lightman, unsupervised PRMs (Math-Shepherd), and Recursive Introspection revision models, this is an early systematic study of test-time compute scaling and its tradeoff against pretraining FLOPs, anticipating the o1-style adaptive-inference "reasoning model" wave.

## 2025

### ACECODER: Acing Coder RL via Automated Test-Case Synthesis
*2025 · code · `code_ACECODER_2502.01718.txt` · arXiv [2502.01718](https://arxiv.org/abs/2502.01718)*

ACECODER supplies reliable reward signals for code via automated large-scale test-case synthesis, enabling both a coding reward model and RL. The RM is a fully fine-tuned instruct coder with a linear head over the last token's hidden state, trained with a Bradley-Terry pairwise loss over test-case pass-rate scores. For RL the authors formalize PPO but adopt Reinforcement++ (Hu, 2025), which drops the value model and computes advantages from RM rewards plus per-token KL, reporting better efficiency/stability than PPO/GRPO. Rewards are either ACECODE-RM-7B or a rule-based binary pass rate (1 iff all tests pass), mirroring Tulu3/DeepSeek-R1 verifiable rewards.

- Bradley-Terry RM loss over pass-rate scores; linear head on last-token hidden state
- RL via Reinforcement++ (no critic); per-token KL advantage
- Rule-based or learned-RM rewards; OpenRLHF, rollout batch 256, 8 samples/Q, LR 5e-7, 8×H100 in 6h
- Ablations favor test-case filtering and a Qwen (vs Llama) coder backbone

**Key results:** ACECODE-RM-7B boosts Llama-3.1-8B-Instruct by +8.4 avg (Best-of-N) and ACECODE-RM-32B by +10.7 across HumanEval/MBPP/BigCodeBench/LiveCodeBench; R1-style RL from Qwen2.5-Coder-7B-Base with rule-based pass rewards gives +25% HumanEval-plus and +6% MBPP-plus in 80 steps (48 H100-hours).

*Evolution:* Building on DeepSeek-R1's verifiable-reward RL-from-base recipe, Tulu3/DeepSeekMath GRPO, and Magicoder's OSS-Instruct synthesis, ACECODER reacts to general reward models failing on code by automating test-case synthesis at scale to supply the missing reward signal.

### Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training
*2025 · code · `code_Agent-R_2501.11425.txt` · arXiv [2501.11425](https://arxiv.org/abs/2501.11425)*

Agent-R is an iterative self-training framework giving ReAct-style LLM agents on-the-fly reflection in POMDP interactive environments. It defines four trajectory types (initial, bad, good, revision), where a revision trajectory splices a bad prefix up to a transition point t' with a good suffix sharing the same MCTS parent, joined by a sampled reflection-signal marker. The key trick is model-guided transition-point selection: the actor itself identifies the first erroneous action within its capability and truncates there, enabling timely revision rather than the Direct-Revision baseline that appends the good path only at trajectory end.

- MCTS with UCT c_uct=0.25, k=8 rollouts, depth 20, 4 candidate actions/node, temp 1.0
- Revision signal rs sampled from 10 templates ('Assistant: [reflection] / Human: OK.')
- Weighted SFT loss (eta=0.2) over good, revision, and general trajectories (Eq. 6)
- Avoids expert supervision; mitigates cold-start and reduces dead loops

**Key results:** Llama-3.1-8B + Agent-R (iter 3) averages 70.71 across WebShop/SciWorld/TextCraft (63.91/70.23/78.00), beating Direct-Revision (62.36), ETO (65.12), and GPT-4o; average revision length drops across iterations (TextCraft 8.3→2.6), indicating earlier error detection.

*Evolution:* Extends MCTS-driven agent training (Agent Q) and self-correction RL (SCoRe), reacting against behavior cloning from all-correct expert trajectories that cannot recover from errors in long-horizon tasks, motivating automatic step-level reflection-data construction and iterative self-play SFT without human/expert supervision.

### CODE I/O: Condensing Reasoning Patterns via Code Input-Output Prediction
*2025 · code · `code_CodeIO_2502.07316.txt` · arXiv [2502.07316](https://arxiv.org/abs/2502.07316)*

CODE I/O reformulates code as an input-output prediction task: given a cleaned function, its textual query, the reference code, and either a specific input or output, the model emits a natural-language Chain-of-Thought predicting the missing side, trained via standard SFT. This decouples structured reasoning (logic-flow planning, state-space search, recursive decomposition) from code syntax while preserving procedural rigor; predictions are verifiable by re-executing code against ground truth. CODE I/O++ adds execution-feedback multi-turn revision: incorrect turn-1 predictions are fed back (wrong-answer notice, executed output, or runtime/exception messages) to DeepSeek-V2.5 for a second attempt, with the final response concatenating both turns plus feedback.

- Predict the missing I/O side via CoT; re-execute to verify correctness
- CODE I/O++ keeps all revised responses (reject-sampling and ground-truth-replacement both underperform)
- Single-turn revision chosen as turn-2 gains diminish
- Outperforms larger instruction-tuning corpora (OpenMathInstruct2 14M, WebInstruct 11.6M)

**Key results:** CODE I/O++ lifts the 14-benchmark average on all four bases vs single-stage instruction tuning (Qwen2.5-Coder-7B 57.7 vs 54.8; LLaMA-3.1-8B 52.1 vs 49.3; DeepSeek-Coder-v2-Lite-16B 53.5 vs 51.6; Gemma-2-27B 61.5 vs 59.5) with balanced, non-regressing gains across symbolic, logic, math, science, and commonsense tasks.

*Evolution:* Builds on the code-pretraining-enhances-reasoning trend (Python-Edu, Code Llama) and execution-prediction learning (scratchpads, CRUXEval, NExT), reacting against math-only reasoning-data scaling and positioning CODE I/O as orthogonal to inference-time scaling and a stronger reasoning base for later RL-based reasoning models.

### RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning
*2025 · code · `code_RAGEN_2504.20073.txt` · arXiv [2504.20073](https://arxiv.org/abs/2504.20073)*

RAGEN implements StarPO (State-Thinking-Actions-Reward Policy Optimization), a general trajectory-level agent-RL framework that, unlike single-turn PPO/GRPO optimizing R(s,a) over prompt-response pairs, maximizes expected cumulative trajectory reward J(θ)=E[R(τ)] over full multi-turn sequences (observations, reasoning, actions, feedback), decomposing π(τ) into token-level likelihoods. Each step emits structured think...answer outputs; rewards are trajectory-level (+10 Sokoban completion, +1 FrozenLake goal, WebShop purchase). It instantiates either PPO (critic + GAE, gamma=lambda=1.0) or critic-free GRPO. The core contribution is StarPO-S, targeting the 'Echo Trap' collapse (reward-std drop, entropy collapse, gradient spikes).

- Uncertainty-based trajectory filtering: keep top-p% (default 25%) highest-reward-variance prompts
- KL-term removal and asymmetric Clip-Higher (eps_high=0.28, eps_low=0.2), both from DAPO
- Response masking and bi-level GAE also evaluated
- 0.5B StarPO-S matches zero-shot GPT-4o and Qwen2.5-72B with ~100× fewer params

**Key results:** 0.5B StarPO-S reaches 20.70% Sokoban / 21.48% FrozenLake, matching zero-shot GPT-4o (27.73%/26.56%) and Qwen2.5-72B (19.53%/23.83%) with ~100× fewer params; keeping the 50% highest-variance rollouts avoids collapse entirely in FrozenLake-PPO, and Bandit generalizes 100% with reasoning vs 81.25% without.

*Evolution:* Extends single-turn RL-for-reasoning (PPO, GRPO from DeepSeek-R1/DeepSeekMath, DAPO's KL-removal and Clip-Higher) into multi-turn stochastic agent training alongside the 2025 agentic-RL wave (Search-R1, WebAgent-R1, VAGEN, ArCHer), surfacing the Echo Trap instability and motivating reasoning-aware reward shaping and turn-aware optimization for long-horizon agents.

### A Self-Improving Coding Agent
*2025 · code · `code_SICA-self-improving_2504.15228.txt` · arXiv [2504.15228](https://arxiv.org/abs/2504.15228)*

SICA is a non-gradient, self-referential agent that edits its own full Python codebase to improve utility, collapsing the meta- and target-agent: the best archive agent becomes the next meta-agent (unlike ADAS, which keeps a fixed meta-agent editing a single DSL forward function). Agent selection uses a utility U = 0.5·p_score + 0.25·(1−min(1, p_cost/$10)) + 0.25·(1−min(1, p_time/300s)), with a 0.5 timeout penalty. The initial agent has file/shell/calculator/archive tools, coding/problem-solver/reasoning sub-agents, and an asynchronous LLM overseer (every 30s) that detects loops and can notify or cancel agents; function calling uses XML tags as stop tokens.

- Self-referential: best archive agent becomes the next meta-agent
- Utility blends score (0.5), cost (0.25), time (0.25) with a 0.5 timeout penalty
- Async LLM overseer (30s) detects loops, notifies or cancels agents
- Claude Sonnet 3.5 v2 for most agents, o3-mini for reasoning; learns SmartEditor, AST/hybrid locators, context summarizers

**Key results:** SWE-Bench Verified (50-Q subset) rises 17%→53% over 15 self-improvement iterations; file-editing synthetic 0.82→0.91 and LiveCodeBench ~0.65→0.71, with per-problem cost ~$1.6-2.7 and total run cost ~$7,000; AIME/GPQA showed minimal gains.

*Evolution:* Extends ADAS (2024) from a fixed-meta-agent/DSL setup to a truly self-referential agent editing its full Python codebase, reacting to hand-crafted orchestrators and prompt-optimization (OPRO, Promptbreeder) and paralleling AlphaEvolve, demonstrating scaffolding-only (non-weight) self-improvement that motivates joint weight-and-system fine-tuning and automated benchmark generation.

### SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
*2025 · code · `code_SWE-RL_2502.18449.txt` · arXiv [2502.18449](https://arxiv.org/abs/2502.18449)*

SWE-RL is the first RL method for real-world software engineering using rule-based rewards over software-evolution (PR) data, avoiding costly execution feedback. Given an issue plus full file contents, the policy reasons in an inner monologue then emits search/replace edits unified into a patch. The reward is R(o)=−1 for malformed output, otherwise compare(patch_pred, patch_gt) via Python's difflib.SequenceMatcher, a continuous 0-1 sequence-similarity score. Policy optimization uses GRPO: G=16 rollouts per prompt, group-normalized advantages A_i=(r_i−mean)/std, a clipped importance-sampling ratio, and a KL penalty β·D_KL(π_θ‖π_ref) to a reference policy. Conditioning on complete files implicitly forces fault localization before repair.

- Continuous difflib sequence-similarity reward beats discrete exact-match (repair 34.8 vs 29.0)
- GRPO with G=16 rollouts, group-normalized advantages, KL penalty to reference
- Full-file context forces fault localization before repair
- Inference uses Agentless Mini scaffold with CodeT dual-execution reranking

**Key results:** Llama3-SWE-RL-70B reaches 41.0% pass@1 on SWE-bench Verified (SOTA for <100B, comparable to GPT-4o) vs 36.2% SFT; repair-only with oracle files 34.8 vs 29.6 (SFT) vs 5.4 (base); OOD generalization: MATH-strict 73.7 vs 63.2 base, CRUXEval-I 71.6 vs 60.5 base.

*Evolution:* Builds on DeepSeek-R1's rule-based-reward RL for math/competitive-code, extending it to real-world software engineering via PR-level data and GRPO without proprietary-teacher distillation (Lingma-SWE-GPT/SWE-Gym/SWE-Fixer), first to show RL on SE artifacts yields emergent 'aha-moment' reasoning generalizing to OOD math/code tasks.

### Beyond Scaling Law: A Data-Efficient Distillation Framework for Reasoning
*2025 · method · `method_DataEfficientDistillation_2508.09883.txt` · arXiv [2508.09883](https://arxiv.org/abs/2508.09883)*

DED (Data-Efficient Distillation) optimizes the Pareto frontier of reasoning distillation under an extremely small corpus. Three core moves: (1) teacher selection via a smoke test—the highest-benchmark LRM is not necessarily the best teacher, so quick distillation trials on the student decide it (QwQ-32B beats stronger DeepSeek-R1/Qwen3); (2) question compression using student pass-rate to drop too-easy items, protecting OOD capability that large corpora degrade; (3) diverse trajectories—for each kept question, sample many CoTs and select the farthest-apart P by Levenshtein distance, mirroring RL's diverse roll-outs. The training objective is standard SFT cross-entropy on these curated long-CoT pairs.

- Teacher chosen by student smoke test, not benchmark ranking (QwQ-32B wins)
- Pass-rate filtering drops too-easy questions to protect OOD capability
- Per-question trajectory diversity selected by Levenshtein distance
- Gains attributed to lower teacher-corpus token entropy (QwQ-32B 0.477 vs DeepSeek-R1 0.705) and smaller PCA latent shift

**Key results:** NTele-R1-32B-Math, distilled from DS-32B on ~0.8k curated examples, scores 81.87% AIME 2024 and 77.29% AIME 2025, beating both teacher models (QwQ-32B 76.25/67.30, DeepSeek-R1 79.2/70) and large-corpus baselines (Light-R1, Skywork-OR1); the mixed NTele-R1-32B doubles Aider pass@2 (12.4→25.8) while raising LCB hard to 30.94.

*Evolution:* Extends the data-efficient SFT line (LIMA, s1, LIMO, Light-R1) and DeepSeek-R1 distillation, reacting against the reasoning 'scaling law' that ever-larger CoT corpora are needed, importing RL ideas (pass-rate filtering, diverse roll-outs) and reframing corpus quality via token entropy and PCA latent shift as cheaper levers than raw scale.

### OpenSIR: Open-Ended Self-Improving Reasoner
*2025 · method · `method_OpenSIR_2511.00602.txt` · arXiv [2511.00602](https://arxiv.org/abs/2511.00602)*

OpenSIR is a verifier-free self-play RL framework where one policy π_θ optimizes two roles (teacher and student). The objective is a GRPO variant with role-specific group-normalised advantages and a KL penalty to the initial model (β=1e-4). Teacher reward = novelty = α·solvability + λ·solution_length + γ·diversity + δ·format (all weights 1.0 except format 0.1), where solvability is a triangular score peaking at the midpoint of [s_min,s_max] and diversity is min cosine distance to the existing pool embeddings; student reward = correctness (majority-vote answer, solutions must end in \boxed{}). Half of the B selected samples train the teacher (highest novelty-variance groups), half the student (highest novelty problems).

- GRPO with role-specific group-normalised advantages + KL to initial model (β=1e-4)
- Teacher novelty = solvability + length + diversity + format; student = majority-vote correctness
- Self-bootstrapped from a single trivial seed; no verifier or human labels
- TRL; 3 H100; AdamW, LR 3e-7, 20 warmup, 200 steps, batch 512, 8 rollouts/prompt, temp 1.0

**Key results:** OpenSIR averages +3.35 math and up to +4.79 general reasoning across four models, beating GRPO baselines trained on >7,000 annotated examples while bootstrapping from a single trivial seed; on DeepSeek-R1-Distill-Llama-8B it improves math 36.31→40.56 and general reasoning 21.62→26.41, with the diversity reward nearly doubling unique concept coverage (3328→5914).

*Evolution:* Reacts to RLVR (DeepSeek-R1, o1) and verifier-free self-play (Absolute Zero, R-Zero, Spiral), diagnosing that prior self-play collapses to familiar concepts on already post-trained models, operationalising the open-endedness thesis (Hughes et al. 2024) by jointly optimising difficulty calibration and embedding-based diversity to motivate verifier-free, diversity-driven self-improvement.

### Revisiting Entropy in Reinforcement Learning for Large Reasoning Models
*2025 · method · `method_PositiveAdvantageReweighting_2511.05993.txt` · arXiv [2511.05993](https://arxiv.org/abs/2511.05993)*

The paper analyzes and mitigates entropy collapse during GRPO-based RLVR, identifying three governing factors: clipping thresholds (Clip-Higher raises ε_high to 0.28; Clip-Lower/Tighter/Free also tested), number of off-policy updates per batch (N_update in {1,2,4}; more updates amplify entropy drift and overfitting), and training-data diversity. Through a gradient analysis of the GRPO objective w.r.t. logits and ablations training only on Adv≥0 vs Adv≤0 tokens, it shows positive-advantage tokens drive collapse: they raise probabilities of high-probability sampled tokens while suppressing low-probability unsampled ones, concentrating the distribution. Building on this, Positive-Advantage Reweighting (Pos-Adv-Reweight) introduces a weight λ on positive-advantage-token losses, realized as Stage-based, Epoch-wise, and Entropy-guided variants.

- Three collapse factors: clipping thresholds, off-policy update count, data diversity
- Gradient analysis shows positive-advantage tokens concentrate the output distribution
- Pos-Adv-Reweight weights positive-advantage-token losses (Stage/Epoch/Entropy-guided)
- Rand-Pos-Clip baseline randomly zeros a subset of positive-advantage gradients; baselines include Ada-Ent-Reg, Clip-Cov, KL-Cov, Entropy-Adv

**Key results:** Qwen2.5-Math-7B + Pos-Adv-Reweight (Entropy-guided) achieves AIME 2024 Avg@64 34.38 / Pass@64 73.33 and the best average Avg@64 (ID 45.66, OOD 19.39), beating Clip-Higher on 6 of 7 benchmarks; ~616 K-means-selected samples match full DAPO-Math-17K (Avg@64 ID 45.13 vs 44.12), showing data diversity, not scale, drives RLVR performance.

*Evolution:* Builds on the 2024-2025 RLVR/GRPO wave (DeepSeek-R1, DAPO's Clip-Higher, Cui et al.'s Clip-Cov/KL-Cov, He et al.'s adaptive entropy regularization) and the covariance-based entropy-collapse theory of Liu/Cui (2025), refining their positive-advantage-token insight into a directly controllable per-token loss weight that motivates finer-grained, entropy-targeted token reweighting for agentic and long-context RL.

### Toward Training Superintelligent Software Agents through Self-Play SWE-RL
*2025 · method · `method_SSR-SelfPlay-SWERL_2512.18552.txt` · arXiv [2512.18552](https://arxiv.org/abs/2512.18552)*

SSR (Self-play SWE-RL) trains one LLM policy in two coupled roles—bug-injector and bug-solver—on the same containerized codebase with Bash/editor tools (CWM scaffold). The injector produces a validated bug artifact; the solver sees only the reversed test-weakening patch as a formal spec and must emit a repair patch. The solver reward r_solve is binary (+1 if all tests pass, −1 otherwise); the injector reward r_inject = −1.0 on consistency failure, −α (α=0.8) for degenerate solve rates s∈{0,1}, and 1−(1+α)s for 0<s<1, pushing the injector toward challenging-yet-solvable bugs near the solver's frontier. Both signals jointly update shared weights.

- Single policy doubles as bug-injector and bug-solver over a shared repo
- r_solve binary ±1; r_inject shaped to drive bugs toward the solver's frontier (α=0.8)
- No human-labeled issues or test suites; both rewards update shared weights
- Async CWM-RL: 512 H100s (64 train, 448 rollout), 131K-token context, 16M-token global batch, group size 8, rollouts discarded after >8 off-policy steps

**Key results:** SSR (CWM-sft 32B base) achieves +10.4 points self-improvement on SWE-bench Verified and +7.8 on SWE-Bench Pro, consistently beating the human-data baseline RL throughout training while using no human-labeled issues or test suites; full self-play beats both repair-only and injection-only ablations.

*Evolution:* Extends the 2025 agentic-SWE-RL wave (SWE-RL, DeepSWE, CWM) and corpus-grounded self-play (SPICE, Absolute Zero, R-Zero) by removing human-curated issues/tests entirely, framing bug injection/repair as a self-play game over real repositories—an early step toward open-ended self-improving software agents that flags unresolved scaling instability and reward-hacking risks.

### SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
*2025 · method · `method_SWE-Bench-Pro_2509.16941.txt` · arXiv [2509.16941](https://arxiv.org/abs/2509.16941)*

The core contribution is a benchmark-construction and evaluation methodology for agentic software engineering, not a training algorithm. Three pillars: (1) contamination-resistant sourcing via GPL copyleft repos plus purchased commercial startup codebases; (2) a three-stage human-in-the-loop augmentation/verification workflow that turns raw commit pairs into self-contained tasks, producing a rewritten problem statement, human-authored requirements grounded in unit tests, and an optional interface specification listing expected class/function signatures to suppress unit-test false negatives; (3) environment creation via expert-built Docker images, automated flaky-test validation (gold tests must pass repeatedly), and human review dropping irrelevant or overly broad tests. Evaluation uses a unified SWE-Agent scaffold, identical prompts, max 50 turns, a $2 cost cap, and vLLM hosting on 8×H100.

- Contamination-resistant sourcing: GPL repos + purchased commercial codebases
- Three-stage human-in-the-loop augmentation/verification with unit-test-grounded requirements
- Expert-built Docker images with flaky-test validation and human review
- Unified SWE-Agent scaffold, max 50 turns, $2 cap, vLLM on 8×H100; Pass@1 metric; GPT-5 LLM-as-judge failure taxonomy

**Key results:** Claude Sonnet 4.5 tops the public set at 43.6% Pass@1 (N=731); the best models stay below 20% on the commercial set (Opus 4.1 17.8%, N=276); top models score ~23% on SWE-Bench Pro versus >70% on SWE-Bench Verified, and removing human augmentations collapses performance (GPT-5 25.9%→8.4%, Opus 4.1 22.7%→8.2%).

*Evolution:* Extends SWE-Bench/SWE-Bench Verified (Jimenez et al., 2024) and the SWE-Agent line, reacting to benchmark contamination from permissively-licensed repos and saturation of SWE-Bench Verified (>70%) partly via trivial 1-2 line tasks, offering a harder, contamination-resistant, enterprise-realistic yardstick that complements SWE-Bench+/LiveBench and motivates next-generation agentic RL such as Agent-RLVR (Da et al., 2025).

### The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models
*2025 · method · `method_ValleyOfCodeReasoning_2510.06101.txt` · arXiv [2510.06101](https://arxiv.org/abs/2510.06101)*

The core contribution is an empirical analysis of SFT distillation dynamics for code reasoning, not a new algorithm. Two small non-reasoning instruct models—Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct (neither emits <think> tags nor has a dedicated think token)—are fine-tuned on CoT traces distilled from strong reasoning teachers. The paper formalizes three questions: (RQ-1) whether performance scales monotonically with data quantity, (RQ-2) whether teacher-response correctness matters, and (RQ-3) whether input-problem difficulty matters. Training uses torchtune on 8×H100 with max sequence length 32,768. The central finding is the 'valley of code reasoning': performance first drops by more than half at 1K, then rises sharper-than-log-linearly to 30K.

- Non-reasoning instruct bases fine-tuned on distilled long-CoT traces via torchtune (8×H100, 32K ctx)
- Three RQs: data-quantity scaling, teacher-correctness, problem-difficulty effects
- 'Valley': performance drops >50% at 1K then rises sharper-than-log-linearly to 30K
- Auxiliary metrics (completion rate, <think>-tag rate) separate structural imitation from genuine capability

**Key results:** Qwen2.5-7B-Instruct LCB Pass@1: 0.126 baseline → 0.055 at 1K (valley) → 0.188 at 10K → 0.264 at 30K; Llama3.1-8B-Instruct reaches 0.299 at 30K (doubling its 10K score); easy/medium questions beat hard (+41% vs +7%), and correct vs incorrect teacher responses yield identical gains (~50%).

*Evolution:* Builds on data-efficient reasoning distillation (s1, LIMO, Li et al.'s 'structure not content') and large-scale coding distillation (OpenThoughts, OCR, rStar-Coder), reacting to their focus on data endpoints rather than intermediate training dynamics, exposing the non-monotonic 'valley' in small models and motivating study of medium-high and high (>100K) data regimes and stage-aware data curation.

### λ-GRPO: Unifying the GRPO Frameworks with Learnable Token Preferences
*2025 · method · `method_lambda-GRPO_2510.06870.txt` · arXiv [2510.06870](https://arxiv.org/abs/2510.06870)*

λ-GRPO unifies GRPO, DAPO, and Dr. GRPO under a single 'Unified Token Preference' objective where all differences reduce to a per-output weighting function f(o_i): GRPO uses f=μ/|o_i| (downweights long responses), DAPO uses f=1 (uniform token-level), Dr. GRPO uses f=μ/P. λ-GRPO replaces these fixed heuristics with a learnable, length-adaptive weight: it standardizes each group's response lengths h_i=1+r·z_i (z_i=(o_i−μ)/σ, r=1/9) then applies g_i=h_i^λ. The scalar λ (a trainable GPU tensor, init 0, optimized by SGD with lr 1e-1, no weight decay) is the token-preference knob—λ>0 favors long, λ<0 favors short, λ=0 neutral; weights are softmax-normalized and rescaled by G.

- Unifies GRPO/DAPO/Dr. GRPO as special cases of a per-output weighting f(o_i)
- Learnable λ (GPU tensor, SGD lr 1e-1, no weight decay) replaces fixed normalization heuristics
- h_i=1+r·z_i length standardization; g_i=h_i^λ softmax-normalized to sum to G
- Jointly optimized with model parameters at no extra compute, countering GRPO's length bias

**Key results:** Qwen2.5-1.5B/3B/7B reach average accuracy 37.8/43.8/53.5 over 8 math benchmarks, +1.9/+1.0/+1.7% over vanilla GRPO and +1.3/+1.0/+1.6% over DAPO; λ-GRPO also maintains higher token-level entropy than DAPO (steps 80–160) at similar response length, with no data or compute changes.

*Evolution:* Builds on GRPO (DeepSeek-R1-Zero) and the length-bias fixes DAPO/Dr. GRPO, recasting their fixed token-aggregation heuristics as special cases of a learnable token-preference framework; within the 2025 RLVR wave of GRPO variants (GMPO, GSPO, GiGPO, KRPO) it motivates adaptive, data-driven weighting rather than hand-designed normalization.

### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
*2025 · report · `report_DeepSeek-R1_2501.12948.txt` · arXiv [2501.12948](https://arxiv.org/abs/2501.12948)*

DeepSeek-R1's core algorithm is Group Relative Policy Optimization (GRPO), a PPO simplification that drops the value/critic model: per question it samples G outputs from the old policy, computes group-relative advantages A_i=(r_i−mean)/std, and maximizes a clipped-ratio objective with an unbiased KL term to a reference policy added to the loss (not as a dense per-token reward as in PPO), periodically replacing the reference with the latest policy. Reward is accuracy (answer match / compiler tests) + format (think/answer tags) for reasoning, plus a language-consistency reward in R1 stage 1 and helpful/safety reward models for general data in stage 2.

- Hyperparameters: lr 3e-6, KL coeff 0.001, large clip ε=10, 16 samples/question, batch 512, 8192 rollouts split into 16 minibatches for one inner epoch
- Infra: vLLM rollouts with expert parallelism + MTP self-speculative decoding, DualPipe, and VRAM offload between modules
- PRM and MCTS were tried and abandoned (reward hacking, value-model difficulty)

**Key results:** DeepSeek-R1 (671B MoE / 37B active) scores 79.8% pass@1 on AIME 2024 (86.7% cons@16) and 97.3 on MATH-500, matching OpenAI o1-1217 (79.2%) and far exceeding DeepSeek-V3 (39.2); Codeforces 96.3rd percentile (rating 2029); R1-Distill-Qwen-32B hits 72.6 AIME / 94.3 MATH, beating RL-only Qwen2.5-32B-Zero.

*Evolution:* Builds on GRPO (DeepSeekMath, 2024) and the outcome-only RL/STaR line, reacting against SFT-first RLHF by showing pure RL on a strong base can elicit emergent long-CoT reasoning (the "aha moment") without human traces; its open R1-Zero/R1 weights and distill recipes catalyzed the open-source reasoning-RL wave (Open-R1, SkyRL, DAPO-style variants).

### DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
*2025 · report · `report_DeepSeek-V3.2_2512.02556.txt` · arXiv [2512.02556](https://arxiv.org/abs/2512.02556)*

DeepSeek-V3.2 pairs DeepSeek Sparse Attention (DSA) with a stabilized, scalable GRPO. DSA combines a lightning indexer (small-head, FP8, ReLU index scores) with fine-grained top-k token selection, cutting core attention from O(L²) to O(Lk) under MLA in MQA mode. The RL side augments GRPO with four stabilizers: an unbiased KL estimate correcting the K3 estimator via the current/old-policy importance ratio; Off-Policy Sequence Masking that zeroes negative-advantage sequences with large π_old−π_θ divergence; Keep Routing that freezes MoE expert routing paths from sampling during training; and Keep Sampling Mask that reuses top-p/top-k truncation masks to keep the two policies' action spaces aligned.

- Rewards: rule-based outcome reward + length penalty + language-consistency reward for reasoning/agent tasks; per-prompt generative reward model for general tasks
- A thinking-context-management rule and cold-start integrate reasoning into tool use
- DSA instantiated under MLA (MQA mode) for long-context efficiency

**Key results:** DeepSeek-V3.2-Speciale reaches AIME 2025 96.0 Pass@1, HMMT Feb 99.2, Codeforces rating 2701, with gold medals at IMO/IOI/ICPC-WF/CMO 2025; V3.2-Thinking matches GPT-5-High on reasoning (AIME 93.1 vs 94.6) with SWE-Verified 73.1 and Terminal Bench 2.0 46.4; DSA cuts 128K decoding cost from $2.4M to $0.6M per million tokens vs V3.1-Terminus.

*Evolution:* Builds on DeepSeek-R1's RL-for-reasoning, V3/V3.1's MoE+MLA, DeepSeekMath's GRPO, and native sparse attention, reacting to the open-vs-closed gap by scaling post-training RL compute beyond 10% of pre-training and synthesizing agentic environments at scale, showing scaled RL plus synthesized agent data can approach frontier parity (GPT-5, Gemini-3.0-Pro).

### GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models
*2025 · report · `report_GLM-4.5_2508.06471.txt` · arXiv [2508.06471](https://arxiv.org/abs/2508.06471)*

GLM-4.5 unifies Expert Model Iteration and self-distillation into a hybrid-reasoning generalist trained with GRPO-style RL (the KL term dropped); the group-wise objective L(θ)=E_x[mean_i(r(x,y_i)−r̄(x))] optimizes only model-generated tokens, with environment feedback excluded from the loss. Key tricks include a difficulty-based curriculum to keep reward variance healthy, dynamic sampling temperature raised (capped at ≤1% validation drop) once rollout reward stabilizes to maintain exploration, token-weighted mean loss for code RL (faster convergence, less length bias than sequence-mean), and iterative self-distillation where RL-improved outputs replace cold-start SFT data before RL resumes.

- Science RL restricted to expert-verified multiple-choice data; XML-tag function-call template cuts character escaping
- Outcome supervision with a process action-format penalty (zero reward on malformed tool calls); step-wise rule-based RL plus end-to-end multi-turn function-calling RL
- Infra: open-source Slime (Megatron training + SGLang rollout + Data Buffer), colocated synchronous and disaggregated asynchronous (Ray-scheduled) modes, online block-wise FP8 quantization, high-concurrency Docker runtime for agent environments

**Key results:** GLM-4.5 (355B/32B-active MoE) scores 91.0% AIME 24, 70.1% TAU-Bench, and 64.2% SWE-bench Verified, ranking 3rd overall and 2nd on agentic benchmarks; GLM-4.5-Air (106B/12B) is competitive with Qwen3-235B-A22B and MiniMax-M1 (6th overall), with both on the SWE-bench-vs-parameters Pareto frontier.

*Evolution:* Builds on the 2024–2025 open MoE and reasoning-RL wave (DeepSeek-V3/R1, Kimi K2, Qwen3, GRPO from DeepSeekMath) and Nemotron-CC-style data bucketing, distinguishing itself by unifying agentic, reasoning, and coding via expert-iteration + self-distillation plus hybrid thinking/non-thinking modes, and motivating standardized agent-RL infrastructure (Slime) for long-horizon tool-using agents.

### Gemma 3 Technical Report
*2025 · report · `report_Gemma3_2503.19786.txt` · arXiv [2503.19786](https://arxiv.org/abs/2503.19786)*

Gemma 3's post-training is a combined distillation-then-RL recipe: on-policy distillation from a large IT teacher (256 teacher logits/token, cross-entropy target) is followed by an RL phase integrating BOND (best-of-n distillation alignment), WARM (weight-averaged reward models for reward stability), and WARP (weight-averaged rewarded policies). Reward signals span human-feedback RMs, code-execution feedback (Rlef), and verifiable ground-truth math rewards (DeepSeek-R1, Tulu 3). Architecturally it swaps Gemma 2's soft-capping for QK-norm and a 5:1 local:global attention interleave (1024-token sliding window) to curb KV-cache growth at 128K context, with a frozen SigLIP vision encoder plus Pan & Scan for multimodality.

- **Distillation:** on-policy logit distillation (Agarwal et al. ideas) with 256 sampled teacher logits per token.
- **RL stack:** BOND alignment + WARM reward averaging + WARP policy averaging, fed by HF/code/math rewards.
- **Architecture tricks:** QK-norm and 5:1 local:global sliding-window attention for 128K context.
- **Goal:** lightweight, long-context, multimodal open model competitive with far larger systems.

**Key results:** Gemma-3-27B-IT scores Chatbot Arena Elo 1338 (~rank 9), MATH 89.0, GSM8K 95.9, HumanEval 87.8, MMLU-Pro 67.5, IFEval 90.4, comparable to Gemini-1.5-Pro and ahead of larger open models.

*Evolution:* Folds the 2024 RL-alignment toolkit (BOND/WARM/WARP) plus verifiable-reward signals (R1, Tulu 3) into distillation-plus-RL post-training for lightweight multimodal open models.

### Kimi K1.5: Scaling Reinforcement Learning with LLMs
*2025 · report · `report_Kimi-K1.5_2501.12599.txt` · arXiv [2501.12599](https://arxiv.org/abs/2501.12599)*

K1.5's core method is a long-CoT RL framework formalized as online policy mirror descent: each iteration i maximizes E[r(x,y,y*)] − τ·KL(π_θ‖π_θ_i) with a closed-form solution, yielding a policy-gradient-like surrogate with a mean-reward baseline, but responses are sampled off-policy from π_θ_i with an L2 regularizer and the optimizer resets each iteration. It deliberately drops value networks, MCTS, and process reward models, using long context to substitute for explicit search. A length penalty (warmed up) rewards short correct and penalizes long incorrect answers. Partial rollouts cap per-rollout token budget, save unfinished segments to a replay buffer, and resume them asynchronously with repeat detection.

- **Algorithm:** online policy mirror descent with KL-regularized off-policy sampling and per-iteration optimizer reset.
- **Ablations:** no value net, no MCTS, no PRM—context length substitutes for search.
- **Tricks:** warmed-up length penalty; partial rollouts with replay buffer and async resumption.
- **Infra:** Megatron+vLLM sidecars, Mooncake RDMA weight transfer (<1 min train→infer, ~10 s back), crun-based code sandbox.

**Key results:** Kimi k1.5 long-CoT: 77.5 AIME 2024 Pass@1, 96.2 MATH-500, 94th percentile Codeforces, 74.9 MathVista, matching OpenAI o1.

*Evolution:* Formalizes RL as online mirror descent and argues context-length scaling suffices without MCTS, value functions, or PRMs, motivating efficient long-context RL infra and long2short distillation.

### Kimi K2: Open Agentic Intelligence (Technical Report of Kimi K2)
*2025 · report · `report_Kimi-K2_2507.20534.txt` · arXiv [2507.20534](https://arxiv.org/abs/2507.20534)*

K2's two method contributions are (1) the MuonClip optimizer—Muon + weight decay + consistent RMS matching + QK-Clip—which enables zero-loss-spike token-efficient training; QK-Clip rescales Q/K projection weights per head when the max attention logit exceeds τ=100 (for MLA scaling qC/kC by √γ_h, qR by γ_h, leaving shared kR untouched) and self-deactivates after stabilization. (2) A general RL framework combining RLVR with a Self-Critique Rubric Reward: a K2 critic does pairwise evaluation against core/prescriptive/human rubrics, bootstrapped in SFT and refined by distilling verifiable-reward signals. The RL algorithm reuses K1.5's policy optimization with an importance-sampling-regularized squared objective plus Budget Control (per-sample max-token budget with truncation penalty), PTX loss (auxiliary pretraining loss against forgetting), and Temperature Decay.

- **Optimizer:** MuonClip (Muon + RMS matching + QK-Clip at τ=100) for stable spike-free training.
- **Reward:** RLVR + Self-Critique Rubric Reward via a bootstrapped, reward-distilled critic.
- **RL additions:** Budget Control, PTX anti-forgetting loss, Temperature Decay on top of K1.5's objective.
- **Infra:** colocated train/inference engines, <30s parameter resharding, partial rollout for long-horizon agents.

**Key results:** Kimi-K2-Instruct (1.04T MoE, 32B activated) scores 65.8 SWE-bench Verified, 76.5 ACEBench, 49.5 AIME 2025, 75.1 GPQA-Diamond, #1 open-source on LMSYS Arena.

*Evolution:* Pushes token-efficient Muon training and agentic RL with self-critic-based alignment to trillion-parameter scale, building on DeepSeek-V3 MLA-MoE and K1.5 RL.

### Llama-Nemotron: Efficient Reasoning Models
*2025 · report · `report_Llama-Nemotron_2505.00949.txt` · arXiv [2505.00949](https://arxiv.org/abs/2505.00949)*

LN post-training combines distillation, SFT, and RLVR. SFT uses token-level cross-entropy over instruction data with a "detailed thinking on/off" system prompt enabling a runtime reasoning toggle. LN-Ultra reasoning RL uses GRPO (rollout batch 72 prompts × 16 responses, temp=1/top_p=1, global batch 576, 2 updates/rollout, ~140k H100-hours) with accuracy rewards (Llama-3.3-70B-Instruct judge) plus format rewards enforcing <think> tags; difficulty filtering discards high pass-rate prompts and a progressive easy-to-hard curriculum schedules batches. Instruction-following RL uses RLOO (<120 steps, batch 128) with an IF verifier. RLHF uses iterative online RPO for LN-Super (maximizing Llama-3.1-Nemotron-70B-Reward over HelpSteer2; LR 4e-7, KL 1e-5, reward scale 3.0, batch 64, 500 steps, 2 iterations) and GRPO for LN-Ultra; LN-Nano uses two rounds of offline RPO with on-policy data.

- **Reasoning RL:** GRPO with accuracy (LLM-judge) + format rewards, difficulty filtering, progressive curriculum.
- **IF RL:** RLOO with an instruction-following verifier.
- **RLHF:** iterative online RPO (LN-Super) / GRPO (LN-Ultra) / offline RPO (LN-Nano).
- **Infra:** NeMo-Aligner + vLLM generation + Megatron-LM co-located, FP8 generation (1.8x), /dev/shm weight hand-off with vLLM sleep mode.

**Key results:** LN-Ultra (253B) reaches GPQA-Diamond 76.0% and AIME24 80.8%, beating DeepSeek-R1 while fitting on one 8xH100 node; GRPO raises GPQA-D from 66.4 (SFT) to 76.0.

*Evolution:* Builds on DeepSeek-R1's distill-then-RLVR recipe and adds NAS/FFN-Fusion efficiency plus a user-facing reasoning toggle for controllable, efficient reasoning.

### Qwen3 Technical Report
*2025 · report · `report_Qwen3_2505.09388.txt` · arXiv [2505.09388](https://arxiv.org/abs/2505.09388)*

Qwen3's contribution is a single model unifying thinking/non-thinking modes via a chat template with /think and /no_think flags (empty thinking block for non-thinking), plus a thinking-budget mechanism that halts reasoning at a user threshold via an injected stop-thinking instruction. Reasoning RL uses GRPO with large batches, many rollouts/query, off-policy sampling for sample efficiency, and entropy control (kept steady/increasing) for stability. General RL covers 20+ tasks (instruction/format following, preference alignment, agent tool-use with real environment feedback, RAG) using three reward types: rule-based, model-based scoring against a reference answer (Qwen2.5-72B-Instruct), and a trained reference-free reward model. Strong-to-Weak Distillation first does off-policy logit distillation from teacher /think and /no_think outputs, then on-policy distillation where the student samples and is fine-tuned by minimizing KL to teacher (Qwen3-32B or 235B-A22B) logits.

- **Unified modes:** chat-template /think, /no_think flags + thinking-budget stop instruction.
- **Reasoning RL:** GRPO with large batches, off-policy sampling, entropy control.
- **General RL:** 20+ tasks with rule-based, model-based, and trained reference-free rewards.
- **Distillation:** off-policy logit distillation then on-policy student-sampled KL distillation.

**Key results:** Qwen3-235B-A22B (Thinking) scores 85.7 AIME'24, 81.5 AIME'25, 2056 CodeForces, beating DeepSeek-R1 on 17/23 benchmarks; on-policy distillation beats RL on Qwen3-8B (74.4 vs 67.6 AIME'24) at ~1/10 GPU hours.

*Evolution:* Unifies chat and reasoning modes with a controllable thinking budget; its Strong-to-Weak distillation motivates later efficient small-model reasoning and agent-RL research.

### Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning
*2025 · report · `report_Seed1.5-Thinking_2504.13914.txt` · arXiv [2504.13914](https://arxiv.org/abs/2504.13914)*

Seed1.5-Thinking targets stable large-scale long-CoT RL via two frameworks: VAPO (actor-critic, SOTA) and DAPO (policy-gradient, critic-free). Reward modeling uses Seed-Verifier (principle-based LLM judge returning YES/NO on a question/reference/model-answer triplet) and Seed-Thinking-Verifier (a verifier trained as a verifiable task emitting a reasoning path, fixing reward hacking and corner cases; 99.3% vs 82.7% test accuracy); non-verifiable tasks use a pairwise generative RM. PPO enhancements include Value-Pretraining (Monte-Carlo return from π_sft), Decoupled-GAE (λ_value=1.0, λ_policy=0.95), Length-adaptive GAE (λ_policy=1−α/l), Dynamic Sampling (drop accuracy 0/1 prompts), Clip-Higher (decoupled ε_low/ε_high), Token-level Loss, and Positive-Example LM loss (L=L_PPO+μ·L_NLL). Online Data Distribution Adaptation rebalances the prompt distribution to cut cross-domain interference.

- **RL frameworks:** VAPO (actor-critic) and DAPO (critic-free) for long-CoT.
- **Verifiers:** Seed-Verifier (principle judge) + Seed-Thinking-Verifier (reasoning-path verifier, 99.3% accuracy).
- **PPO tricks:** Value-Pretraining, Decoupled/Length-adaptive GAE, Dynamic Sampling, Clip-Higher, Token-level Loss, Positive-Example LM loss.
- **Infra:** HybridFlow/Ray, Streaming Rollout System (3x faster), hybrid co-located engine, FP8 rollout, ByteCheckpoint.

**Key results:** Seed1.5-Thinking (20B-active/200B-total MoE) scores 86.7 AIME 2024 (matching o3-mini-high), 55.0 Codeforces avg@8, 77.3 GPQA, with +8.0% win rate over DeepSeek R1 on non-reasoning tasks.

*Evolution:* Builds on the o1/R1 test-time-scaling wave and the authors' VAPO/DAPO work to make long-CoT RL stable and reproducible at MoE scale, emphasizing reward verification and RL infrastructure.

### AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning
*2025 · rl · `rl_AceReason-Nemotron_2505.16400.txt` · arXiv [2505.16400](https://arxiv.org/abs/2505.16400)*

AceReason-Nemotron's method is large-scale GRPO (Shao et al. 2024), chosen over PPO for not needing a value model, using DAPO's token-level policy-gradient loss variant. Per prompt, G rollouts are scored by a rule-based verifier: math uses a sympy/antlr4 checker on the boxed answer after \boxed; code uses a local sandbox executing the extracted ```python``` block against all test cases, binary reward only if all pass within the time limit. Strict on-policy training—one gradient update per rollout batch—keeps the importance weight 1 and zeros the KL term (β=0), reducing GRPO to REINFORCE with group-normalized rewards and preventing the entropy collapse seen with 2–4 updates. Math uses batch 128, G=8 (8K) or 16, lr 1e-6, AdamW; code uses batch 128, lr 5e-6. Implemented on veRL with vLLM v0.7.3 on 128 H100 GPUs.

- **Algorithm:** GRPO with DAPO token-level loss, on-policy (1 update/batch), KL off (β=0) → REINFORCE with group baseline.
- **Verifiers:** sympy/antlr4 boxed-answer checker (math); sandboxed all-test-case execution (code, binary).
- **Hyperparams:** batch 128; math G=8/16, lr 1e-6; code lr 5e-6; veRL + vLLM on 128 H100s.
- **Tricks:** stage-wise length extension, hard-prompt curriculum, on-policy stabilization.

**Key results:** AceReason-Nemotron-14B reaches 78.6/67.4 avg@64 on AIME24/25 and 61.1/54.9 avg@8 on LiveCodeBench v5/v6, surpassing DeepSeek-R1-Distill-Qwen-32B and SOTA distillation models.

*Evolution:* Counters the R1/Llama-Nemotron claim that distillation beats RL for small/mid models, motivating sequential multi-domain RL curricula and RL-on-distilled-model recipes.

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale
*2025 · rl · `rl_DAPO_2503.14476.txt` · arXiv [2503.14476](https://arxiv.org/abs/2503.14476)*

DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) is a GRPO-derived RL algorithm for long-CoT reasoning built on the verl framework, removing the KL penalty (policy diverges far from base during long-CoT) and using rule-based verifiable rewards. Four techniques: (1) Clip-Higher decouples PPO clip bounds (ε_low=0.2, ε_high=0.28) so low-probability exploration tokens can rise, counteracting entropy collapse; (2) Dynamic Sampling over-samples and filters prompts with group accuracy 0 or 1 (zero advantage/gradient) so every batch has effective-gradient prompts; (3) Token-Level Policy Gradient Loss replaces GRPO's sample-level mean (which under-weights long sequences) with a token-level sum normalized by total tokens, giving longer sequences proportional influence and penalizing gibberish/repetition; (4) Overlong Reward Shaping masks loss of truncated samples (Overlong Filtering) and adds Soft Overlong Punishment, a linear length penalty over a cache window before the 20,480-token hard cap.

- **Clip-Higher:** decoupled ε_low=0.2/ε_high=0.28 to counter entropy collapse.
- **Dynamic Sampling:** filter 0/1 group-accuracy prompts (zero advantage).
- **Token-level loss:** normalize by total tokens, not per-response mean, to weight long sequences.
- **Overlong shaping:** Overlong Filtering + Soft linear length penalty before 20,480-token cap.

**Key results:** DAPO on Qwen2.5-32B base reaches 50 on AIME 2024 (avg@32), surpassing DeepSeek-R1-Zero-Qwen-32B's 47 using 50% of the steps; the four techniques add ~20 points over naive GRPO.

*Evolution:* Open-sources the algorithm, verl code, and DAPO-Math-17K dataset to democratize large-scale RLVR for long-CoT reasoning, reacting to R1's withheld training details.

### Understanding R1-Zero-Like Training: A Critical Perspective
*2025 · rl · `rl_DrGRPO_2503.20783.txt` · arXiv [2503.20783](https://arxiv.org/abs/2503.20783)*

Dr. GRPO ("GRPO Done Right") fixes two biases in standard GRPO. Standard GRPO sets per-token advantage A_i,t=(R(q,o_i)−mean(R))/std(R) and averages the PPO-clipped surrogate over |o_i| tokens. The authors identify (1) response-level length bias from dividing by |o_i|—for correct (positive-advantage) responses shorter answers get larger gradients, while incorrect ones have longer answers penalized less, inflating incorrect-response length; and (2) question-level difficulty bias from dividing by std(R), up-weighting near-trivial/near-impossible questions. Dr. GRPO removes both the |o_i| length normalization and the std(R) normalization, recovering the PPO objective with Monte-Carlo return and an unbiased group-mean baseline (equivalent to REINFORCE-Leave-One-Out up to an LR-absorbable scale). They also fix biased open-source PPO losses (trl, verl, OpenRLHF, SimpleRL-Zero, Open-Reasoner-Zero) that normalize by response length. KL is dropped (β=0) since rule-based verifiers avoid distributional-shift concerns, saving reference-model memory/compute.

- **Bias 1:** length normalization 1/|o_i| inflates incorrect-response length—removed.
- **Bias 2:** std(R) normalization up-weights trivial/impossible questions—removed.
- **Result:** recovers PPO with MC return + group-mean baseline (≈ REINFORCE-Leave-One-Out).
- **Implementation:** sparse 1/0 Math-Verify outcome reward, Oat framework, actor-learner collocation.

**Key results:** Oat-Zero-7B (Qwen2.5-Math-7B + Dr. GRPO on MATH L3-5) achieves 43.3% AIME 2024 and 51.4% avg over 5 math benchmarks, a 7B R1-Zero-like SOTA, in 27h on 8×A100.

*Evolution:* Reacts critically to the "Aha moment"/long-CoT emergence narrative, showing pretraining and GRPO biases inflate response length; prefigures DAPO-style correctives and motivates scrutiny of base-model biases.

### LIMO: Less is More for Reasoning
*2025 · rl · `rl_LIMO_2502.03387.txt` · arXiv [2502.03387](https://arxiv.org/abs/2502.03387)*

LIMO contributes the Less-Is-More Reasoning Hypothesis and a data-efficient elicitation pipeline rather than a new RL algorithm: in models whose domain knowledge is already encoded during pretraining, sophisticated reasoning emerges from a few strategically designed long-CoT demonstrations acting as "cognitive templates." The elicitation threshold is governed by (1) completeness of pretrained knowledge and (2) effectiveness of minimal exemplars in leveraging inference-time compute. Technically the method is a curated 800-sample dataset plus standard SFT; the "trick" is difficulty-gated question selection (short-CoT reject + 32-attempt success-rate band) and a four-axis rule-based chain scoring favoring elaborated, self-verifying, exploratory, granular reasoning. It directly challenges data-intensive SFT (NuminaMath, OpenThoughts) and the view that SFT merely memorizes rather than generalizes.

- **Hypothesis:** minimal high-quality long-CoT exemplars elicit reasoning latent in pretrained knowledge.
- **Method:** standard SFT on 800 curated samples (not a new RL algorithm).
- **Selection:** difficulty-gated (reject short-CoT solvable, 32-attempt success-rate band).
- **Scoring:** four-axis rule-based chain scoring (elaborated, self-verifying, exploratory, granular).

**Key results:** LIMO (Qwen2.5-32B-Instruct + 800 SFT samples): 63.3% AIME24 (vs o1-preview 44.6%, NuminaMath-100k SFT 6.5%) and 95.6% MATH500, +45.8% absolute OOD gain, using ~1% of prior approaches' data.

*Evolution:* Extends LIMA's "less is more" alignment lesson from instruction-following to math reasoning, complementing RLVR by showing minimal high-quality SFT can rival far larger pipelines.

### Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning
*2025 · rl · `rl_Logic-RL_2502.14768.txt` · arXiv [2502.14768](https://arxiv.org/abs/2502.14768)*

Logic-RL is a rule-based RL post-training framework using a modified REINFORCE++ (preferred over GRPO and PPO; PPO is 138% slower, GRPO weakest). Rewards are purely rule-based: a Format Reward (regex-enforced think/answer tags, each appearing exactly once in order; +1 correct, −1 incorrect) that blocks shortcuts like skipping reasoning or stuffing text into the answer tag, and an Answer Reward (+2 full match, −1.5 partial, −2 unparseable) against the deterministic ground truth. A system prompt plus a leading think tag eases base-model compliance. Two REINFORCE++ modifications follow DeepSeek-Math/GRPO: (1) KL divergence moved from per-token reward into the loss (KL loss), and (2) GRPO's unbiased, always-non-negative KL estimator. Train batch 8, rollout N 8, KL coef 0.001, max response 4096, γ=1. The 7B model spontaneously develops reflection, verification, backtracking, and formula application absent from the corpus.

- **Algorithm:** modified REINFORCE++ (chosen over GRPO/PPO) with KL moved into the loss.
- **Rewards:** Format Reward (regex, ±1) + Answer Reward (+2/−1.5/−2) against ground truth.
- **Tricks:** system prompt + leading think tag for base-model compliance; GRPO's non-negative KL estimator.
- **Emergence:** reflection, verification, backtracking arise spontaneously on ~5K K&K puzzles.

**Key results:** Qwen2.5-7B-Instruct-1M + Logic-RL on ~5K K&K puzzles: K&K accuracy 0.19→0.89, AIME +125%, AMC +38%; REINFORCE++ beats GRPO and PPO is 138% slower.

*Evolution:* An early R1-replication study using a controlled synthetic logic corpus to dissect training dynamics—finding no discrete aha moment and that curriculum/cold-start matter little—motivating long-to-short CoT compression and KL-constraint relaxation.

### Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
*2025 · rl · `rl_Open-Reasoner-Zero_2503.24290.txt` · arXiv [2503.24290](https://arxiv.org/abs/2503.24290)*

ORZ's core contribution is a minimalist, scalable RL recipe: vanilla PPO with GAE(λ=1, γ=1), no KL regularization, and a simple rule-based reward. Diverging from DeepSeek-R1-Zero's GRPO, the authors choose PPO for its learned critic, enabling accurate token-level value estimation and credit assignment that devalues repetitive/degenerate patterns GRPO cannot penalize. With γ=λ=1, the advantage collapses to Â_t=R−V_φ(s_t) and the value target to R. The reward is binary (1 for exact match of the answer-tag content, else 0), with no format reward. Clip ε=0.2; AdamW (β=[0.9,0.95]); LRs 1e-6 (policy) and 5e-6 (critic) with 50-step linear warmup; 128 prompts/step, 64 rollouts/prompt at temp=top_p=1.0; batch-level advantage normalization; no entropy bonus.

- **Algorithm:** vanilla PPO with GAE(λ=1, γ=1), no KL, learned critic for token-level credit assignment.
- **Advantage:** collapses to Â_t=R−V_φ(s_t) with γ=λ=1; batch-level normalization.
- **Reward:** binary exact-match on answer tag (no format reward).
- **Hyperparams:** clip 0.2, AdamW, LR 1e-6 policy/5e-6 critic, 128×64 rollouts at temp=top_p=1.0.

**Key results:** ORZ-32B matches/exceeds DeepSeek-R1-Zero-Qwen-32B on AIME2024 (48.1 vs 47.0), MATH500 (92.2 vs 91.6), GPQA (55.5 vs 55.0) using 1/10 the steps; ORZ-R1-Distill-Qwen-14B reaches AIME2024 75.2.

*Evolution:* The first fully open-source large-scale reasoning-RL framework (code, data, weights 0.5B–32B), reverting from GRPO to vanilla PPO with a learned critic for better credit assignment.

### Process Reinforcement through Implicit Rewards
*2025 · rl · `rl_PRIME-ProcessReinforcement_2502.01456.txt` · arXiv [2502.01456](https://arxiv.org/abs/2502.01456)*

PRIME (Process Reinforcement through Implicit Rewards) uses implicit process reward modeling: an ORM trained with only outcome labels is repurposed as a token-level PRM, with reward r_φ(y)=β·log(π_φ(y)/π_ref(y)) (both causal LMs) and per-token process reward r_φ(y_t)=β·log(π_φ(y_t|y<t)/π_ref(y_t|y<t)). Each iteration samples K responses per prompt, grades them with a rule-based outcome verifier, updates the Implicit PRM online via cross-entropy on (x,y,r_o), derives dense token rewards, computes advantages, and updates the policy with a PPO clip surrogate. Advantages combine separately-computed returns of outcome rewards (RLOO/leave-one-out baseline) and implicit process rewards to avoid numerical instability. Key tricks: initialize the PRM from the SFT/base model (eliminating a costly RM stage and reducing distribution shift), online prompt filtering for median difficulty, KL=0, β=0.05. The method is a generic plug-in compatible with REINFORCE, RLOO, GRPO, and PPO.

- **Implicit PRM:** ORM repurposed as token-level PRM via β·log(π_φ/π_ref); both causal LMs.
- **Loop:** sample K, grade with outcome verifier, online-update PRM, derive dense rewards, PPO policy update.
- **Advantages:** combine RLOO outcome returns with implicit process returns (separate to avoid instability).
- **Tricks:** PRM initialized from SFT/base (no RM stage), median-difficulty prompt filtering, KL=0, β=0.05; runs on veRL with 8×A800.

**Key results:** Eurus-2-7B-PRIME (from Qwen2.5-Math-7B-Base): 26.7% AIME 2024 pass@1 vs 13.3% for Qwen2.5-Math-7B-Instruct, 2.5× sample efficiency and +6.9% over outcome-only RLOO, 11× faster than VinePPO.

*Evolution:* Reacts to R1/Kimi K1.5's conclusion that PRMs are impractical at scale, demonstrating dense, online-updatable implicit process rewards can be both scalable and beneficial.

### Skywork Open Reasoner 1 Technical Report
*2025 · rl · `rl_Skywork-OR1_2505.22312.txt` · arXiv [2505.22312](https://arxiv.org/abs/2505.22312)*

Skywork-OR1's core contribution is MAGIC (Multi-stage Adaptive entropy scheduling for GRPO In Convergence), a modified GRPO. It uses a token-level policy loss (length-normalization term 1/|y| removed to reduce length bias), group-relative binary {0,1} rule-based rewards, and PPO-style clipping (ε=0.2). Key tricks: (1) rejection sampling keeping only non-zero-advantage groups; (2) high-temperature rollouts τ=1.0 for exploration; (3) on-policy updates (N_SGD=1) to slow entropy collapse; (4) Adaptive Entropy Control with target entropy tgt-ent=0.2 and step Δ=0.005 that dynamically scales the entropy-loss coefficient, activating only when entropy falls below target to keep it lower-bounded; (5) no KL loss (β=0), since KL pulls the actor back to the reference and blocks late-stage gains. Advantage masking for truncated responses and clip-higher were ablated but not adopted. Rewards come from Math-Verify (boxed-wrapping) and a secure subprocess code sandbox.

- **Algorithm:** MAGIC = modified GRPO with token-level loss (no 1/|y|), binary rule rewards, PPO clip 0.2.
- **Exploration:** τ=1.0 rollouts, rejection sampling of non-zero-advantage groups, on-policy (N_SGD=1).
- **Entropy control:** Adaptive Entropy Control (tgt-ent=0.2, Δ=0.005) dynamically scales entropy-loss coef.
- **No KL:** β=0 to avoid blocking late-stage gains; clip-higher ablated but not adopted.

**Key results:** Skywork-OR1-32B reaches 82.2 AIME24, 73.3 AIME25, 63.0 LiveCodeBench (avg@4), surpassing DeepSeek-R1 and Qwen3-32B on math; avg improves 57.8%→72.8% for 32B.

*Evolution:* Applies RL to already-distilled long-CoT models (not base models) with a systematic entropy-collapse study, motivating entropy-aware, on-policy RL recipes for long-CoT reasoning.

### s1: Simple test-time scaling
*2025 · rl · `rl_s1_2501.19393.txt` · arXiv [2501.19393](https://arxiv.org/abs/2501.19393)*

s1's core contribution is budget forcing, a decoding-time sequential test-time-scaling technique (not a training algorithm). To cap thinking, it appends the end-of-thinking token delimiter (optionally 'Final Answer:') to force early exit; to extend thinking, it suppresses that delimiter and appends 'Wait' to the current trace, prompting the model to reflect and self-correct. The model itself is produced by plain SFT of Qwen2.5-32B-Instruct on 1K Gemini-distilled reasoning traces. The authors formalize evaluation via three desiderata: Control (fraction of generations within a min/max token budget; 100% ideal), Scaling (average slope of the accuracy-vs-compute piecewise-linear curve; must be positive), and Performance (max accuracy). Budget forcing attains 100% control and the best positive slope (15) versus token/step/class-conditional prompt control and rejection sampling (which shows inverse scaling).

- **Method:** budget forcing—append end-of-thinking delimiter to cap, suppress it + append 'Wait' to extend.
- **Training:** plain SFT of Qwen2.5-32B-Instruct on 1K Gemini-distilled reasoning traces.
- **Evaluation:** Control (token-budget compliance), Scaling (accuracy-vs-compute slope), Performance.
- **Result:** 100% control and best positive slope (15), beating token/step/prompt control and rejection sampling.

**Key results:** s1-32B, SFT on 1,000 traces for 26 min on 16 H100s, hits 56.7% AIME24 with budget forcing vs o1-preview's 44.6%, and 93.0% MATH500; budget forcing extrapolates AIME24 from 50% to 57%.

*Evolution:* Builds on LIMA's 1K-examples lesson and Snell et al.'s test-time-compute scaling, showing 1K distilled SFT plus a simple decoding trick matches o1-preview and motivating later sample-efficient reasoning work.

## 2026

### CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation
*2026 · method · `method_CUDA-Agent_2602.24286.txt` · arXiv [2602.24286](https://arxiv.org/abs/2602.24286)*

CUDA Agent is a large-scale agentic-RL system for CUDA-kernel generation built on Seed1.6 (MoE, 23B active / 230B total), running a ReAct-style loop aligned with OpenHands that exposes Bash/Read/Write/Edit/Glob/Grep tools plus a CUDA SKILL.md prescribing a profile→implement→compile/verify→iterate workflow targeting ≥5% speedup over torch.compile. Reward is a discrete robust schedule r∈{-1,1,2,3} (-1 on correctness failure, 3 if >5% faster than both eager and compile, 2 if faster than eager only, 1 otherwise) that avoids the outlier/easy-kernel bias of raw speedup. PPO uses DAPO-style asymmetric clipping (eps_lower=0.2, eps_higher=0.28).

- Reward-hacking blockers: permission-protect eval scripts, forbid `torch.nn.functional` fallbacks via context managers, validate on 5 random inputs, careful sync/warmup/averaged profiling, disable web access.
- A CPU-GPU decoupled sandbox (Docker terminal + 128 NVIDIA H20 GPU pool) gives process-isolated, noise-free latency measurement.
- Multi-stage warm-up is what enables stable PPO training for 150 steps versus collapse at 17 steps without it.

**Key results:** On KernelBench, 98.8% pass rate, 96.8% faster-rate vs torch.compile, 2.11× geomean speedup; 100%/100%/92% faster-rate on Level-1/2/3 and ~40% above Claude Opus 4.5 / Gemini 3 Pro on Level-3.

*Evolution:* Extends the agentic-RL-for-coding line by combining DAPO asymmetric clipping, ReAct/OpenHands loops, and Anthropic Agent Skills, showing execution-rewarded agentic RL can match or beat compiler-based optimization.

### Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic
*2026 · method · `method_GRPO-Ustatistic-theory_2603.01162.txt` · arXiv [2603.01162](https://arxiv.org/abs/2603.01162)*

This work proves GRPO's policy-gradient estimator is a second-order U-statistic (Lemma 1), and gives a meta-algorithm unifying REINFORCE, A2C, and GRPO that differ only in the baseline subtracted from reward Z: zero, the true value function (oracle), or the leave-one-out group mean (GRPO). GRPO thus replaces the critic by sampling G outputs per prompt. A Hoeffding decomposition splits the U-statistic into a first-order term equal to the oracle gradient plus a G⁻² degenerate residual, yielding MSE bounds, an oracle property (MSE→oracle as G→∞), and optimality (minimizes MSE among prompt-only baselines, strictly beating vanilla REINFORCE).

- The suboptimality gap is bounded under L-smoothness, PL, and constant or 1/i learning rates, producing a scaling law: the bound c1/B + c2/(BG) + c3/(BG²) is minimized by a universal group size G* = √(c3/c1), independent of budget and iterations.
- Theorem 8 gives consistency and a chi-squared-mixture asymptotic for the gap without parameter identifiability (overparameterized regime).
- Appendix A extends the analysis to production GRPO with reward standardization, token-level importance sampling, and a Schulman K3 KL penalty.

**Key results:** GRPO's gradient MSE strictly beats vanilla REINFORCE and matches the oracle for G≥8; the optimal per-prompt group size is universal — G*=32 on GSM8K (Qwen2.5-1.5B) and ~64 on MATH (Qwen2.5-Math-7B).

*Evolution:* Supplies the first unified finite-sample and asymptotic theory plus a budget-independent group-size scaling law for the 2026 flood of critic-free RLVR algorithms.

### GRPO-VPS: Enhancing Group Relative Policy Optimization with Verifiable Process Supervision for Effective Reasoning
*2026 · method · `method_GRPO-VPS_2604.20659.txt` · arXiv [2604.20659](https://arxiv.org/abs/2604.20659)*

GRPO-VPS augments GRPO with model-free, verifier-free process supervision to fix its indiscriminate credit assignment. It (1) partitions each response at adaptive entropy-based cutpoints — tokens with entropy above a percentile threshold τ=0.95 are cutpoints, and the response is split into M≈4 segments each holding roughly the same number of cutpoints, placing boundaries at high-uncertainty decision junctions; (2) reads segment-wise progress by appending the ground-truth answer y* at each boundary and computing the conditional probability C(z≤k)=π_θ(y*|x,z≤k) in a single forward pass, with per-segment signal ΔCk=C(z≤k)−C(z≤k−1)∈[−1,1]; and (3) forms a hybrid advantage Ã_k = A_i + α·ΔCk where A_i=r_i−mean(r) is the standard group-relative outcome advantage and α=1.2 (stable over [0.8,1.4]).

- The on-policy gradient fuses sparse outcome reward with dense process reward; no critic, PRM, or Monte Carlo rollouts are needed, so it stays lightweight and scalable like GRPO.

**Key results:** On Qwen2.5-Math-1.5B, up to +2.6 Pass@1 over GRPO while cutting reasoning length 11.0–13.7%; on the 7B model, +1.1 point with up to 34.0% length reduction; general reasoning gains reach +1.8 on MMLU-Pro and +2.4 on TheoremQA.

*Evolution:* Published at ICLR 2026, it attacks GRPO's credit assignment without PRMs or MC rollouts by reading the model's own conditional answer probability as a free dense signal.

### Computer Environments Elicit General Agentic Intelligence in LLMs
*2026 · method · `method_LLM-in-Sandbox_2601.16206.txt` · arXiv [2601.16206](https://arxiv.org/abs/2601.16206)*

LLM-in-Sandbox virtualizes the computer as a lightweight Ubuntu Docker code sandbox (Python + NumPy/SciPy, one ~1.1 GB shared image) exposing three tools — bash, file_editor, finish — in a ReAct-style multi-turn loop until finish or a turn cap. LLM-in-Sandbox-RL is the core contribution: RL inside this sandbox using only general context-based data, contrasting with text-only LLM-RL (no environment) and SWE-RL (environment but domain-specific data). Training uses GRPO++ (the GRPO variant from DeepSWE/rLLM) with outcome-based, rule-based rewards — F1 for multi-choice, ROUGE-L for free-form, binary +1/0 for math — and zero reward for trajectories exceeding max turns/tokens.

- Design is minimal (no domain-specific configs) and exploratory (a prompt encouraging tool use and anti-hardcoding); trajectories are generated in sandbox mode.
- The problem solved: whether computer environments elicit general intelligence and whether weaker models can be trained to harness them, with skills that transfer even to text-only LLM mode.

**Key results:** Training-free, Qwen3-Coder-30B-A3B on AIME25 math +15.5% (26.0→41.5) and MiniMax-M2 instruction-following +14.4%, with long-context tokens cut up to 8×; after RL, weak Qwen3-4B-Instruct-2507 beats its own LLM mode and also improves in pure LLM mode.

*Evolution:* Isolates the computer environment itself as a source of general intelligence and shows general-data sandbox RL transfers to non-code domains and even text-only inference.

### Latent Thought Flow: Efficient Latent Reasoning in Large Language Models
*2026 · method · `method_LTF-LatentThoughtFlow_2606.16222.txt` · arXiv [2606.16222](https://arxiv.org/abs/2606.16222)*

LTF replaces explicit CoT with variable-length continuous latent trajectories τ=(z_{1:T}, bot), where each z_t is a Gaussian-sampled continuous thought (reparameterized mean/diagonal variance from a decoder+latent head) and bot is an adaptive stop; only the LoRA module and latent head are trainable. The target is a reward-induced posterior p*(τ|x,y) ∝ R_{x,y}(τ)=V_{x,y}(τ)·exp(−λ_c·C(τ)), where V combines a verifier accuracy term and normalized answer likelihood and C(τ)=T penalizes length. A continuous GFlowNet matches this posterior via an Entropy-Weighted Subtrajectory Balance (EW-SubTB) objective that reweights SubTB residuals by length-normalized latent entropy, preserving diverse high-utility paths while suppressing posterior collapse. A reference-prior branch L_prior anchors early exploration to golden-rationale embeddings.

- Final loss L = L_flow + λ_ans·L_ans + λ_prior·L_prior.
- The problem solved is the distributional allocation of probability mass across correctness/cost trade-offs that deterministic or reward-maximizing latent reasoning misses.

**Key results:** Beats the strongest latent baseline ReGuLaR by +12.9% avg accuracy while cutting reasoning length 34.5% (LLaMA-8B on GSM8K-Aug 53.14% vs 50.14%); transfer gains +6.0% accuracy and −19.9% length; extreme compression (#L=1) lifts MATH 8.10% vs 6.62%.

*Evolution:* Reframes latent reasoning as reward-proportional distribution matching over accuracy and cost, reacting to the posterior collapse of RL-style latent reasoning and the verbosity of maximum-likelihood CoT compression.

### LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification
*2026 · method · `method_LaTER_2605.07315.txt` · arXiv [2605.07315](https://arxiv.org/abs/2605.07315)*

LaTER (Latent-Then-Explicit Reasoning) is a hybrid test-time paradigm: a bounded continuous latent rollout for exploration, then explicit CoT for verification, reusing the latent KV cache so explicit decoding conditions on the latent trajectory. Training-free LaTER maps the final-layer hidden state h_s to input-embedding space via the pseudo-inverse W_out⁺W_in (the LatentMAS construction) and feeds it back without token commitment, switching via switch(s)=1[H_s>τ_H or ŷ_s∈T_stop] (entropy threshold τ_H=7 and terminating-token probes like `<|im_end|>`). Trained LaTER replaces the pseudo-inverse with a learned projector g_φ, adds `<latent_think>`/`</latent_think>` boundary tokens, and supervises latent placeholders only indirectly via gradients from downstream explicit/answer tokens (no CE inside the latent segment).

- Loss L=L_CE+λ_KL·L_KL+L_halt^eff: CE split into CoT (λ_CoT=0.5) and non-CoT terms; top-k (k=128) teacher KL self-distillation (T=1.0, λ_KL=0.25); a halting loss with BCE at the boundary, gated by EMA(L_CE)/L_CE (base weight 0.025).
- Optimized with AdamW (lr 1e-7), DeepSpeed ZeRO-3 on 8×A800 80G (~5 days).

**Key results:** Trained LaTER on Qwen3-14B reaches 80.0% on AIME 2025 (+10.0 over the CoT baseline's 70.0%) while using 33% fewer tokens (10,575 vs 15,730); training-free LaTER lifts AIME 2025 70.0%→73.3% and MATH-500 93.4%→97.2%.

*Evolution:* Splits labor — latent exploration for search, explicit CoT for verification — reacting against fully replacing explicit CoT, which degrades symbolic tasks like MATH-500/AIME.

### Mid-Think: Training-Free Intermediate-Budget Reasoning via Token-Level Triggers
*2026 · method · `method_Mid-Think_2601.07036.txt` · arXiv [2601.07036](https://arxiv.org/abs/2601.07036)*

Mid-Think is a training-free prompting format built on the discovery that hybrid-thinking models' reasoning is governed by a few trigger tokens, not high-level instructions. Attention analysis on Qwen3-8B shows the `Okay` token after the think-opening marker activates reasoning (3.18× attention ratio) while the double-newline after the think-closing marker suppresses it (5.27×); Mid-Think concatenates both trigger regions with a `<reason>` tag (replaceable with `<begin>`, `<less think>`) to induce intermediate-budget reasoning at roughly a 0.5 budget. The trigger is template-derived: DASD-4B-Thinking (trained on "We need" traces) shifts the trigger to `We`, and a mismatched trigger collapses to No-Think.

- Beyond inference, Mid-Think serves as the GRPO rollout/training mode after SFT, raising policy entropy and shortening outputs to cut training time ~15% while improving accuracy.
- Versus fixed-token and prompt-based budget baselines it gives finer-grained, Pareto-optimal control.

**Key results:** Qwen3-8B trained with Mid-Think as the GRPO objective improves Think-test AIME 69.8%→72.4% and GPQA 58.5%→61.1% while cutting RL training time ~15% (54h→46h); training-free Mid-Think on MATH500 reaches 92.3% at 2,589 tokens vs 94.6%/4,904 (Think) and 83.2%/899 (No-Think).

*Evolution:* Reframes Think/No-Think switching as overfitting to a few trigger tokens inherited from SFT data templates, offering a lightweight route to intermediate reasoning budgets and cheaper, higher-entropy RL training.

### Polar: Agentic RL on Any Harness at Scale
*2026 · method · `method_Polar-AgenticRL_2605.24220.txt` · arXiv [2605.24220](https://arxiv.org/abs/2605.24220)*

Polar is a rollout framework for scalable asynchronous RL over arbitrary agent harnesses, treating the harness as a black box. Its key idea is to move the integration boundary to the model API endpoint: a gateway proxy intercepts LLM calls (Anthropic Messages, OpenAI Chat/Responses, Google generateContent), normalizes them to OpenAI Chat shape, forwards to the local inference backend, and captures prompt/response token IDs, logprobs, and finish reasons. A rollout server plus gateway nodes separate task scheduling from per-session execution; gateway worker pools (INIT/READY/RUNNING/POSTRUN) with evaluator prewarm keep CPU runtime setup off the GPU-bound agent path.

- Two trajectory builders are provided: per_request (one trace per completion, lossless but fragmenting) and token-faithful prefix_merging, which partitions completions into append-only chains via a strict token-prefix check, copies only sampled assistant tokens as trainable (loss mask 1), masks canonical interstitials (loss mask 0), and broadcasts outcome rewards.
- Used with standard Slime GRPO, registered as a NeMo Gym environment, and rewrites ProRL Agent.

**Key results:** Qwen3.5-4B + standard GRPO via Polar improves SWE-Bench Verified pass@1 by +22.6 on Codex (3.8%→26.4%); prefix_merging cuts a 3-step run from 189.5 to 35.2 min (5.39×) with 87.7% rollout utilization vs 20.4%.

*Evolution:* Moves the integration boundary from the agent SDK to the model API endpoint, enabling training of closed-source/binary harnesses unchanged and positioning rollout infra as trainer-agnostic.

### R³: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning
*2026 · method · `method_R3-ReplayReflectionRanking_2601.19620.txt` · arXiv [2601.19620](https://arxiv.org/abs/2601.19620)*

R³ extends GRPO to fix intra-group advantage collapse (when all G samples share a reward, std→0 and gradients vanish). Three components: (1) Cross-Context Replay (CCR) injects k historical samples with opposing rewards from the buffer into a uniform group, forming G_mix = G ∪ G_C, with advantage Â_i=(R_i−mean)/(α·std+λ) keeping off-policy data from swamping on-policy signal; (2) In-Context Self-Reflection (ISR) augments hard queries (mean historical reward < τ) with retrieved past errors plus structured reflection guidance prompting diagnostic self-correction; and (3) Structural Entropy Ranking Reward (SERR) gives unsupervised dense rewards to truncated/failed traces via Peak Entropy E_peak (mean entropy of the top-⌊p·L⌋ most uncertain tokens, capturing exploration) and Global Entropy E_global (mean over all tokens, capturing stability).

- A partial order i≻j requires E_peak higher AND E_global lower; dominance counts rank samples and linearly scaled rewards R^(k)=R_max·(1−(k−1)/(N−1)) are assigned.
- These plug into the standard clipped-PPO + KL GRPO objective.

**Key results:** R³-1.5B reaches 60.59 average across five math benchmarks (+12.78 over DeepSeek-R1-Distill-Qwen-1.5B), with AIME24 47.50 at only 7,574 tokens vs the base's 28.1 at 12,270; R³-7B scores 67.18 (AIME24 61.04, beating Thinker-7B 60.00).

*Evolution:* Reacts specifically to GRPO's intra-group advantage collapse on hard tasks, motivating unsupervised, process-level reward signals for truncated trajectories and more sample-efficient replay-augmented RL.

### ReSyn: Autonomously Scaling Synthetic Environments for Reasoning Models
*2026 · method · `method_ReSyn_2602.20117.txt` · arXiv [2602.20117](https://arxiv.org/abs/2602.20117)*

ReSyn autonomously synthesizes diverse reasoning environments for RLVR by exploiting the generator-verifier gap: the LLM specifies how to check a solution (a code verifier R:S×A→{0,1}) rather than solve it, enabling tasks beyond the teacher's solving ability. Each environment is a tuple (S,A,R,O,ρ_0) where ρ_0(d,n) generates instances at difficulty d, O renders natural-language questions, and R programmatically verifies answers. Training uses the open-source DAPO recipe on verl with Qwen2.5-7B-Instruct as policy (chosen over base to retain instruction following). RLVR samples G candidates a_i~π(·|Q), computes reward r_i=V(a_i) times a format score, and updates via GRPO advantages.

- Hyperparameters: clip ratios 0.2/0.28, no KL penalty (kl_coef=0), dynamic sampling that filters all-correct/all-wrong groups, an overlong-buffer penalty, lr=1e-6, 16 responses/prompt, 400 steps.
- Prompts force  `<think>` tags and `<answer>` formatting.

**Key results:** Qwen2.5-7B-Instruct + ReSyn (DAPO, 400 steps): BBH 75.2 (+14% rel.), BBEH 14.3 (+27% rel.), AIME 2024 14.0 (+40% rel.), GSM8K 91.4 (vs 82.3); it also beats SynLogic-7B on BBH (75.2 vs 66.5) and BBEH (14.3 vs 8.0).

*Evolution:* Automates environment authoring via LLM-synthesized code verifiers, scaling task diversity >10× beyond SynLogic's 35 handcrafted tasks and pointing toward labor-free, self-scaling reasoning curricula.

### Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability
*2026 · method · `method_SOAR_2601.18778.txt` · arXiv [2601.18778](https://arxiv.org/abs/2601.18778)*

SOAR (Self-Optimization via Asymmetric RL) is an asymmetric self-play, bilevel meta-RL framework in which both teacher and student initialize from the base model. The bilevel objective maximizes expected student accuracy on D_train after an RL update on a teacher-generated dataset X, instantiated as a "double meta-RL loop": an outer loop trains the teacher with RLOO to emit question-answer pairs, and an inner loop trains the student with RLOO (10 steps, batch 8) on those pairs. The teacher reward is grounded — R(X_k)=Acc(student' on Q_R)−Acc(baseline student on Q_R), with Q_R sampled from D_train and averaged over r=4 parallel students — rather than an intrinsic learnability reward, which avoids unrolling the inner loop (no BPTT).

- Key tricks: student promotion when the 3-step MA reward exceeds τ=0.01, rejection-sampled format filtering (provably RLOO-equivalent), and a shaped math reward (120/20/10/0).
- The teacher never sees the hard problems.

**Key results:** On fail@128 MATH with Llama-3.2-3B-Instruct, SOAR-PQ reaches 18.9±5.3% pass@32 vs 9.6±2.6% Hard-Only (~2× / +9.3%) and ~4× pass@1 (1.7 vs 0.5); on HARP, +4.2 points; PQ-MATH recovers 75% of the full-curated-MATH upper bound's gain and transfers to held-out OlympiadBench.

*Evolution:* Reacts against data-free self-play that relies on intrinsic/proxy rewards (causing diversity collapse), grounding the teacher in measured student progress via a tractable RLOO bilevel loop to escape RLVR plateaus.

### SWE-Fuse: Empowering Software Agents via Issue-free Trajectory Learning and Entropy-aware RLVR Training
*2026 · method · `method_SWE-Fuse_2603.07927.txt` · arXiv [2603.07927](https://arxiv.org/abs/2603.07927)*

SWE-Fuse is an entropy-aware RLVR algorithm for multi-turn SWE agents. It samples G trajectories per prompt and computes RLOO advantages A_i = R_i − mean_{j≠i} R_j (leave-one-out, unbiased vs GRPO's self-coupled group mean, lower variance for small models), with binary rewards R(τ)=1[patch passes all tests]. The key novelty is entropy-adaptive clipping: per-sample sequence entropy H_i is batch-normalized to H_norm in [0,1], then mapped to a clipping radius ε_i in [ε_min, ε_max] that is asymmetric in advantage sign — for A_i>0, ε grows with entropy (encourage exploration of uncertain-but-good samples); for A_i≤0, ε shrinks with entropy (conservative, avoiding over-penalizing exploratory behavior under noisy negative advantages). This replaces PPO's fixed ratio constraint.

- Paired with an issue-free-driven trajectory learning module (distillation + SFT) that addresses real-world data noise (issue/solution misalignment), the full framework requires only basic bash tool calls.

**Key results:** SWE-Fuse-Qwen3-32B resolves 60.2% of SWE-bench Verified issues (65.2% at TTS@8), SOTA among open-source ≤32B models and +1.8% over OpenAI-o3; SWE-Fuse-Qwen3-8B reaches 43.0% (49.8% at TTS@8); data scaling 0→14k trajectories improves resolve rate 13.5%→39.0% (2.9×).

*Evolution:* Grafts entropy-adaptive clipping onto RLOO and attacks real-world data noise via issue-free trajectory learning, showing lightweight 8B/32B agents can rival 100B+–1T models.

### Synthetic Sandbox for Training Machine Learning Engineering Agents
*2026 · method · `method_SandMLE_2604.04872.txt` · arXiv [2604.04872](https://arxiv.org/abs/2604.04872)*

SandMLE makes trajectory-wise on-policy RL feasible for MLE agents by attacking the execution-latency bottleneck. The core insight is that MLE rollout latency is driven by dataset size, not compilation, so it replaces real Kaggle-scale tasks (~4.09M samples, ~196s/step) with synthetic micro-scale sandboxes (50–200 samples, ~14.31s, a 13.7× speedup) that preserve structural/mathematical complexity. It runs trajectory-level GRPO (a critic-free PPO variant: group-normalized advantage A_i=(r_i−μ)/σ, clipped surrogate, KL to reference) using the ReAct framework, computing policy gradients only on action tokens with observation masking and fully masking trajectories exceeding the time limit.

- To combat reward sparsity in long-horizon multi-turn rollouts, it uses a dense milestone-based reward r = w_format·r_format + w_execute·I_execute + Σ w_si·I_si, where tiered indicators check whether the final score exceeds Kaggle-style thresholds (median/bronze/silver/gold).
- Implemented on RLLM; an ablation shows sparse reward (format + gold only) collapses 30B Any Medal from 27.3% to 13.6%.

**Key results:** Cuts per-rollout execution time 13.7× (196.17s→14.31s); on MLE-Bench-Lite, Qwen3-8B/14B/30B-SandMLE achieve 22.7%/22.7%/27.3% Any Medal (+66.9%/+24.7%/+100.7% vs Base).

*Evolution:* Addresses the MLE-specific execution-latency bottleneck that had forced prior MLE-agent work back to SFT or off-policy async RL, showing synthetic micro-scale environments can proxy real MLE tasks for on-policy RL.

### Self-Verified Distillation: Your Language Model Is Secretly Its Own Synthetic Data Pipeline
*2026 · method · `method_SelfVerifiedDistillation_2605.26132.txt` · arXiv [2605.26132](https://arxiv.org/abs/2605.26132)*

Self-Verified Distillation turns unlabeled seed questions into supervised data without external teachers, tool feedback, or ground-truth answers. Given a post-trained model p_θ, it samples n candidates per question, then filters each through a UQ-inspired multi-stage self-verifier: (1) cycle-consistency (infer the question from the answer, compare to the seed, 2 calls), (2) a factuality check for factual/arithmetic/logical errors, and (3) a total-correctness check requiring ~95% complete solutions — each repeated v times with unanimous voting. A candidate is accepted only if it passes all stages in all judge calls; the model is then trained via SFT on accepted (question, solution) pairs.

- The core problem is that naive self-training reinforces the model's own mistakes; the contribution shows strong prompt-based self-verification yields higher-precision filtering than unfiltered generations.
- Decomposed multi-stage UQ verification beats a simple correctness prompt (+8.4 vs +4.9 mean delta) at equal strength.

**Key results:** Qwen3-4B aggregate held-out pass@1 gains: +16.7 math (AIME26+HMMT), +11.1 science (GPQA Diamond+HLE), +8.3 coding (LCBv5+LCBv6); it beats UQ-TTC on 5 of 6 benchmarks while using a single inference call at test time (vs up to 168).

*Evolution:* Shows prompt-based self-verification can make self-generated data reliable enough to further refine already post-trained models without teachers or tools, contrasting with Simple Self-Distillation on raw unverified outputs.

### Beyond Model Scaling: Test-Time Intervention for Efficient Deep Reasoning
*2026 · method · `method_Think-with-Me_2601.11252.txt` · arXiv [2601.11252](https://arxiv.org/abs/2601.11252)*

Think-with-Me is a test-time interactive intervention framework for Large Reasoning Models, driven by two observations: (1) transitional conjunctions ("wait", "so", "but", "therefore", "alternatively") naturally segment reasoning and mark self-validation/exploration phases; (2) judicious conjunction use raises accuracy but overuse hurts it. At inference the model pauses when it emits a trigger conjunction; an evaluator (human or LLM proxy) scores the partial reasoning on two content-agnostic criteria, Rationality and Completeness, yielding four statuses mapped to rule-based guidance (continue / conclude / rethink-revise / correct-refine). Feedback wrapped in special tags is appended to the context and the model resumes, looping up to 10 interventions or until `</thinking>`. The model is adapted via GRPO with LoRA (α=8, r=4), with each rollout sampled through this multi-round interactive mode.

- A composite reward sums correctness (r_c), format (r_f), and length (r_l) terms; equal weights (1,1,1) work best.
- An information-theoretic proposition shows target-relevant feedback reduces conditional entropy, motivating reduced redundant self-validation.

**Key results:** AIME24@32: 73.85% accuracy at 1,182.50 tokens (8K window) vs QwQ-32B 66.66% at 4,052.80 tokens — +7.19% accuracy and ~81% shorter reasoning; MATH500 90.60% at 1,081.87 tokens with the LLM proxy; self-termination reaches 73–100% across datasets.

*Evolution:* Shifts control from internal numerical signals to external, semantically-grounded feedback at the model's intrinsic phase boundaries, anticipating controllable, human/AI-collaborative reasoning budgets under tight context windows.

### DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
*2026 · report · `report_DeepSeek-V4_2606.19348.txt` · arXiv [2606.19348](https://arxiv.org/abs/2606.19348)*

The central post-training contribution is full-vocabulary On-Policy Distillation (OPD) replacing the mixed-RL merging stage. After SFT+GRPO specialist training (GRPO with domain-tailored reward models), a single student is consolidated from 10+ teacher experts by minimizing a reverse-KL objective L=Σ w_i·D_KL(π_θ‖π_Ei) over student-sampled trajectories; full-vocabulary logits (not token-level KL estimates) are used for stable gradients. Hard-to-verify tasks drop scalar reward models in favor of a Generative Reward Model (GRM) where the actor itself judges rubric-guided trajectories, jointly optimizing evaluative and generative ability.

- Supporting machinery: an XML/DSML tool-call schema, interleaved thinking that persists reasoning across tool turns, Quick Instruction special tokens that reuse KV cache for auxiliary tasks, FP4 quantization-aware training for MoE weights and the CSA indexer QK path, teacher-weight offloading with on-the-fly logit reconstruction, token-granular write-ahead-logged rollout, and a DSec sandbox platform for agentic execution.

**Key results:** DeepSeek-V4-Pro-Max: SimpleQA-Verified 57.9 Pass@1, Codeforces rating 3206 (ranks 23rd among humans), SWE-Verified 80.6% resolved, Putnam-2025 120/120 proof-perfect; at 1M-token context it needs only 27% of V3.2's single-token inference FLOPs and 10% of its KV cache.

*Evolution:* Replaces the mixed-RL merging stage with full-vocabulary on-policy distillation from 10+ specialist teachers, cementing distillation-based consolidation as the default for multi-capability models.

### GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents
*2026 · report · `report_GLM-5V-Turbo_2604.26752.txt` · arXiv [2604.26752](https://arxiv.org/abs/2604.26752)*

Core contributions include CogViT (a parameter-efficient vision encoder with a two-stage distillation-then-contrastive recipe, the Muon optimizer, and QK-Norm for stability) and Multimodal Multi-Token Prediction (MMTP), which extends text-only MTP to multimodal inputs by passing a single shared learnable placeholder token in place of all visual tokens to the MTP head. A 0.5B ablation shows this beats both direct visual-embedding passing (lower loss, more stable convergence, since the lightweight MTP head cannot absorb visual representations) and full masking (it stays compatible with sequence/context parallelism and avoids cross-stage embedding propagation). For RL, the team builds a unified VLM RL Gym and an independent reward system.

- RL infra: full-pipeline decoupling (rollout, reward, batch build, weight transfer overlapped; reference model prefetched from CPU), multimodal-aware memory management (ViT/projector recomputation + CPU offload), and topology-aware CP/TP partitioning moved upstream into data loading with async all-to-all and joint bin-packing over sequence length and ViT token count.
- Relative visual policy optimization is used for UI-to-code tasks.

**Key results:** GLM-5V-Turbo scores 94.8 on Design2Code (beating Claude Opus 4.6); multimodal agent benchmarks: 30.7 ImageMining, 51.9 BrowseComp-VL, 75.7 AndroidWorld, 62.3 OSWorld, MMSearchPlus 30.0 (~8× over GLM-4.6V).

*Evolution:* Pushes toward native multimodal agents where perception, not just reasoning, is foundational, reacting against text-centric agent models.

### GLM-5: from Vibe Coding to Agentic Engineering
*2026 · report · `report_GLM-5_2602.15763.txt` · arXiv [2602.15763](https://arxiv.org/abs/2602.15763)*

The RL backbone is GRPO augmented with IcePop to mitigate training-inference mismatch: it explicitly separates the training policy π_train (gradient updates) from the inference policy π_infer (rollouts), removes KL regularization, and applies a pop(ρ, 1/β, β) operator that zeroes samples whose mismatch ratio falls outside [1/β, β] (β=2, eps_low=0.2, eps_high=0.28, group 32, batch 32, on-policy). For DSA RL, a deterministic torch.topk in the indexer (vs non-deterministic CUDA topk) plus freezing indexer parameters stabilizes training. The headline contribution is asynchronous Agent RL: training and inference engines run on separate GPUs, a Multi-Task Rollout Orchestrator (>1k concurrent rollouts) standardizes trajectories, and a TITO gateway preserves exact action-level token correspondence.

- Direct Double-sided Importance Sampling reuses rollout log-probs with token-level clipping [1−eps_l, 1+eps_h] (dropping π_old) to bound off-policy bias; stale and environment-crashed samples are dropped, and DP-aware routing maximizes KV-cache reuse.
- Cross-stage distillation replaces the advantage with sg log(π_infer/π_train) against teacher logits.

**Key results:** GLM-5 scores 50 on the Artificial Analysis Intelligence Index v4.0 (first open-weights model to do so), reaches SWE-bench Verified 77.8, BrowseComp 75.9, tau2-Bench 65.8, and Vending-Bench 2 $4,432, ~20% average gain over GLM-4.7.

*Evolution:* Pushes the open-weights frontier from passive coding toward long-horizon agentic engineering with an asynchronous agent-RL stack (TITO, double-sided importance sampling, multi-task orchestrator, cross-stage distillation).

### Kimi K3: Open Frontier Intelligence (Technical Report of Kimi K3)
*2026 · report · `report_Kimi-K3_2607.24653.txt` · arXiv [2607.24653](https://arxiv.org/abs/2607.24653)*

The RL algorithm extends the partial-rollout scheme from Kimi K1.5/K2.5: each iteration samples K completions for N prompts but pauses generation once a fraction λ completes, enqueueing paused rollouts for resumption via the sandbox, so a single long-horizon trajectory spans multiple iterations. Policy optimization follows the Kimi K2.5 algorithm with per-token regularization that tolerates the extreme off-policy/stale data this introduces. Reasoning-effort control overrides the task reward with −1 when a trajectory's token budget exceeds τ·b_0(x). For non-verifiable tasks, an Agentic Generative Reward Model (GRM) uses tournament-style binary comparisons with a mandatory rubric protocol (read output, generate rubric, score each candidate, record) and budget-based verbosity control.

- Multi-Teacher On-Policy Distillation gives a per-token reward = clip(sg(log teacher/student), −R_max, R_max); top-k variants showed no gain.
- A unified white-box RL environment represents an agent harness as composable modules (tools, system prompts, context management, skills, memories, subagents) that can instantiate Kimi Code, Claude Code, Codex, OpenClaw, or Hermes.
- Architectural contributions include Quantile Balancing (auxiliary-loss-free routing with expert bias set to the router-score quantile matching target load, estimated via histogram all-reduce) and Stable LatentMoE (RMSNorm before up-projection plus bounded SiTU-GLU); infra includes MoonEP (perfectly balanced expert-parallel MoE with E/R redundant experts and static shapes), AgentENV microVM sandboxes (133ms/49ms checkpoint/resume, fork, snapshot), an external KV-cache pool with auto-throttling, and KDA Context Parallelism via a prefix scan.

**Key results:** Kimi K3 is a 2.8T-parameter MoE (104B activated) open model scoring 91.2 on BrowseComp, 77.8 on ProgramBench (best), 93.5 on GPQA Diamond, and ranking #1 on WebDev Arena (1678 Elo, the first open model to lead it); #4/580 on Artificial Analysis Intelligence Index v4.1 (57.1).

*Evolution:* Pushes both scaling axes together — 3T-class pre-training versus the prior ~1T open regime, and million-token agentic RL/test-time scaling — as the first open 3T-class model released in 2026.

### Qwen3-Coder-Next Technical Report
*2026 · report · `report_Qwen3-Coder-Next_2603.00729.txt` · arXiv [2603.00729](https://arxiv.org/abs/2603.00729)*

The core contribution is scaling agentic training: synthesizing verifiable executable tasks and learning from environment feedback via mid-training + RL. Infra is MegaFlow, a cloud-native Kubernetes/Argo orchestration system where each task is an Argo workflow of agent-rollout, evaluation, and post-processing pods that co-locate the agent and execution containers for low-overhead long-horizon interaction. Mid-training uses next-token + FIM with best-fit packing (BFP, reimplemented in C++ in Megatron) to avoid context hallucination and head-side truncation, and masks highly repetitive segments. SFT filters data via closed-loop verification (a Mini-SWE-agent user-simulator executes proposed code/commands and checks compiler/runtime feedback) and pairwise preference judging.

- Single-turn RL extends execution-driven RL beyond competitive programming to library-use, multilingual, and secure-coding tasks, with auto-synthesized unit tests chosen by majority-vote consensus driving execution-based rewards.
- Multi-turn agentic RL uses trajectory-level completion rewards plus an unfinished-trajectory penalty and a turn-level tool-format penalty (token-level), and a reinforced reward-hacking blocker that rejects tool calls pairing a repo URL with network keywords (git/curl/wget).
- Distillation merges the four experts into the SFT model.

**Key results:** Qwen3-Coder-Next (80B total / 3B active MoE) scores 70.6/71.1/71.3% on SWE-Bench Verified across SWE-Agent/MiniSWE-Agent/OpenHands, 42.7% on SWE-Bench Pro, and 36.2% on Terminal-Bench 2.0; tool-template following averages 92.7% across 5 IDE/CLI scaffolds vs 49.3 (GPT-5-2) and 85.4 (Claude-sonnet-4-5).

*Evolution:* Pushes agentic training onto a small-active-footprint MoE (80B/3B active), claiming that scaling agentic training rather than model size drives real-world coding-agent capability.

### Qwen3.5-Omni Technical Report
*2026 · report · `report_Qwen3.5-Omni_2604.15804.txt` · arXiv [2604.15804](https://arxiv.org/abs/2604.15804)*

The core contribution is ARIA (Adaptive Rate Interleave Alignment), which collapses Qwen3-Omni's dual-channel text/speech generation into a single interleaved stream by enforcing an adaptive rate constraint: for any generated prefix, the cumulative speech-to-text token ratio must not exceed the item-level global ratio. This fixes skipped words, mispronunciations, and number rendering caused by mismatched text/speech tokenization rates, with minimal latency cost. Architecturally, both Thinker and Talker use Hybrid-Attention MoE with a Gated Delta Net module to cut KV-cache I/O for long audio-video sequences; the Talker predicts RVQ multi-codebook tokens via an MTP module rendered by a causal ConvNet (Code2Wav). Temporal modeling replaces sparse TM-RoPE time IDs with explicit second-format timestamp strings plus random-interval audio timestamps.

- Training algorithms include Specialist Distillation, On-Policy Distillation, Interaction-Aligned RL for the Thinker, and DPO plus GSPO with rule-based rewards for the Talker.
- Zero-shot voice cloning is enabled by a dedicated Talker system prompt encoding voice characteristics.

**Key results:** Qwen3.5-Omni-Plus achieves SOTA across 215 audio/audio-visual benchmarks, surpassing Gemini-3.1 Pro on audio understanding, recognition, translation, and dialogue (FLEURS ASR avg WER 6.6% vs 7.3%; en2xx avg BLEU 33.8 vs 31.8); on SEED-TTS test-en WER 1.26, and cross-lingual TTS cuts zh→ko error from 14.4 (CosyVoice3) to 4.03 (~72% relative reduction).

*Evolution:* Scales native omnimodal training to hundreds of billions of parameters with Hybrid-Attention MoE and the ARIA streaming-alignment trick, moving from passive perception-response toward native omni-agents that act and even generate executable code from audio-visual instructions.

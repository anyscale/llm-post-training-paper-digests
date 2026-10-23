# 02 — Training Curriculum

*Post-training summaries, generated solely from the full-text files in [`../texts/`](../texts/). Papers are sorted by arXiv-ID year (2022→2026), then by corpus order within each year. Each entry synthesizes **one lens only** for that paper; the chronological cross-lens narrative — older trends, how the field evolved, and why the newer methods were proposed — lives in [`EVOLUTION_OVERVIEW.md`](EVOLUTION_OVERVIEW.md).*

**Lens:** how training stages are ordered and combined — pretrain→SFT→preference→RL→distillation, multi-stage RL, self-play loops, warmup/anneal schedules, cross-stage data mixing — and the rationale behind each curriculum.

**Coverage:** 94 of the 148 papers contribute substantive content on this lens; papers for which this lens was not a focus are omitted here and appear under their relevant topic files.

---

## 2022

### CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning
*2022 · code · `code_CodeRL_2207.01780.txt` · arXiv [2207.01780](https://arxiv.org/abs/2207.01780)*

A four-stage curriculum on CodeT5-large (770M): (1) two-stage pretraining — masked span prediction on CSN for 150 epochs, then generation-corrupted-prediction and next-token-prediction for 10 epochs each; (2) warm-start finetuning with cross-entropy only (imitation) for up to 10 epochs to avoid an unstable actor-critic loop; (3) critic training on synthetic samples with the actor frozen; (4) RL finetuning combining `Lce` and `Lrl` at equal weight. The mix is justified because `Lrl` alone vanishes and `Lce` alone overfits after ~epoch 10. At inference an optional Critic Sampling loop (one repair then one refine round) re-runs generation conditioned on unit-test outcomes. The pipeline is model-agnostic, also applied to GPT-Neo and GPT-J actors.

- Pretrain (MSP→GCPY→NTP) → CE warm-start SFT → frozen-actor critic training → mixed `Lce`+`Lrl` RL
- Mixed objective stabilizes the actor-critic; CE-only overfits after epoch 10, RL-only suffers vanishing gradients
- Test-time Critic Sampling adds one repair + one refine round conditioned on execution outcomes
- Applied model-agnostically to CodeT5-770M, GPT-Neo, and GPT-J actors

**Key results:** CodeRL+CodeT5-770M sets APPS SOTA at 2.69% pass@1, 20.98% pass@1000; zero-shot MBPP 63.0% pass@80, beating GPT-137B with a far smaller model.

*Evolution:* Extends REINFORCE/actor-critic and CodeT5 pretraining into code, anticipating execution-feedback and reward-model-driven post-training for code.

### STaR: Bootstrapping Reasoning With Reasoning
*2022 · data · `data_STaR_2203.14465.txt` · arXiv [2203.14465](https://arxiv.org/abs/2203.14465)*

Not a stacked pretrain→SFT→preference→RL pipeline but a single self-improvement loop on a pretrained GPT-J. Outer loop order: (1) few-shot-prompt the current model to generate (rationale, answer) for all of D; (2) filter by answer correctness; (3) rationalize failures using the ground-truth answer as a hint; (4) fine-tune the original GPT-J on the union; (5) repeat with the improved model until performance plateaus. Schedule: 100-step LR warmup then constant LR=1e-6 (Adam), first loop 40 fine-tuning steps growing 20% per loop, batch 8 packed 1024-token sequences on one TPU-v3. Arithmetic shows a stage-wise curriculum (n-digit only after (n-1)-digit mastered) that rationalization collapses, letting many lengths advance together; adding unseen 9-10 digit problems at iteration 20 yields OOD generalization.

- Iterative self-bootstrapping loop over data: generate → correctness-filter → rationalize failures → SFT → repeat
- Constant LR=1e-6 after 100-step warmup; fine-tuning steps grow 20% per outer loop
- Rationalization collapses the stage-wise arithmetic curriculum, advancing many digit-lengths at once
- Keeping few-shot prompts during SFT raises CommonsenseQA accuracy (69.9→72.5% with rationalization)

**Key results:** GPT-J (6B) + STaR reaches 72.5% CommonsenseQA dev (matching a 30x larger GPT-3), 89.5% arithmetic after 16 iterations, and GSM8K 10.7%.

*Evolution:* Turns CoT prompting and Expert Iteration into an iterative self-bootstrapping SFT loop using answer-correctness as reward, anticipating verifier-reward RL and self-play.

### SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions
*2022 · data · `data_SelfInstruct_2212.10560.txt` · arXiv [2212.10560](https://arxiv.org/abs/2212.10560)*

A single supervised finetuning stage, not a multi-stage curriculum: generate synthetic instruction data, then finetune the same GPT3 davinci (175B) that produced it via OpenAI's API with default hyperparameters, `prompt_loss_weight=0`, and 2 epochs to avoid overfitting. Multiple prompt templates (varied Task:/Input:/Output: prefixes and line breaks) encode each example for format robustness. The bootstrapping loop is over data, not training stages; a scaling ablation shows gains plateau after ~16K instructions on user-oriented eval (and at hundreds on SUPERNI). An optional second step regenerates outputs with InstructGPT003 (distillation) for ~+10% on user eval, but this is framed as future work rather than the core recipe.

- One SFT stage finetuning the same 175B model that generated the 52K instructions
- 2 epochs, `prompt_loss_weight=0`, multi-template encoding for format robustness
- Bootstrapping loop is over data not stages; gains plateau at ~16K instructions
- Optional InstructGPT003 distillation of outputs adds ~+10% user eval (future work)

**Key results:** GPT3SELF-INST improves over vanilla GPT3 by +33.1 ROUGE-L on SUPERNI (39.9 vs 6.8), nearly matching InstructGPT001 (40.8).

*Evolution:* Pioneers self-bootstrapped synthetic instruction data, enabling Alpaca/Baize and motivating distillation/reward-based refinement of synthetic corpora.

## 2023

### AgentTuning: Enabling Generalized Agent Abilities for LLMs
*2023 · code · `code_AgentTuning_2310.12823.txt` · arXiv [2310.12823](https://arxiv.org/abs/2310.12823)*

A single-stage hybrid SFT recipe, not a multi-stage RL/preference pipeline. AgentInstruct (agent trajectories) is mixed with ShareGPT (general-domain) at ratio η (agent-data fraction); η was scanned 0-1 in 0.1 steps on the 7B model and η=0.2 chosen for best held-out performance. All data is standardized to a Vicuna-style multi-turn chatbot format with loss computed only on model outputs. Base models are Llama-2-{7,13,70}b-chat, trained with Megatron-LM, AdamW, cosine schedule with 2% warmup, batch 64 at 4096 seq len, LR 5e-5 (7B/13B) and 1e-5 (70B); 70B adds pipeline parallelism atop tensor parallelism. The rationale is that general capabilities are pivotal for agent generalization and training solely on agent data collapses both general and held-out performance.

- Single hybrid SFT: agent trajectories + general ShareGPT at η=0.2 agent fraction
- Loss only on model outputs; Vicuna-style multi-turn chatbot format
- Cross-stage data mixing scan (η 0-1) picks 0.2 for best held-out generalization
- Llama-2-chat bases with Megatron-LM; 70B adds pipeline parallelism over tensor parallelism

**Key results:** AgentLM-70B held-out overall 1.40 (+176% over Llama-2-70B), roughly matching GPT-3.5 (1.49), while preserving general ability (MMLU 59.5, GSM8K 59.7).

*Evolution:* Builds on ReAct/Self-Instruct/FLAN and reacts to the AgentBench gap, anticipating agentic post-training and tool-use RL over environment rewards.

### What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
*2023 · data · `data_DEITA_2312.15685.txt` · arXiv [2312.15685](https://arxiv.org/abs/2312.15685)*

The pipeline is pretrain → SFT on DEITA-selected 6K/10K data → optional DPO. The paper centers on the SFT stage since its contribution is SFT data curation; DPO is added only for stronger reference points. For DPO, 10K comparison pairs sampled from UltraFeedback (the same pairs Zephyr uses) train on top of the best DEITA SFT checkpoint (DEITA-Mistral-7B6K). Because the SFT set is tiny, epochs are raised to 6 for SFT and 9 for DPO; Mistral-7B SFT uses batch 512, lr 2e-5, cosine warmup 0.1, while DPO uses batch 32, lr 5e-7, linear warmup 0.1. No multi-stage RL or self-play loop is used.

- pretrain → curated 6K/10K SFT → optional 10K-pair DPO (UltraFeedback, Zephyr's pairs)
- Tiny data → 6 SFT epochs, 9 DPO epochs; SFT lr 2e-5 cosine, DPO lr 5e-7 linear
- DPO stacks on the best SFT checkpoint; no RL/self-play loop
- 3K curated samples match full 300K training (100x reduction)

**Key results:** DEITA-Mistral-7B6K+DPO reaches 7.55 MT-Bench and 90.06% AlpacaEval, comparable to Zephyr-beta trained on ~30x more data.

*Evolution:* Formalizes automatic SFT data selection across complexity, quality, and diversity, driving the data-centric turn in alignment.

### Magicoder: Empowering Code Generation with OSS-Instruct
*2023 · data · `data_Magicoder-OSS-Instruct_2312.02120.txt` · arXiv [2312.02120](https://arxiv.org/abs/2312.02120)*

Two-stage instruction tuning on a pretrained code base. Stage 1 finetunes the base LLM (CodeLlama-Python-7B or DeepSeek-Coder-Base-6.7B) for 2 epochs on 75K OSS-Instruct data, producing Magicoder; Stage 2 continually finetunes Magicoder on evol-codealpaca-v1 (~110K Evol-Instruct samples), producing MagicoderS. The rationale is that the two data sources are orthogonal — OSS-Instruct grounds generation in real code for diversity/controllability while Evol-Instruct increases complexity — so stacking compounds gains. Training uses 2 A100-80GB GPUs with PyTorch DDP, Adafactor, lr 5e-5, 15 warmup steps, linear schedule, batch 512, sequence length 1216 (stage 1) or 1024 (stage 2). No pretraining, preference, or RL stage is involved.

- Two-stage SFT: 75K OSS-Instruct (2 epochs) → ~110K Evol-Instruct continual finetune
- Stacking rationale: OSS (real-code diversity) ⊥ Evol (complexity) compounds gains
- Adafactor, lr 5e-5, 15 warmup, linear, batch 512; seq 1216 then 1024
- No preference/RL stage; a pure instruction-tuning curriculum

**Key results:** MagicoderS-CL-7B surpasses ChatGPT on HumanEval+ pass@1 (66.5 vs 65.9) with only 7B parameters.

*Evolution:* Extends Self-Instruct by grounding synthetic code instruction in open-source code, orthogonal to Evol-Instruct, motivating later data-centric scaling.

### Scaling Relationship on Learning Mathematical Reasoning with Large Language Models
*2023 · data · `data_RFT-rejection-sampling_2308.01825.txt` · arXiv [2308.01825](https://arxiv.org/abs/2308.01825)*

A two-stage pipeline on a frozen pretrained base: (1) SFT on GSM8K (3 epochs, batch 128, peak LR 2e-5, 3% warmup) to obtain an SFT model π that performs zero-shot chain-of-thought; (2) rejection sampling with π to build an augmented dataset D′=D∪{augmented paths}, then fine-tune the original pretrained base ρ (not π) on D′ to get πRFT. There is no preference, RL, or distillation stage. The rationale is that SFT alone plateaus (log-linear with diminishing returns for better bases), so RFT injects multiple diverse correct CoT paths; order matters because the SFT model is only a data generator while RFT retrains from the base. Aggregating rejection samples from several SFT models into D′U13B/D′U33B acts as cross-model data mixing that fills the pre-training gap. The paper argues pre-training loss is the ultimate lever and SFT/RFT are cheap (~1e-5 to 1e-4 of pre-train FLOPs).

- SFT on GSM8K → rejection-sample correct CoT paths → retrain from base on the augmented set (not from the SFT model)
- SFT model is a data generator only; RFT reinitializes from pretrained base ρ
- Cross-model data mixing: aggregating samples from multiple SFT models (D′U13B/D′U33B)
- No preference/RL/distillation; SFT/RFT cost ~1e-5–1e-4 of pre-train FLOPs

**Key results:** LLaMA-7B RFT-U13B reaches 49.3% maj1@1 on GSM8K (+13.4 over SFT); RFT helps weaker (higher pre-training-loss) bases most and adds nothing for 33B/65B/70B.

*Evolution:* Builds on STaR and self-consistency, replacing verifiers/MCTS with rejection-sampling deduplication, prefiguring RLAIF data-augmentation and RLVR for math.

### Statistical Rejection Sampling Improves Preference Optimization
*2023 · data · `data_RSO_2309.06657.txt` · arXiv [2309.06657](https://arxiv.org/abs/2309.06657)*

A single-round staged offline pipeline: (1) train an SFT policy on Dsft; (2) train a pairwise T5-XXL reward-ranking model on Dhf; (3) sample candidates from the SFT policy and apply statistical rejection sampling guided by the reward to approximate samples from the optimal policy, then label pairs with the reward model; (4) fit the policy with a classification loss on these labeled pairs. Checkpoints are selected by highest reward-model win rate against the SFT target, and sampling/ranking prompts are drawn from the SFT training set. The rationale is that since the MLE of the optimal policy wants pairs sampled from π*, drawing from the reward-induced optimal policy πrψ is closer to on-policy online RLHF than fitting on Dhf (DPO) or sampling only from the SFT policy (SLiC). One round mirrors one ReST round, and the authors note RSO can be iterated.

- Staged offline pipeline: SFT → reward model → reward-guided rejection sampling → pair-labeling → classification-loss fit
- Rejection sampling approximates samples from the reward-induced optimal policy πrψ (closer to on-policy than DPO/SLiC)
- Checkpoints picked by reward-model win rate vs the SFT target
- One round equals one ReST round; RSO is explicitly iterable over multiple rounds

**Key results:** RSO (T5-large) reaches 84.40% Gold Reward / 71.86% AutoSxS on Reddit TL;DR vs DPO 76.09/58.65; human raters choose RSO >2x more often than DPO.

*Evolution:* Bridges offline preference optimization and online RLHF, foreshadowing iterative-DPO and best-of-N / on-policy preference methods.

### ZEPHYR: Direct Distillation of LM Alignment
*2023 · data · `data_Zephyr_2310.16944.txt` · arXiv [2310.16944](https://arxiv.org/abs/2310.16944)*

The pipeline mirrors InstructGPT's three stages: (1) dSFT on filtered UltraChat, (2) AI-feedback preference collection from UltraFeedback, (3) dDPO on those preferences. The final Zephyr-7B uses one SFT epoch then three DPO epochs. SFT is a hard prerequisite — skipping it makes DPO fail to learn the chat template (MT-Bench collapses to 4.76). Counterintuitively, DPO overfits to perfect train accuracy after one epoch yet keeps improving downstream through three epochs; however, if SFT runs more than one epoch, extended DPO causes regression. SFT uses a cosine schedule, peak LR 2e-5, 10% warmup, global batch 512, packed 2048-token sequences; DPO uses a linear schedule, peak 5e-7, global batch 32, beta=0.1. Training finishes in 2-4 hours on 16 A100-80GB GPUs.

- Three-stage distilled alignment: dSFT (UltraChat) → AIF preference collection → dDPO (UltraFeedback)
- SFT is a hard prerequisite; skipping it collapses MT-Bench to 4.76 (chat template not learned)
- Counterintuitive schedule: 1 SFT epoch then 3 DPO epochs; DPO overfits train yet improves downstream
- SFT cosine lr 2e-5 batch 512; DPO linear lr 5e-7 batch 32 beta=0.1

**Key results:** Zephyr-7B reaches MT-Bench 7.34 (surpassing Llama2-Chat-70B's 6.86) and AlpacaEval 90.60%, trained in 2-4 hours on 16 A100s with no human annotation.

*Evolution:* Replaces PPO+human-feedback with DPO over GPT-4 AI feedback, popularizing the distilled SFT→DPO-on-AIF pipeline for small aligned models.

### Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations
*2023 · rl · `rl_Math-Shepherd_2312.08935.txt` · arXiv [2312.08935](https://arxiv.org/abs/2312.08935)*

A multi-stage pipeline: (1) SFT generators and completers on MetaMath; (2) use these models to sample solutions and auto-annotate step labels, training an ORM and a PRM (1 epoch, lr 1e-6); (3) reinforce the generator with step-by-step PPO supervised by the PRM; (4) optionally apply the same PRM as a verifier (best-of-N reranking) over PPO outputs, with RL and verification shown to be complementary. Cross-stage reuse is explicit — the Mistral-7B PRM supervises both LLaMA2-7B and Mistral-7B generators, and analysis recommends using a stronger reward model than the generator. Iterative RL (re-training the PRM after PPO) is flagged as future work because the initial PRM becomes insufficient for the post-PPO model. No pretraining-from-scratch stage; all stages start from existing base LLMs (LLaMA2, LLemma, Mistral, DeepSeek-67B).

- Four stages: MetaMath SFT → auto-annotated ORM/PRM training → step-wise PPO → optional PRM best-of-N verification
- Cross-stage reuse: Mistral-7B PRM supervises LLaMA2-7B and Mistral-7B generators; recommend an RM stronger than the generator
- RL and verification are complementary; iterative PRM re-training flagged as future work (initial PRM insufficient post-PPO)
- All stages build on existing base LLMs (LLaMA2, LLemma, Mistral, DeepSeek-67B); no from-scratch pretraining

**Key results:** Mistral-7B step-wise PPO lifts GSM8K 77.9→84.1% and MATH 28.6→33.0%; with verification, 89.1%/43.5%; DeepSeek-67B-MetaMath + verification reaches 93.3%/48.1%, SOTA for open-source tool-free models.

*Evolution:* Makes process supervision scalable via auto-generated step labels, plugging a bootstrapped PRM into step-wise PPO and anticipating the 2024 RLVR/process-reward and iterative-RLHF wave.

## 2024

### AgentGym: Evolving Large Language Model-based Agents across Diverse Environments
*2024 · code · `code_AgentGym_2406.04151.txt` · arXiv [2406.04151](https://arxiv.org/abs/2406.04151)*

Two ordered stages turn Llama-2-Chat-7B into a multi-environment agent. Stage 1 is behavioral cloning (BC) on AGENTTRAJ plus general-domain chat (3 epochs, lr 1e-5) to seed instruction-following. Stage 2 is AGENTEVOL: M=4 iterations alternating exploration (one trajectory sampled per instruction over a larger instruction set) and reward-weighted fine-tuning (1 epoch/iteration, lr 1e-5). Crucially, each iteration merges new rollouts back with the *initial* AGENTTRAJ rather than the previous iteration's data, preventing drift; the general-domain data is retained throughout. Ablations show gains grow with M but converge, and that anchoring to the initial dataset is more stable than chaining prior-iteration data.

- BC: 3 epochs, lr 1e-5, log pi(tau|e,u) objective
- Evolution: 4 iterations, ~20h on 8x A100-80GB, K=1 sample/instruction
- Merge-with-initial (not with previous iter) is the key anti-drift design
- General-domain chat retained across all stages

**Key results:** AGENTEVOL beats the BC upper bound BClarge on WebShop (76.5 vs 73.5), ALFWorld (88.0 vs 83.0), BabyAI (82.7 vs 74.19), and TextCraft (64.0 vs 60.0), and matches or surpasses GPT-4-Turbo on several environments.

*Evolution:* An early 2024 attempt to push LLM agent self-evolution beyond isolated environments toward multi-environment generalists, complementing behavioral-cloning agent-tuning works like AgentTuning/AgentOhana.

### Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions
*2024 · code · `code_Marco-o1_2411.14405.txt` · arXiv [2411.14405](https://arxiv.org/abs/2411.14405)*

Training is essentially a single flat SFT stage: full-parameter fine-tuning of Qwen2-7B-Instruct on combined CoT + instruction data yields Marco-o1-CoT. There is no separate preference, RL, or distillation stage; MCTS, action-granularity variants, and the reflection prompt are applied only at inference time. The synthetic CoT data is itself generated via MCTS, creating a loose generate-then-train loop, but no multi-stage RL or self-play schedule is described. Future work proposes adding ORM/PRM and RL to fine-tune decision-making, confirming the current pipeline is SFT-then-search rather than a curriculum.

- Single SFT stage, no preference/RL/distillation training
- MCTS + reflection operate only at inference, not as training stages
- Synthetic CoT data generated via MCTS (a soft generate-then-train loop)
- Future ORM/PRM + RL proposed but absent here

**Key results:** Marco-o1-MCTS (step): 90.40% on MGSM-En vs 84.00% Qwen2-7B-Instruct (+6.17%); mini-step 32 reaches 82.40% on MGSM-Zh vs 76.80% (+5.60%); Test@32 reaches 99.60% (En) / 96.80% (Zh).

*Evolution:* An early open attempt (Nov 2024) to demystify o1's reasoning by combining CoT SFT with AlphaZero-style MCTS and self-reflection, extending the reasoning-model trend to open-ended and multilingual tasks.

### WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
*2024 · code · `code_WebRL_2411.02337.txt` · arXiv [2411.02337](https://arxiv.org/abs/2411.02337)*

A two-stage pipeline. Part 1 SFTs the base model on WebArena-Lite; the SFT model seeds both the replay buffer (successful trajectories) and the failure set (unsolved instructions). Part 2 is an 8-phase self-evolving curriculum RL loop: each phase generates ~500 critic-filtered instructions from the failure set, rolls out the actor, labels success via an ORM, then trains actor and critic on new rollouts mixed with replayed history. Difficulty rises progressively because seeds come from prior failures and generated instructions grow more complex. Historical replay is capped at 2x new-interaction volume and selected by actor-confidence perplexity in [1/0.95, 1/0.5], revisiting moderately-difficult successes while avoiding over-familiar or still-too-hard ones.

- SFT seeds replay buffer (successes) and failure set (unsolved)
- 8-phase curriculum RL; each phase ~500 critic-filtered new instructions
- Replay capped at 2x new volume, perplexity-band selection
- Difficulty escalates as seeds derive from prior failures

**Key results:** Llama-3.1-8B + WebRL: 4.8% to 42.4% SR on WebArena-Lite (vs GPT-4-Turbo 17.6%); Llama-3.1-70B reaches 49.1% and the trained 8B ORM verifies success at ~80% accuracy.

*Evolution:* Builds on DigiRL/AWR online RL and WizardLM evol-instruct, adding a self-evolving curriculum and KL-anchored off-policy updates to DigiRL's fixed task set.

### DataComp-LM: In search of the next generation of training sets for language models
*2024 · data · `data_DataCompLM_2406.11794.txt` · arXiv [2406.11794](https://arxiv.org/abs/2406.11794)*

Curriculum is not the focus, but a fixed pretraining recipe isolates data effects and a staged recipe is used for the 7B flagship. The 2.5T run is two-stage: stage-1 pretrains 2T tokens on DCLM-BASELINE + StarCoder + ProofPile2, then cooldown on a reweighted mix (70% tighter-fastText DCLM-BASELINE + 30% ProofPile). Two cooldowns (200B, 270B tokens) are averaged via a model soup (0.8/0.2). Continual pretraining (~120B tokens, Grow-Linear variable-length curriculum, RoPE base 10k→100k) extends context 2048→8192. Instruction tuning uses OH-2.5 or a custom 4M-instance/8B-token DCLM-IT mix (Adam, 10 epochs, cosine LR); no PPO/DPO alignment is applied.

- Two-stage pretrain: 2T tokens then reweighted cooldown
- Two cooldowns averaged via model soup (0.8/0.2)
- Continual pretraining: Grow-Linear curriculum, RoPE 10k→100k, 2K→8K context
- Instruction tuning only; no preference/RL stage

**Key results:** DCLM-BASELINE 7B on 2.6T tokens reaches 63.7% MMLU 5-shot, +6.6pp over MAP-Neo with 40% less compute; DCLM-IT lifts 5-shot GSM8K from 2.1 to 52.5.

*Evolution:* Systematizes data-centric LM research at 240T-token/7B scale as a reaction to closed-data models, seeding later open pretraining datasets like Nemotron-CC, OLMo-2, and WebOrganizer.

### KTO: Model Alignment as Prospect Theoretic Optimization
*2024 · data · `data_KTO_2402.01306.txt` · arXiv [2402.01306](https://arxiv.org/abs/2402.01306)*

Assumes the standard pretrain→SFT→preference-optimization pipeline and studies where KTO slots in. Default usage is SFT→KTO, with SFT targets a subset of {yw}. A key finding: at sufficient scale (Llama-13B, 30B), KTO alone — skipping SFT — matches SFT+KTO and is the only tested method with this property, whereas DPO without SFT rambles and hallucinates entire conversations. Hyperparameter guidance ties to stage: β∈[0.01,0.10] for larger post-SFT models, β∈[0.10,1.00] for smaller models trained with KTO directly. No multi-stage RL, self-play, or cross-stage data mixing is proposed; the contribution is the single alignment stage and whether SFT precedes it.

- Standard pretrain→SFT→KTO, or skip SFT at scale
- KTO-alone matches SFT+KTO at 13B/30B (unique among methods tested)
- β guidance depends on stage and scale
- No iterative RL or self-play

**Key results:** KTO matches or exceeds DPO across 1B–30B; on Zephyr-β-SFT/UltraFeedback GSM8K rises 40.0→53.5 (+13.5) over DPO, and human-eval winrate vs SFT targets is 72.9% (KTO) vs 62.1% (DPO, p<0.05).

*Evolution:* Reframes alignment through prospect theory, enabling alignment from cheap binary feedback and showing SFT can be skipped at scale, motivating later non-pairwise, reference-free alignment losses.

### ORPO: Monolithic Preference Optimization without Reference Model
*2024 · data · `data_ORPO_2403.07691.txt` · arXiv [2403.07691](https://arxiv.org/abs/2403.07691)*

ORPO's central design is collapsing the standard alignment pipeline. Instead of pretrain→SFT warm-up→preference alignment (RLHF/PPO or DPO), which needs a frozen reference SFT model and a separate warm-up, ORPO performs SFT and preference alignment in a single monolithic run with no reference model. Baselines follow the conventional ordering: one epoch SFT on chosen responses, then DPO (β=0.1, 3 epochs) or PPO. ORPO instead trains up to 10 epochs (best by eval loss) with a single learning rate (8e-6) and a λ weighting the odds-ratio term, removing the staged curriculum entirely.

- Collapses SFT + preference alignment into one monolithic run
- No reference model, no separate SFT warm-up
- Single LR 8e-6, up to 10 epochs, λ-weighted odds-ratio term
- Baselines keep conventional SFT→DPO/PPO ordering

**Key results:** Mistral-ORPO-β (7B): 12.20% AlpacaEval2.0, 7.32 MT-Bench, 66.19% IFEval loose — surpassing Zephyr-β and Llama-2-Chat (13B) on UltraFeedback alone; ORPO win rate vs DPO reaches 70.9% at OPT-1.3B with margin growing with scale.

*Evolution:* Builds on DPO's reference-free spirit and unlikelihood training, reacting against the unstable multi-stage SFT→RLHF/PPO pipeline and the cost of a frozen reference model; helped popularize single-stage, reference-free alignment.

### Self-Rewarding Language Models
*2024 · data · `data_SelfRewarding_2401.10020.txt` · arXiv [2401.10020](https://arxiv.org/abs/2401.10020)*

Stage ordering is M0 (base Llama 2 70B)→M1 (SFT on IFT+EFT seed)→M2 (DPO on AIFT(M1))→M3 (DPO on AIFT(M2)): an iterative DPO loop where each model generates and self-rewards the preference data training its successor; the reward model is not frozen and co-improves. SFT uses lr 5.5e-6 cosine-decaying to 1.1e-6, batch 16, dropout 0.1, loss on target tokens only. DPO uses lr 1e-6→1e-7, batch 16, dropout 0.1, β=0.1, with early stopping every 200 steps via Claude 2 AlpacaEval on 253 validation examples. Only 3 iterations run; a positive-only SFT augmentation (score-5 examples) did not help, so preference pairs were necessary.

- M0→SFT→DPO→DPO: iterative DPO with self-generated, self-rewarded pairs
- Reward model folded into policy and co-improves (not frozen)
- SFT lr 5.5e-6→1.1e-6; DPO lr 1e-6→1e-7, β=0.1, early-stop via Claude 2
- Positive-only SFT augmentation ablated out; preference pairs required

**Key results:** M3 reaches 20.44% AlpacaEval 2.0 win rate over GPT-4 Turbo, beating Claude 2 (17.19%), Gemini Pro (16.85%), GPT-4 0613 (15.76%); reward-modeling pairwise accuracy rises 65.1% (SFT)→78.7% (M1)→80.4% (M2)→81.7% (M3).

*Evolution:* Folds the reward model into the policy so it co-improves across DPO iterations rather than staying frozen — a 2024 companion to SPIN's reward-model-free self-play, motivating scalable self-alignment.

### SimPO: Simple Preference Optimization with a Reference-Free Reward
*2024 · data · `data_SimPO_2405.14734.txt` · arXiv [2405.14734](https://arxiv.org/abs/2405.14734)*

A standard two-stage offline pipeline with no iterative or online loops. Base setup: pretrain→SFT on UltraChat-200k (lr 2e-5, batch 128, seq len 2048, 1 epoch, cosine with 10% warmup)→preference optimization on UltraFeedback. Instruct setup skips SFT, starting from an off-the-shelf instruct checkpoint. ORPO is the only baseline that can drop SFT, but for fairness it is also started from the same SFT checkpoint. Preference optimization uses batch 128, 1 epoch, method-specific LRs (SimPO lr 3e-7 to 1e-6), Adam. The paper explicitly restricts itself to offline, non-iterative training.

- Two-stage offline: SFT then single preference-optimization stage
- SFT on UltraChat-200k: lr 2e-5, 1 epoch, 10% warmup
- Preference stage: 1 epoch, SimPO lr 3e-7–1e-6, no reference model
- Explicitly offline and non-iterative; ORPO baseline also started post-SFT

**Key results:** Gemma-2-9B-it-SimPO reaches 72.4% AlpacaEval 2 LC, 59.1% Arena-Hard WR, ranks 1st among <10B Chatbot Arena models; outperforms DPO by up to 6.4 pts (AlpacaEval 2 LC) and 7.5 pts (Arena-Hard) while cutting DPO runtime ~20% and GPU memory ~10%.

*Evolution:* Builds on DPO and reference-free work like ORPO, reacting to DPO's reward/generation mismatch and reference-model cost; became a widely adopted lightweight DPO replacement.

### Tülu 3: Pushing Frontiers in Open Language Model Post-Training
*2024 · data · `data_Tulu3_2411.15124.txt` · arXiv [2411.15124](https://arxiv.org/abs/2411.15124)*

A four-stage recipe applied to Llama 3.1 base models (8B, 70B, 405B), each stage resuming from the prior checkpoint: (1) data curation, (2) SFT (2 epochs, sum loss, LR 5e-6/2e-6, batch 128, len 4096), (3) preference tuning with length-normalized DPO (1 epoch, LR 5e-7/2e-7, β=5, len 2048), (4) RLVR with PPO (KL β swept 0.1–0.01; final 8B 0.05, 70B 0.07). At 405B, GSM8K was dropped from RLVR (saturated) and only MATH was used. Cross-stage data is mixed deliberately: skill-specific SFT data, then on+off-policy preferences, then verifiable prompts. Online DPO and rejection-sampling self-play were tried but did not help and were dropped.

- Four stages: SFT→length-normalized DPO→RLVR (PPO), resuming checkpoints
- SFT 2 epochs; DPO 1 epoch β=5; RLVR KL β swept 0.1–0.01
- 405B drops GSM8K from RLVR (saturated), keeps MATH
- Online DPO and rejection-sampling self-play ablated out

**Key results:** Tülu 3 70B averages 76.2 on Tülu 3 Eval, surpassing Llama 3.1 70B Instruct (74.1), GPT-4o-mini (69.6), Claude 3.5 Haiku (75.3); RLVR lifts 8B GSM8K 84.3→87.6, MATH 42.0→43.7, IFEval 81.1→82.4.

*Evolution:* Extends Tülu 2/Zephyr-β toward closed-style multi-stage recipes, formalizing verifiable-reward RL (RLVR) and anticipating the 2024–25 GRPO/DeepSeek-R1 wave and reproducible open post-training.

### DeepSeek-V3 Technical Report
*2024 · report · `report_DeepSeek-V3_2412.19437.txt` · arXiv [2412.19437](https://arxiv.org/abs/2412.19437)*

The pipeline is pre-training (14.8T tokens), then a two-stage long-context extension with YaRN (4K→32K, then 32K→128K, 1000 steps each), then post-training = SFT then RL on the base model. SFT runs 2 epochs with cosine-decay LR 5e-6→1e-6, packing multiple samples per sequence with sample masking. R1 reasoning distillation is folded into the SFT stage via expert-model-generated data, balancing accuracy and output length. RL (GRPO) then runs across coding, math, writing, role-play, and QA; for non-verifiable domains feedback comes from self-rewarding / constitutional AI using DeepSeek-V3 voting. Post-training is remarkably cheap: ~5K H800 GPU-hours vs 2.664M for pre-training.

- Pretrain 14.8T → YaRN long-context (4K→32K→128K, 1000 steps each)
- SFT 2 epochs, LR 5e-6→1e-6, sample-masked packing
- R1 reasoning distillation folded into SFT via expert-generated data
- GRPO RL last; non-verifiable domains use self-rewarding/constitutional voting

**Key results:** DeepSeek-V3 (671B-total/37B-active MoE) achieves AIME 2024 39.2 Pass@1, MATH-500 90.2 EM, 85.5 Arena-Hard win rate (first open-source >85%), competitive with GPT-4o and Claude-3.5-Sonnet; full training ~2.788M H800 GPU-hours (~$5.576M), post-training only ~5K.

*Evolution:* Pioneers distilling long-CoT reasoning from the R1 series into a standard aligned MoE via SFT/RL-generated data, anticipating the 2025 wave of distilling strong reasoning models into efficient base models.

### Gemma 2: Improving Open Language Models at a Practical Size
*2024 · report · `report_Gemma2_2408.00118.txt` · arXiv [2408.00118](https://arxiv.org/abs/2408.00118)*

Post-training proceeds SFT→RLHF→model merging. SFT performs behavioral cloning on synthetic and real prompts with responses predominantly teacher-generated, plus on-policy distillation from the teacher on the student's own distribution. RLHF reuses the Gemma 1.1 algorithm but a new reward model ~an order of magnitude larger than the policy, oriented toward multi-turn conversation and trained on labeled English-only preferences over the same SFT prompts. Finally, checkpoints from different hyperparameter runs are weight-averaged. Pre-training differs by size: 2B and 9B students are distilled from the 27B teacher rather than trained from scratch, while 27B is trained from scratch. Rationale is to improve helpfulness while minimizing safety and hallucination harms.

- SFT→RLHF→weight-averaged model merging
- SFT uses teacher-generated responses plus on-policy teacher distillation
- RLHF reward model ~10x policy size, multi-turn oriented
- 2B/9B pretrained by distillation from 27B; 27B trained from scratch

**Key results:** Gemma 2 27B-IT reaches LMSYS Elo 1218, beating Llama-3 70B-IT (1206); 9B-IT Elo 1187 matches GPT-4-0314, 2.6B-IT (1126) beats GPT-3.5-Turbo-0613; distillation lifts a 2B model from 60.3 to 67.7 average over 500B tokens.

*Evolution:* Repurposes knowledge distillation as a pre-training substitute to push small models past compute-optimal token counts, helping motivate teacher-distilled pre-training and weight-averaged RLHF recipes.

### InternLM2 Technical Report
*2024 · report · `report_InternLM2_2403.17297.txt` · arXiv [2403.17297](https://arxiv.org/abs/2403.17297)*

Training proceeds in three pre-training phases then alignment. Phase 1 (~90% of steps) trains at ≤4k context. Phase 2 (~9%) introduces 50% ≤32k data and raises RoPE base 50k→1M. Phase 3 is capability-specific enhancement on 24B high-quality tokens with smaller LR/batch. English, Chinese, and code are mixed throughout. Alignment is SFT (10M ChatML instructions, 1 epoch, AdamW lr 4e-5) followed by COOL RLHF over three online rounds. Long-context pre-training data is reused in SFT and RLHF to preserve long-context ability; tool/agent capability is added via Agent-FLAN disentanglement and a RICO code-interpreter strategy. Checkpoints at every stage (Base, post-enhancement, -SFT, post-RLHF) are released to expose the staged evolution.

- Three pretrain phases: ≤4k → 50% ≤32k (RoPE 50k→1M) → 24B capability-specific
- Alignment: SFT (10M ChatML, 1 epoch, lr 4e-5) → COOL RLHF, 3 online rounds
- Long-context data reused across pretrain, SFT, and RLHF
- All-stage checkpoints released to expose staged evolution

**Key results:** InternLM2-Chat-20B: AlpacaEval win rate 21.8 (SOTA among compared), GSM8K 79.6, MATH 32.4, HumanEval 67.7, MTBench 7.9, AlignBench 6.8, near-perfect 200k Needle-in-a-Haystack, beating GPT-3.5 on reasoning.

*Evolution:* COOL RLHF refines LLaMA2's separate helpful/harmless RMs into a single system-prompt-conditioned RM with multi-round online patching against reward hacking, anticipating later iterative/online RLHF.

### The Llama 3 Herd of Models
*2024 · report · `report_Llama3_2407.21783.txt` · arXiv [2407.21783](https://arxiv.org/abs/2407.21783)*

Pipeline is pre-train (15T tokens)→long-context pre-training (8K→128K)→six iterative rounds of post-training, with the 405B flagship used to improve the 8B/70B models. Each round collects fresh preference and SFT data and samples synthetic data from the latest checkpoints. Within a round the order is: train RM on all preference data→SFT (cross-entropy on rejection-sampled + synthetic + human data, LR 1e-5, 8.5K–9K steps)→DPO on the latest preference batches (LR 1e-5, β=0.1)→weight averaging across RM/SFT/DPO variants. Domain experts (a code expert via ~1T-token continued pre-training >85% code; a multilingual expert at 90% non-English) are branched from pre-training to gather higher-quality annotations. Rationale: keep the recipe simple and stable rather than adopt complex RL.

- Pretrain 15T → long-context 8K→128K → six iterative post-training rounds
- Per round: RM→SFT→DPO→weight averaging; 405B improves 8B/70B
- Domain experts branched from pre-training (code ~1T tokens, multilingual 90% non-EN)
- Stable SFT/RS/DPO deliberately favored over PPO

**Key results:** Llama 3 405B matches GPT-4 on human pairwise win rate (within margin of error) and is the best openly available model; HumanEval 89.0, MGSM 91.6 (best), 100% Needle-in-a-Haystack to 128K; 8B/70B best-in-class.

*Evolution:* Refines Llama-2's RM+SFT+rejection-sampling recipe and the DPO trend, scaling human preferences to millions with self-generated synthetic data across six rounds; the open 405B became a widely used base and distillation teacher.

### MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies
*2024 · report · `report_MiniCPM_2404.06395.txt` · arXiv [2404.06395](https://arxiv.org/abs/2404.06395)*

Training is ordered as three stages with Adam throughout: (1) stable training on ~1T coarse tokens (batch 3.93M, max LR 0.01, WSD scheduler); (2) decay/annealing via exponential decay (f=0.5^((s-S)/T), T=5000 steps ~20B tokens) over a mixture of pre-training and high-quality SFT data; (3) a separate SFT stage (~6B tokens) with LR aligned to annealing's end and again WSD decay. Ablations (A-1/A-2, B-1/B-2/B-3) show injecting high-quality data during decay beats adding it only in SFT, even with more SFT tokens. Aligned variants: MiniCPM-DPO adds one DPO epoch (LR 1e-5, cosine) after SFT; MiniCPM-128K reuses the last stable checkpoint with ABF then NTK RoPE and a 32K→128K curriculum; MiniCPM-MoE is built via Sparse Upcycling from the dense stable checkpoint then continued pre-train/decay/SFT for 130K steps.

- Three stages: stable (~1T) → anneal on high-quality mix → separate SFT (~6B tokens)
- WSD scheduler throughout; exponential decay T=5000 steps
- Anneal-with-high-quality beats SFT-only-with-more-tokens (key ablation)
- DPO/128K/MoE variants branch from the stable or SFT checkpoint

**Key results:** MiniCPM-2.4B surpasses Mistral-7B and Llama2-13B on average across C-Eval/CMMLU/MMLU/HumanEval/MBPP/GSM8K/MATH/BBH; MiniCPM-2.4B-DPO reaches MTBench 7.25, beating Llama2-70B-Chat; fitted scaling law yields ~192x compute-optimal data/model ratio vs Chinchilla's 20x.

*Evolution:* Recovers a much higher (~192x) compute-optimal ratio under modern overtrained configs; its WSD scheduler and "anneal on high-quality data" recipe presage later annealing-with-curated-data practices.

### LLM Pruning and Distillation in Practice: The Minitron Approach
*2024 · report · `report_Minitron_2408.11796.txt` · arXiv [2408.11796](https://arxiv.org/abs/2408.11796)*

A fixed four-stage ordering: (1) teacher correction — a lightweight ~100B-token fine-tuning of the unpruned model onto Nemotron-4 CT (120 warmup steps, peak LR at one-fifth of original, cosine decay) so the teacher adapts to the distillation distribution; (2) structured pruning of the corrected teacher into a student (width or depth, single-shot); (3) retraining via logit-only distillation (forward KL) to recover accuracy; (4) alignment with NeMo Aligner, identical for all models — math/code SFT, then instruction SFT, then two rounds of Reward-aware Preference Optimization (RPO). An ablation shows teacher correction can run in parallel with distillation rather than strictly beforehand, with no loss in final accuracy. Rationale: corrected supervision must precede or accompany distillation, and pruning must precede retraining.

- Four stages: teacher correction → pruning → logit-distillation retraining → alignment
- Teacher correction ~100B tokens, peak LR at 1/5 original, cosine decay
- Alignment: math/code SFT → instruction SFT → two RPO rounds (identical for all)
- Teacher correction can parallelize distillation with no accuracy loss

**Key results:** MN-Minitron-8B matches Llama 3.1 8B using 40x fewer tokens (380B vs 15T); Llama-3.1-Minitron-4B reaches near-teacher quality with 150x fewer tokens (94B), width variant beating depth on MMLU (60.5 vs 58.7) and GSM8K (41.24 vs 16.8).

*Evolution:* Extends the original Minitron pruning+distillation recipe to the private-teacher-data case via teacher correction and a downstream-task depth saliency metric, exemplifying the 2024 shift to cheap SLM derivation from frontier models.

### Mixtral of Experts
*2024 · report · `report_Mixtral_2401.04088.txt` · arXiv [2401.04088](https://arxiv.org/abs/2401.04088)*

Two stages for the chat model. First, the sparse-MoE base Mixtral 8x7B is pretrained on multilingual data with a 32k context. Second, Mixtral 8x7B – Instruct is produced by SFT on an instruction dataset followed by DPO on a paired feedback dataset. This is the standard pretrain→SFT→preference-optimization ordering; there is no RLHF/PPO loop, no multi-stage RL, and no self-play. Rationale: SFT+DPO yields a chat model beating GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B-chat on human evaluation.

- Two stages: SFT then DPO on the pretrained MoE base
- Standard pretrain→SFT→preference ordering
- No RLHF/PPO, no multi-stage RL, no self-play
- DPO chosen over PPO-based RLHF for instruction tuning

**Key results:** Mixtral 8x7B (13B active / 47B sparse) matches or beats Llama 2 70B and GPT-3.5 with 5× fewer active params: MMLU 70.6%, GSM8K 74.4% (8-shot maj@8), MATH 28.4% vs 13.8%; Mixtral 8x7B – Instruct reaches MT-Bench 8.30 and LMSys Elo 1121, best open-weights as of Dec 2023.

*Evolution:* Builds on the GShard/Shazeer sparse-MoE line and Mistral 7B, adopting DPO in place of PPO-RLHF; as one of the first open-weights MoE LLMs matching dense 70B quality at 5× lower active compute, it motivated the wave of open MoE releases.

### Phi-4 Technical Report
*2024 · report · `report_Phi-4_2412.08905.txt` · arXiv [2412.08905](https://arxiv.org/abs/2412.08905)*

Pretraining (~10T tokens, linear warmup/decay, peak LR 3e-4, weight decay 0.1, batch 5760) extends phi-3's two-phase strategy: phase 1 mostly filtered web, phase 2 mostly synthetic tokens plus ultra-filtered reasoning web; phi-4 raises the synthetic allocation. A midtraining stage (4K→16K context, 250B tokens, RoPE base raised to 250K, LR dropped 10x) uses 30% newly curated >4K/16K data plus 70% recall tokens from pretraining. Post-training is ordered SFT→DPO stage 1 (pivotal-token DPO)→DPO stage 2 (judge-guided, full-length pairs), all in chatml. Data-mixture design uses 1T-token ablations at 7B scale (high rank correlation with 14B) then transfers to 14B. Ablations show more epochs on synthetic data beat fresh web tokens, but pure-synthetic models underperform on knowledge, motivating retained web/acquired data.

- Pretrain 2-phase (web→synthetic+reasoning web); midtrain 4K→16K, 250B tokens
- Post-training: SFT→pivotal-token DPO→judge-guided full-length DPO
- Mixture tuned via 1T-token 7B ablations, transferred to 14B
- More synthetic epochs beat fresh web; pure-synthetic hurts knowledge benchmarks

**Key results:** phi-4 (14B): MATH 80.4 and GPQA 56.1, surpassing its teacher GPT-4o (74.6 / ~50.6) and beating Qwen-2.5-14B-Instruct on 9/12 benchmarks; Nov 2024 AMC-10/12 average ~90+/150, above GPT-4o and Llama-3.3-70B at far lower inference cost than long-CoT models.

*Evolution:* Extends the Phi "textbooks are all you need" synthetic-data thesis beyond GPT-4 distillation, showing curated synthetic data plus token-level (PTS) preference optimization can let a 14B model surpass its teacher on STEM reasoning at ~10x lower cost than long-CoT models.

### Qwen2.5 Technical Report
*2024 · report · `report_Qwen2.5_2412.15115.txt` · arXiv [2412.15115](https://arxiv.org/abs/2412.15115)*

Pre-training is two-phase: 4096-token context, then a final extension to 32,768 with RoPE base raised 10k→1M via ABF; Qwen2.5-Turbo uses progressive stages 32K→65K→131K→262K (RoPE base 10M, 40% current-max + 60% shorter). Post-training order is SFT→Offline RL→Online RL. SFT runs 2 epochs at length 32,768, LR annealed 7e-6→7e-7, weight decay 0.1, grad clip 1.0. Offline RL (DPO) uses ~150K pairs, 1 epoch, Online Merging Optimizer, LR 7e-7. Online RL uses GRPO: global batch 2048, 2048 samples/episode, 8 responses/query, queries ordered by reward-score variance (high variance first). Rationale: Offline RL targets capabilities hard for the RM where ground-truth checks exist; Online RL refines nuance. Turbo adds two-stage long-context SFT (short-only, then short+long to 262K); RL stays short-only because long-context RL is costly and long-context RMs are scarce, yet still aids long-query alignment.

- Pretrain 2-phase (4K→32K via ABF); Turbo progressive 32K→262K
- Post-training: SFT→offline DPO→online GRPO (ordered by reward variance)
- Offline RL handles RM-hard capabilities; online RL refines nuance
- Turbo: two-stage long-context SFT; RL kept short-only for cost/RM scarcity

**Key results:** Qwen2.5-72B-Instruct matches or exceeds Llama-3.1-405B-Instruct (~6x larger) on MMLU-redux 86.8 vs 81.6 plus MATH, MBPP, MultiPL-E, LiveCodeBench, Arena-Hard, MTBench; pretraining scaled 7T→18T tokens; Qwen2.5-Turbo reaches 100% on the 1M-token passkey retrieval task.

*Evolution:* Extends the Qwen2 lineage with a deliberate offline-DPO-then-online-GRPO two-stage RL design and aggressive 18T-token data scaling; its finding that RM-benchmark scores fail to predict downstream RL quality motivates better reward-model evaluation.

### Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models — The Story Goes On
*2024 · report · `report_Skywork-Math_2407.08348.txt` · arXiv [2407.08348](https://arxiv.org/abs/2407.08348)*

A two-stage SFT curriculum mirrors the two-stage data synthesis. Stage 1 (2.1M normal synthetic problems) fine-tunes a base 7B into an intermediate model grasping general math concepts and lifting accuracy on easy (MATH Level 1-2) problems. Stage 2 (additional 0.4M hard synthetic problems, seeded from MATH Level 4-5) fine-tunes the intermediate model into the final model, targeting hard (Level 3-5) problems. The ordering is explicitly motivated by curriculum learning (Bengio et al.): learn easy operations first, then progress to harder problems. Ablations confirm 2.1M+0.4M(hard) beats 7.5K+0.4M(hard), supporting the easy-to-hard progression. No continual pre-training, RLHF, or DPO is used — SFT is the only alignment stage.

- Two-stage SFT curriculum: 2.1M normal → +0.4M hard (seeded from MATH L4-5)
- Easy-to-hard ordering motivated by Bengio curriculum learning
- Ablation: 2.1M+0.4M(hard) beats 7.5K+0.4M(hard)
- SFT is the sole alignment stage; no RL/preference/continual pretraining

**Key results:** Skywork-Math-Mistral-7B (SFT only on 2.5M synthetic GPT-4 data) achieves 51.2% on MATH and 83.9% on GSM8K, SOTA among models <10B and surpassing an early GPT-4 on MATH; stage 2's 0.4M hard problems lift MATH Level 3-5 accuracy markedly.

*Evolution:* Builds on 2023-24 synthetic-SFT trends (MetaMath, WizardMath/Evol-Instruct, Xwin-Math) and pushes back against LIMA "less is more" by showing synthetic SFT scaling on 7B rivals 120B-token continual pre-training (DeepSeekMath).

### ReFT: Reasoning with Reinforced Fine-Tuning
*2024 · rl · `rl_ReFT_2401.08967.txt` · arXiv [2401.08967](https://arxiv.org/abs/2401.08967)*

Two-stage ordering on the same data. Stage 1 (warm-up): SFT on (question, CoT) tuples for 1-2 epochs on GSM8K/SVAMP (5 epochs MathQA-MCQ N-CoT, 10 epochs MathQAnumeric), batch 48, lr 1e-5, AdamW with 10% warm-up ratio, max length 1024. Stage 2 (RL): PPO for up to 300 epochs, lr 3e-7, batch 32, max question length 300 / sampling length 700. Rationale: the warm-up gives the policy enough skill to generate plausible CoTs so on-policy sampling in RL can explore many valid reasoning paths and learn from both correct and incorrect samples, whereas SFT only imitates a single annotation. The SFT baseline is trained 40 epochs (chosen for convergence). A sweep over warm-up epochs (1-4) shows all ReFT variants dip slightly post-warm-up then surpass SFT after ~8 epochs.

- Stage 1 warm-up SFT: 1-2 epochs (varies by dataset), lr 1e-5, batch 48
- Stage 2 RL: PPO up to 300 epochs, lr 3e-7, batch 32
- Same data both stages; warm-up enables on-policy exploration
- Warm-up sweep (1-4 epochs): brief dip then surpass SFT after ~8 epochs

**Key results:** CodeLLAMA-7B + ReFT reaches 75.28 P-CoT on GSM8K vs 63.68 for SFT (+11.6), and 81.2 with reward-model reranking, surpassing GPT-3.5-turbo (78.0) using only a 7B model; average gains over SFT across GSM8K/SVAMP/MathQA are +6.7 N-CoT and +7.4 P-CoT.

*Evolution:* An early-2024 demonstration that outcome-only RL (PPO with answer-derived rewards, no trained RM) on the same SFT data can substantially beat SFT for math, anticipating the GRPO/RLVR direction later popularized by DeepSeekMath and DeepSeek-R1.

### Technical Report on Slow Thinking with LLMs: II — Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems
*2024 · rl · `rl_STILL-2_2412.09413.txt` · arXiv [2412.09413](https://arxiv.org/abs/2412.09413)*

A three-phase "imitate, explore, self-improve" pipeline replaces the authors' earlier reward-model-plus-tree-search design. (1) Imitate: SFT on distilled long-form thought data elicits the slow-thinking format and capability. (2) Explore: the fine-tuned model performs multiple rollouts (≤20) on hard problems with ground-truth answers, collecting correct thought+solution trajectories. (3) Self-improve: an iterative loop refines the dataset — train M_t on D_{t-1}, generate new trajectories, add them to form D_t, repeat until the problem pool is exhausted. Two optimizers instantiate self-improvement: SFT as rejection sampling (length+perplexity filtering) and DPO using positive (correct, high-perplexity) vs negative (incorrect, low-perplexity) pairs, with an added SFT loss for stability and DPO seeded from the round-1 checkpoint. Cross-stage data mixes math, code, science, and puzzle domains; inference is single-pass with no tree search.

- Three phases: imitate (SFT) → explore (≤20 rollouts on hard problems) → self-improve loop
- Self-improve optimizers: rejection-sampling SFT and DPO (positive vs negative pairs)
- DPO seeded from round-1 checkpoint, with added SFT loss for stability
- Cross-stage mix of math/code/science/puzzle; single-pass inference, no search

**Key results:** STILL-2 (Qwen2.5-32B-Instruct + 3.9k distilled SFT) reaches 90.2% MATH-OAI and 46.7% AIME2024 (vs 80.0/13.3 backbone), approaching o1-preview's 85.5/44.6; exploration+self-improvement from only 1.1k seed instances lifts AIME from 33.3% to 40–46.7%.

*Evolution:* Revises the authors' own Nov-2024 reward-guided tree-search report, rejecting test-time search and domain-specific RMs in favor of distillation from newly-open R1/QwQ plus self-improvement, evidencing that small distilled long-CoT data plus rejection-sampling/DPO can elicit cross-domain slow thinking.

## 2025

### ACECODER: Acing Coder RL via Automated Test-Case Synthesis
*2025 · code · `code_ACECODER_2502.01718.txt` · arXiv [2502.01718](https://arxiv.org/abs/2502.01718)*

The pipeline flows from seed data to automated test-case synthesis/filtering (ACECODE-87K), then on-policy preference-pair construction, reward-model fine-tuning (ACECODE-RM-7B/32B), and finally Best-of-N inference or RL. RL is launched from three initial policies (Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B-Base, Qwen2.5-Coder-7B-Instruct) paired with two reward types (learned RM and rule-based binary pass rate), yielding six models. Echoing DeepSeek-R1/R1-Zero, the authors run RL directly from the coder base with no SFT, arguing SFT traps the policy in a local minimum; coding is treated as a verifiable task like math. RL trains on a subsampled "hard" slice (the 25% of questions with lower pass rate and higher variance) to keep rollouts diverse and informative. No multi-stage SFT→RL curriculum or warmup/anneal schedule is described.

- Stage order: data synth → preference pairs → RM → Best-of-N/RL
- RL-from-base (no SFT) on a verifiable code task, R1-style
- Six-model grid: 3 init policies × 2 reward types
- Hard-slice (bottom-25% pass-rate) subsampling for informative rollouts

**Key results:** ACECODE-RM-7B boosts Llama-3.1-8B-Instruct by +8.4 avg; R1-style RL from Qwen2.5-Coder-7B-Base with rule rewards gives +25% HumanEval-plus in 80 steps (48 H100 GPU-hours).

*Evolution:* Building on DeepSeek-R1's verifiable-reward RL-from-base recipe, it motivates later code-RL work showing very short RL runs from a base coder can rival distilled reasoning models.

### Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training
*2025 · code · `code_Agent-R_2501.11425.txt` · arXiv [2501.11425](https://arxiv.org/abs/2501.11425)*

Agent-R uses a two-phase iterative self-training loop repeated for three iterations. Phase I re-collects MCTS revision trajectories with the current actor; Phase II runs iterative SFT on mixed good+revision+general data. The curriculum is encoded by a rising alpha threshold (0.5 → 0.7 → 1.0), so good trajectories progressively converge to optimal and the actor learns to revise weaker-to-stronger errors within its capability. Iteration 1 uses 3 epochs as cold-start, then 1 epoch afterward to avoid overfitting; LR 2e-5, 3% warmup, cosine schedule, AdamW, seq len 8192, grad-clip 1, 8×A100-80GB. Multi-task training across all three environments consistently beats single-task, motivating joint data mixing across the iterations.

- Two-phase loop (trajectory generation → iterative SFT) × 3 iterations
- Difficulty encoded by rising alpha threshold 0.5→0.7→1.0
- Cold-start (3 epochs) in iter 1, 1 epoch thereafter; cosine + 3% warmup
- Joint multi-task mixing beats single-task, especially in later iterations

**Key results:** Llama-3.1-8B + Agent-R (iter 3) averages 70.71 across WebShop/SciWorld/TextCraft, beating Direct-Revision (62.36), ETO (65.12), and GPT-4o (+5.59%); revision length drops across iterations.

*Evolution:* Extends MCTS-driven agent training (Agent Q) and self-correction RL (SCoRe), motivating automatic step-level reflection-data construction and iterative self-play SFT without human/expert supervision.

### CODE I/O: Condensing Reasoning Patterns via Code Input-Output Prediction
*2025 · code · `code_CodeIO_2502.07316.txt` · arXiv [2502.07316](https://arxiv.org/abs/2502.07316)*

A two-stage order precedes instruction tuning. Stage 1 trains on CODE I/O (or CODE I/O++) as a continual-pretraining-like intermediate step to build a reasoning base; Stage 2 fine-tunes on ~1.18M in-house instruction-tuning samples (math, coding, writing, multilingual). The rationale is that CODE I/O is much larger than the IT set, so mixing would bias the distribution and under-train instruction following, so the order first builds reasoning then adapts. Stage 1: 1 epoch, constant LR (1e-5 for 7–16B, 4e-6 for Gemma 27B), batch 1024, seq len 4096, no warmup. Stage 2: 700 steps (~3 IT epochs), batch 1024, LR 3e-5 (1e-5 Gemma) with cosine decay. Ablations show two-stage beats single-stage; mixing helps LLaMA but not Qwen, so main runs keep stages separate.

- Stage 1: continual-pretraining-like CODE I/O (1 epoch, constant LR, no warmup)
- Stage 2: instruction tuning on ~1.18M samples (cosine decay)
- Rationale: CODE I/O too large to mix without biasing IT distribution
- Two-stage beats single-stage; mixing helps LLaMA but not Qwen

**Key results:** CODE I/O++ lifts the 14-benchmark reasoning average on all four base models vs single-stage IT (Qwen 57.7 vs 54.8, Gemma 61.5 vs 59.5), beating larger baselines like OpenMathInstruct2 14M.

*Evolution:* Builds on the code-pretraining-enhances-reasoning trend and execution-prediction learning, reacting against math-only reasoning-data scaling and positioning CODE I/O as a stronger reasoning base for RL-based reasoning models.

### A Self-Improving Coding Agent
*2025 · code · `code_SICA-self-improving_2504.15228.txt` · arXiv [2504.15228](https://arxiv.org/abs/2504.15228)*

SICA's "curriculum" is an iterative self-improvement loop rather than a pretrain→SFT→RL pipeline. It initializes a base agent A0, then for i=0..n−1 evaluates A_i on a benchmark set B, stores results in an archive, and selects the best-so-far agent A_i_hat (argmax utility) to act as the meta-agent that inspects the archive and edits the codebase to produce A_{i+1}. Within each meta-iteration an archive_explorer analyzes best/worst cases, reasoning sub-agents propose a feature, a developer implements and self-tests it, and a second developer independently verifies. The rationale is that coding-ability gains compound across iterations, and selecting the best agent as the next meta-agent (unlike fixed-meta-agent ADAS) lets improvements feed forward.

- Self-improvement loop: eval → archive → select best agent → meta-edit → next agent
- Best-so-far agent becomes next meta-agent (vs ADAS's fixed meta-agent)
- Per-iteration: archive explorer, feature proposer, implementer, verifier sub-agents
- Scaffolding-only (no weight updates); gains compound across iterations

**Key results:** SWE-Bench Verified (50-Q subset) 17%→53% over 15 iterations; file-editing 0.82→0.91 and LiveCodeBench ~0.65→0.71, with total run cost ~$7,000.

*Evolution:* Extends ADAS to a self-referential agent editing its full Python codebase, motivating 2025+ work on jointly fine-tuning foundation-model weights with the agent system and on automated benchmark generation.

### Beyond Scaling Law: A Data-Efficient Distillation Framework for Reasoning
*2025 · method · `method_DataEfficientDistillation_2508.09883.txt` · arXiv [2508.09883](https://arxiv.org/abs/2508.09883)*

DED is a data-construction pipeline feeding a single SFT distillation stage, not a multi-stage training pipeline. The three stages are data-side: (1) Teacher Selection (smoke-test several top LRMs by quick distillation+eval, selecting QwQ-32B over DeepSeek-R1, Qwen3-32B, Qwen3-235B-A22B), (2) Corpus Filtering (quality/correctness checks then pass-rate-based question compression), (3) Diverse Responses (Levenshtein-based trajectory selection). Training is plain SFT: LLaMA-Factory, 16,384-token context, batch 48, lr 1e-5, AdamW, 10 epochs on one 8×H800 node (~9h for 1k samples). A mixed math+code corpus is evaluated for cross-domain transfer. The rationale is to maximize reasoning gain per example and preserve OOD ability rather than scale corpus size.

- Data pipeline (not training): teacher select → filter → diverse trajectories
- Teacher QwQ-32B chosen via smoke-test over R1/Qwen3 variants
- Single SFT stage: 10 epochs, lr 1e-5, 8×H800, ~9h/1k samples
- Rationale: per-example reasoning gain and OOD preservation over corpus scale

**Key results:** NTele-R1-32B-Math distilled on ~0.8k examples scores 81.87% AIME 2024 and 77.29% AIME 2025, beating both teacher models and large-corpus baselines (Light-R1, Skywork-OR1).

*Evolution:* Extends the data-efficient SFT line (LIMA, s1, LIMO, Light-R1), reacting against the reasoning "scaling law" by importing RL-style pass-rate filtering and entropy-aware curation as cheaper levers than raw data scale.

### OpenSIR: Open-Ended Self-Improving Reasoner
*2025 · method · `method_OpenSIR_2511.00602.txt` · arXiv [2511.00602](https://arxiv.org/abs/2511.00602)*

OpenSIR runs a four-phase self-play loop per iteration: (1) problem generation (teacher), (2) solution sampling (student, G attempts/problem), (3) scoring (teacher novelty + student correctness), (4) GRPO model update over selected pairs, then pool expansion. A single policy alternates teacher/student roles and both co-evolve; freezing the teacher collapses solve-rate stability (variance ±4.49→±17.37) and drops accuracy (29.57→25.89). Difficulty follows a self-calibrated V-shape (3.4→3.0→3.8 on a 1–5 scale) and topics broaden from algebra/arithmetic/geometry to calculus, optimisation, trigonometry, statistics. Solve rate is held near a ~0.7 target via a triangular solvability reward over [s_min, s_max] (lower 0.5, upper 0.9). Both sequential (annotated→OpenSIR) and concurrent mixes beat either alone.

- Four-phase self-play loop: generate → sample → score → GRPO update, repeated
- Single policy alternates teacher/student roles; both co-evolve (freezing teacher fails)
- Self-calibrated V-shape difficulty (3.4→3.0→3.8); topics broaden over iterations
- ~0.7 solve-rate target via triangular solvability reward

**Key results:** OpenSIR averages +3.35 math and up to +4.79 general reasoning across four models, beating GRPO baselines on >7,000 annotated examples while bootstrapping from a single trivial seed.

*Evolution:* Reacts to RLVR (DeepSeek-R1, o1) and verifier-free self-play (Absolute Zero, R-Zero, Spiral), operationalizing open-endedness via joint difficulty calibration and embedding-based diversity.

### Toward Training Superintelligent Software Agents through Self-Play SWE-RL
*2025 · method · `method_SSR-SelfPlay-SWERL_2512.18552.txt` · arXiv [2512.18552](https://arxiv.org/abs/2512.18552)*

SSR's curriculum is the self-play loop itself, online and evolving rather than fixed. A single shared policy alternates between bug-injection and bug-solving roles; the bug distribution naturally tracks the agent's current capability, which static bug-generation pipelines cannot match. Control parameters (min_changed_files, min_passing_tests, min_failing_tests) tune bug complexity. Two injection strategies are compared—removal-only and removal+history (revert git-log changes)—with the latter giving the best curriculum. Failed initial solver attempts seed higher-order bugs (capped at second order to avoid overlap), introducing layered interdependent edits resembling real development. Training runs 150 global steps (~2.5B tokens) with a 30-step LR warmup to 3e-6, building on the CWM-sft 32B pre-RL checkpoint. Ablations show full self-play beats both injection-only and repair-only, confirming an evolving, jointly-trained task distribution is essential.

- Self-play loop: shared policy alternates bug-injection and bug-solving roles
- Evolving bug distribution tracks agent capability (static pipelines can't match)
- removal+history injection beats removal-only; higher-order bugs capped at 2nd order
- 150 steps (~2.5B tokens), 30-step warmup to 3e-6, from CWM-sft 32B

**Key results:** SSR (CWM-sft 32B base) achieves +10.4 points self-improvement on SWE-bench Verified and +7.8 on SWE-Bench Pro, beating the human-data baseline throughout training with no human-labeled issues or test suites.

*Evolution:* Extends the 2025 agentic-SWE-RL wave (SWE-RL, DeepSWE, CWM) and corpus-grounded self-play by removing human-curated issues/tests entirely, framing bug injection/repair as a self-play game over real repositories.

### The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models
*2025 · method · `method_ValleyOfCodeReasoning_2510.06101.txt` · arXiv [2510.06101](https://arxiv.org/abs/2510.06101)*

The study compares two checkpoints as distinct training stages rather than a multi-stage pipeline: the original non-reasoning instruct model and its 30K-distilled successor (Qwen2.5-30K). For Qwen2.5, eleven experiments are run: three scaling runs (1K/10K/30K) plus four feature-controlled runs (correctness and difficulty) repeated on each of the two checkpoints, isolating how the same data features behave across the non-reasoning phase and an intermediate reasoning-acquisition phase. Each run is a single SFT pass of 5 epochs with AdamW, global batch 128, lr 8e-5, warmup ratio 0.10, evaluated on the final checkpoint. Llama3.1 receives only the three scaling runs. There is no preference or RL stage; staging is used purely to attribute data-feature effects to the model's current learning phase.

- Two checkpoints as stages: non-reasoning instruct vs 30K-distilled successor
- 11 Qwen2.5 experiments: 3 scaling (1K/10K/30K) + 4 feature-controlled × 2 checkpoints
- Single SFT pass: 5 epochs, lr 8e-5, warmup 0.10, batch 128
- No preference/RL stage; staging isolates data-feature effects per learning phase

**Key results:** Qwen2.5-7B-Instruct LCB Pass@1: 0.126 baseline → 0.055 at 1K (valley) → 0.188 at 10K → 0.264 at 30K; Llama3.1-8B reaches 0.299 at 30K, doubling its 10K score.

*Evolution:* Builds on data-efficient reasoning distillation (s1, LIMO) and large-scale coding distillation, reacting to their focus on data endpoints by exposing the non-monotonic "valley" in small models and motivating stage-aware data curation.

### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
*2025 · report · `report_DeepSeek-R1_2501.12948.txt` · arXiv [2501.12948](https://arxiv.org/abs/2501.12948)*

DeepSeek-R1-Zero skips SFT entirely and applies GRPO RL directly to DeepSeek-V3-Base (671B total / 37B active MoE) with only rule-based rewards. DeepSeek-R1 uses a four-stage pipeline: (1) cold-start SFT on a few thousand long-CoT traces to seed a first-person reasoning style; (2) RL stage 1 = reasoning RL with GRPO plus a language-consistency reward (target-language word fraction) to curb language mixing; (3) rejection sampling + SFT on ~800K samples (600K reasoning + 200K non-reasoning) to broaden writing/QA/code-engineering ability; (4) RL stage 2 = mixed reasoning (rule-based) and general data (model-based reward) at temp 0.7, with general/preference rewards only in the final 400 of 1,700 steps to limit reward hacking, and the reference policy refreshed every 400 steps. The rationale is that RL explores non-human trajectories SFT cannot, while SFT covers tasks lacking reliable verifiers; both are indispensable.

- R1-Zero: GRPO RL directly on base, no SFT, rule-based rewards only
- R1 four-stage: cold-start SFT → reasoning RL (+language-consistency reward) → rejection-sampling SFT (~800K) → mixed RL stage 2
- Stage-2 RL applies general/preference rewards only in final 400/1,700 steps to curb hacking
- Reference policy refreshed every 400 steps; rationale: RL explores, SFT covers unverifiable tasks

**Key results:** DeepSeek-R1 scores 79.8% pass@1 on AIME 2024 (86.7% cons@16) and 97.3 MATH-500, matching o1-1217; R1-Distill-Qwen-32B hits 72.6 AIME / 94.3 MATH, beating RL-only Qwen2.5-32B-Zero.

*Evolution:* Builds on GRPO and outcome-only RL/STaR, reacting against SFT-first RLHF by showing pure RL on a strong base elicits emergent long-CoT reasoning ("aha moment") without human traces, catalyzing the open-source reasoning-RL wave.

### DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
*2025 · report · `report_DeepSeek-V3.2_2512.02556.txt` · arXiv [2512.02556](https://arxiv.org/abs/2512.02556)*

The order is: (1) continued pre-training in two stages — a Dense Warm-up stage (1000 steps, 2.1B tokens, only the lightning indexer trained, dense attention kept, KL loss aligning indexer to main attention) then a Sparse Training stage (15,000 steps, 943.7B tokens, all parameters, 2048 KV tokens selected per query); (2) post-training combining specialist distillation followed by a single mixed RL stage. The mixed stage merges reasoning, agent, and human-alignment training simultaneously rather than sequentially, explicitly to avoid catastrophic forgetting from multi-stage training. A cold-start phase first unifies reasoning and tool-use within single trajectories (DeepSeek-V3 methodology), then large-scale synthesized agentic data drives RL. DeepSeek-V3.2 runs thousands of continued RL steps from specialist-distilled data; Speciale is trained only on reasoning data with reduced length penalty. RL compute exceeds 10% of pre-training cost.

- Continued pre-training: dense warm-up (indexer only, 2.1B tok) → sparse (943.7B tok, all params)
- Post-training: specialist distillation → single mixed RL stage (reasoning+agent+alignment together)
- Mixed stage deliberately avoids sequential multi-stage training to prevent forgetting
- Cold-start unifies reasoning+tool-use per trajectory; RL compute >10% of pre-training

**Key results:** DeepSeek-V3.2-Speciale: AIME 2025 96.0 Pass@1, HMMT Feb 99.2, Codeforces rating 2701, gold medals at IMO/IOI/ICPC-WF/CMO 2025; Thinking matches GPT-5-High on reasoning.

*Evolution:* Builds on DeepSeek-R1's RL-for-reasoning and V3/V3.1's MoE+MLA, reacting to the open-vs-closed gap by scaling post-training RL compute beyond 10% of pre-training and synthesizing agentic environments at scale.

### GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models
*2025 · report · `report_GLM-4.5_2508.06471.txt` · arXiv [2508.06471](https://arxiv.org/abs/2508.06471)*

The multi-stage recipe is: pre-training (15T general + 7T code/reasoning continual, 4K context) → mid-training (500B repo-level code at 32K, 500B synthetic reasoning, 100B long-context & agent data at 128K) → post-training. Post-training has two stages: Stage 1 builds three expert models (Reasoning, Agent, General chat) via cold-start SFT then domain-specific RL; Stage 2 unifies them via Overall SFT self-distillation into a hybrid reasoning generalist, followed by General RL. Within Reasoning RL, a two-stage difficulty curriculum (moderate data at samples_per_prompt=16, then extremely hard pass@8=0/pass@512>0 data) is used. Notably, single-stage RL directly at 64K output length beats progressive length-increase schedules. Agentic RL alternates RL with iterative self-distillation. General RL runs Holistic, Instruction-Following, Function-Calling (step-wise then end-to-end multi-turn), and Pathology RL sequentially. Best-fit packing is used only in mid-training.

- Pipeline: pre-train (15T+7T) → mid-train (repo-code/synthetic-reason/long-ctx) → post-train
- Post-train Stage 1: 3 expert models (SFT+domain RL); Stage 2: self-distill unify + General RL
- Reasoning RL two-stage difficulty: moderate → extremely hard (pass@8=0/pass@512>0)
- Single-stage 64K RL beats progressive length-increase schedules

**Key results:** GLM-4.5 (355B/32B-active MoE) scores 91.0% AIME 24, 70.1% TAU-Bench, 64.2% SWE-bench Verified, ranking 3rd overall and 2nd on agentic benchmarks among all evaluated models.

*Evolution:* Builds on the 2024-25 open MoE and reasoning-RL wave (DeepSeek-V3/R1, Kimi K2, Qwen3), distinguishing itself by unifying agentic/reasoning/coding capabilities via expert-iteration + self-distillation plus hybrid thinking/non-thinking modes.

### Gemma 3 Technical Report
*2025 · report · `report_Gemma3_2503.19786.txt` · arXiv [2503.19786](https://arxiv.org/abs/2503.19786)*

Training proceeds pre-train (with distillation) → long-context extension → instruction tuning (distillation + RL) → optional QAT. Pre-training starts at 32K sequences, then 4B/12B/27B are scaled to 128K at the end of pre-training via RoPE rescaling (factor 8), with the global-attention RoPE base raised from 10k to 1M while local layers stay at 10k. Instruction tuning combines knowledge distillation from a large IT teacher with an RL fine-tuning phase built on improved BOND, WARM, and WARP; reward functions target helpfulness, math, coding, reasoning, instruction-following, and multilinguality while minimizing harmfulness. SFT and RLHF are also used for safety alignment. Finally, Quantization Aware Training (~5,000 steps) produces per-channel int4, per-block int4, and switched-fp8 checkpoints from the non-quantized models.

- Order: pre-train (distillation) → long-ctx extension → IT (distillation + RL) → optional QAT
- Long-context via RoPE rescaling (factor 8), global base 10k→1M, local stays 10k
- IT phase = distillation + RL (BOND/WARM/WARP); multi-objective rewards
- QAT ~5,000 steps yields int4 / switched-fp8 checkpoints

**Key results:** Gemma-3-27B-IT scores Chatbot Arena Elo 1338 (~rank 9), MATH 89.0, GSM8K 95.9, HumanEval 87.8, MMLU-Pro 67.5, IFEval 90.4, comparable to Gemini-1.5-Pro and ahead of DeepSeek-V3 (1318).

*Evolution:* Builds on Gemma 2's distillation-based pre-training, folding in the 2024 RL-alignment toolkit (BOND/WARM/WARP) and verifiable-reward signals popularized by DeepSeek-R1/Tulu 3, packaging distillation-plus-RL into lightweight multimodal open models.

### Kimi K1.5: Scaling Reinforcement Learning with LLMs
*2025 · report · `report_Kimi-K1.5_2501.12599.txt` · arXiv [2501.12599](https://arxiv.org/abs/2501.12599)*

The pipeline is pretraining → vanilla SFT → long-CoT SFT → RL, with long2short transfer as a final option. Pretraining has three sub-stages: vision-language pretraining (language-only first, then interleaved up to 30%); cooldown (curated + synthetic QA via rejection sampling); and long-context activation (max length 4,096 → 32,768 → 131,072, RoPE base 1e6, 40% full / 60% partial attention). Vanilla SFT runs 1 epoch at 32k then 1 epoch at 128k, with LR 2e-5 → 2e-6 then re-warmup to 1e-5 → 1e-6. Long-CoT SFT uses a small prompt-engineered warmup set teaching planning, evaluation, reflection, and exploration. RL uses curriculum sampling (easy → hard) then prioritized sampling (proportional to 1 − success rate), and warms up the length penalty (none initially, then constant). Long2short options: model merging, shortest rejection sampling, DPO, and a second long2short RL phase with reduced max rollout length.

- Pipeline: pretraining (3 sub-stages) → vanilla SFT → long-CoT SFT → RL → optional long2short
- SFT: 1 epoch @32k then 1 @128k, LR re-warmup between phases
- Long-CoT SFT warmup set teaches planning/evaluation/reflection/exploration
- RL: easy→hard curriculum then prioritized (1−success-rate) sampling; length penalty warmed up

**Key results:** Kimi k1.5 long-CoT: 77.5 AIME 2024 Pass@1, 96.2 MATH-500 EM, 94th percentile Codeforces, 74.9 MathVista — matching OpenAI o1; short-CoT outperforms GPT-4o/Claude 3.5 Sonnet by up to +550%.

*Evolution:* Builds on RLHF and OpenAI o1's long-CoT RL, formalizing RL as online mirror descent and arguing context-length scaling suffices without MCTS, value functions, or process reward models; motivates long2short distillation.

### Kimi K2: Open Agentic Intelligence (Technical Report of Kimi K2)
*2025 · report · `report_Kimi-K2_2507.20534.txt` · arXiv [2507.20534](https://arxiv.org/abs/2507.20534)*

The multi-stage pipeline is pre-training → annealing + long-context activation → SFT → joint RL. Pre-training uses the WSD schedule (500-step warmup, 10T tokens at constant LR 2e-4, then 5.5T cosine decay to 2e-5; weight decay 0.1; 67M-token global batch), followed by an annealing phase (400B tokens at 4k, then 60B at 32k) and YaRN extension to 128k context. SFT uses the Muon optimizer (recommended for Muon-pretrained checkpoints) and initializes the critic. RL scales K1.5's recipe in task diversity and FLOPs within a Gym-like framework, jointly combining verifiable rewards and self-critic rubric rewards. Within-stage schedule tricks include temperature decay (high→low for exploration→exploitation), per-task budget control, and a PTX auxiliary loss to prevent forgetting of curated high-quality data. The critic is refined in a closed loop using on-policy verifiable-reward rollouts distilled into subjective judgments.

- Pipeline: pre-train (WSD) → anneal + long-ctx → SFT (Muon) → joint RL
- WSD: 500-step warmup, 10T constant 2e-4, 5.5T cosine→2e-5; anneal 400B@4k then 60B@32k
- Joint RL fuses verifiable + self-critic rubric rewards; temperature decay high→low
- PTX auxiliary loss prevents forgetting; critic refined via on-policy rollout distillation

**Key results:** Kimi-K2-Instruct (1.04T MoE, 32B activated) scores 65.8 SWE-bench Verified (71.6 multi-attempt), 76.5 ACEBench, 49.5 AIME 2025, ranked #1 open-source / #5 overall on LMSYS Arena.

*Evolution:* Builds on DeepSeek-V3's sparse-MoE+MLA, Kimi K1.5's RL scaling, and the RLVR wave, pushing token-efficient Muon training and agentic RL with synthetic tool-use environments to trillion-parameter scale.

### Llama-Nemotron: Efficient Reasoning Models
*2025 · report · `report_Llama-Nemotron_2505.00949.txt` · arXiv [2505.00949](https://arxiv.org/abs/2505.00949)*

Models are built in five ordered stages: (1) NAS via the Puzzle framework from Llama 3 Instruct plus FFN Fusion; (2) recovery training = knowledge distillation (LN-Super 40B tokens; LN-Ultra 65B) then continued pretraining (LN-Ultra 88B tokens on Nemotron-H phase-4 data); (3) SFT on standard instruction data + DeepSeek-R1 reasoning traces, installing a "detailed thinking on/off" toggle; (4) large-scale RL on math/STEM (GRPO, LN-Ultra only) to surpass the teacher; (5) a short alignment phase = instruction-following RL (RLOO) then RLHF (online RPO/GRPO). LN-Nano uses a three-step SFT sub-pipeline (reasoning-only → reasoning+non-reasoning → chat/IF/tool). LN-Ultra RL uses a progressive-batching curriculum: a Gaussian target pass-rate distribution sliding from easy to hard, with prompts of pass-rate ≥0.75 discarded. RL is initialized from an earlier (lower-scoring) SFT checkpoint to improve final RL outcomes.

- Five stages: NAS+FFN-Fusion → recovery (distill+continued PT) → SFT (+R1 traces) → reasoning RL (GRPO) → alignment RL (RLOO+RLHF)
- SFT installs "thinking on/off" toggle; LN-Nano uses 3-step SFT sub-pipeline
- LN-Ultra RL: progressive-batching, Gaussian target pass-rate sliding easy→hard, discard ≥0.75
- RL initialized from earlier (lower-scoring) SFT checkpoint for better final RL

**Key results:** LN-Ultra (253B) reaches GPQA-Diamond 76.0% and AIME24 80.8%, beating DeepSeek-R1 while fitting on one 8×H100 node; GRPO raises LN-Ultra GPQA-D from 66.4 (SFT) to 76.0.

*Evolution:* Builds on DeepSeek-R1's distill-then-RLVR recipe, reacting to inference-cost problems of long-CoT models by marrying NAS/FFN-Fusion efficiency and a user-facing reasoning toggle, motivating controllable efficient reasoning and curriculum-driven RLVR to surpass the teacher.

### Qwen3 Technical Report
*2025 · report · `report_Qwen3_2505.09388.txt` · arXiv [2505.09388](https://arxiv.org/abs/2505.09388)*

Pre-training is three stages: S1 General (30T+ tokens, seq len 4096), S2 Reasoning (5T higher-quality STEM/coding/synthetic tokens, accelerated LR decay), and Long-Context (hundreds of billions at 32768, 75% text 16K–32K and 25% 4K–16K, using ABF RoPE plus YaRN/DCA for 4× inference length). Post-training for flagship models (235B-A22B, 32B) is four stages: (1) Long-CoT cold-start SFT, (2) Reasoning RL via GRPO, (3) Thinking Mode Fusion via continual SFT, (4) General RL. The first two build thinking; the last two add non-thinking and broad alignment. Lightweight models skip this and use Strong-to-Weak Distillation (off-policy then on-policy logit distillation) at ~1/10 the GPU hours, with better pass@1/pass@64.

- Pre-train 3 stages: S1 General (30T) → S2 Reasoning (5T, faster LR decay) → Long-Context (YaRN/DCA, 4× inference length)
- Post-train 4 stages: cold-start SFT → Reasoning RL (GRPO) → Thinking-Mode Fusion (SFT) → General RL
- First two build thinking; last two add non-thinking + broad alignment
- Lightweight models use Strong-to-Weak Distillation (off- then on-policy logits) at ~1/10 GPU-hours

**Key results:** Qwen3-235B-A22B (Thinking) scores 85.7 AIME'24, 81.5 AIME'25, 70.7 LiveCodeBench v5, 2056 Codeforces, outperforming DeepSeek-R1 on 17/23 benchmarks; on-policy distillation beats RL on Qwen3-8B (74.4 vs 67.6) at ~1/10 GPU-hours.

*Evolution:* Builds on Qwen2.5 and the 2024-25 reasoning wave (o1, DeepSeek-R1, QwQ), reacting to the cost of separate chat/reasoning models by unifying both modes with a controllable thinking budget; its Strong-to-Weak distillation motivates efficient small-model reasoning.

### Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning
*2025 · report · `report_Seed1.5-Thinking_2504.13914.txt` · arXiv [2504.13914](https://arxiv.org/abs/2504.13914)*

The pipeline is SFT then RL. SFT (400k instances, truncated to 32k tokens) trains the base model for 2 epochs with cosine-decay LR peaking at 2e-5 decaying to 2e-6; it improves readability and reduces hallucination/harm versus starting RL from base, but too much non-CoT SFT data hurts later exploration. RL is a unified stage fusing three data types: verifier-scored verifiable data, reward-model-scored general data, and hybrid data combining both. Puzzle difficulty is gradually raised based on model performance. An ablation shows initializing RL with a rejection-fine-tuned (RFT) model saturates faster but ends lower (AIME avg@32 54% vs 58% baseline). Algorithm ranking (VAPO > DAPO) is consistent across Qwen-32B-Dense and Seed-150B-MoE, so the smaller dense model serves as an RL-algorithm proxy.

- Pipeline: SFT (400k, 32k cap, 2 epochs, cosine 2e-5→2e-6) → unified RL
- RL fuses verifier-scored, RM-scored, and hybrid data in one stage
- Difficulty raised gradually based on model performance
- RFT-init RL saturates faster but ends lower (AIME avg@32 54 vs 58 baseline)
- VAPO > DAPO consistently, so Qwen-32B-Dense proxies the MoE for RL-algorithm choice

**Key results:** Seed1.5-Thinking (20B-active/200B-total MoE) scores 86.7 on AIME 2024 (matching o3-mini-high), 55.0 on Codeforces avg@8, and 77.3 on GPQA, with +8.0% win rate over DeepSeek R1 on non-reasoning tasks.

*Evolution:* Builds on the o1/R1 test-time-scaling wave and the authors' own VAPO/DAPO work to make long-CoT RL stable and reproducible at MoE scale, responding to the 2025 need for harder, less gameable reasoning benchmarks.

### AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning
*2025 · rl · `rl_AceReason-Nemotron_2505.16400.txt` · arXiv [2505.16400](https://arxiv.org/abs/2505.16400)*

Training starts from distilled SFT checkpoints DeepSeek-R1-Distill-Qwen2.5-7B/14B. The recipe runs domain-separated RL: math-only RL first, then code-only RL, originally motivated because code verification (~552s/1,024 instances) is far slower than math (~3.9s/1,024). Math-only RL uses stage-wise maximum response length extension 8K→16K→24K→32K, with harder prompts (model pass rate >6/16 filtered out) introduced at the 24K and 32K stages. Code-only RL has two stages: Stage 1 (24K length, temp 0.6, 8 rollouts, difficulty up to level 5 for 7B / level 7 for 14B) and Stage 2 (32K length, full problem set, epoch-wise filtering of easy problems, temperature rising 0.6→1.0 and rollouts 8→16). The final integrated curriculum is math-RL (8K→24K) → code-RL (24K→32K) → math-RL at 32K, found slightly more effective than math 8K→32K then code 24K→32K. Rationale: cross-domain generalization without the catastrophic forgetting seen in domain-specific SFT.

- Init from R1-distilled SFT 7B/14B; domain-separated math-RL then code-RL
- Math RL: stage-wise length 8K→16K→24K→32K; hard prompts introduced at 24K/32K
- Code RL 2 stages: 24K (level 5/7) → 32K (full set, temp 0.6→1.0, rollouts 8→16)
- Best integrated curriculum: math 8K→24K → code 24K→32K → math 32K

**Key results:** AceReason-Nemotron-14B reaches 78.6/67.4 avg@64 on AIME24/25 and 61.1/54.9 avg@8 on LiveCodeBench v5/v6, surpassing DeepSeek-R1-Distill-Qwen-32B; math-only RL alone gives +14.6%/+17.2% on AIME 2025 (7B/14B).

*Evolution:* Builds on DeepSeek-R1's GRPO-plus-rule-verification recipe, directly countering the DeepSeek-R1/Llama-Nemotron claim that distillation beats RL for small/mid models and motivating sequential multi-domain RL curricula and RL-on-distilled-model recipes.

### Understanding R1-Zero-Like Training: A Critical Perspective
*2025 · rl · `rl_DrGRPO_2503.20783.txt` · arXiv [2503.20783](https://arxiv.org/abs/2503.20783)*

The paper studies the R1-Zero-like paradigm: RL applied directly to a base model with no SFT, no KL term (beta=0), and a rule-based verifier reward. The main recipe is Qwen2.5-Math-7B + Qwen-Math template → Dr. GRPO RL on MATH level 3-5 (~27h on 8×A100). For weak base models (Llama-3.2-3B) they insert a continual math-pretraining stage (FineMath, then concatenated NuminaQA) before RL to raise the RL ceiling. A template-by-question-set study shows RL can reconstruct capability destroyed by template mismatch and that even small out-of-distribution sets (GSM-8K) can induce reasoning when the template matches. Multi-stage curriculum ordering itself is not the primary focus.

- R1-Zero-like: RL direct on base, no SFT, no KL (beta=0), rule-based verifier
- Main recipe: Qwen2.5-Math-7B + Dr. GRPO on MATH L3-5 (~27h, 8×A100)
- Weak bases get a continual math-pretraining stage (FineMath + NuminaQA) before RL
- Template/question-set study: RL reconstructs template-mismatched capability

**Key results:** Oat-Zero-7B (Qwen2.5-Math-7B + Dr. GRPO on MATH L3-5) achieves 43.3% AIME 2024 and 51.4% avg over 5 math benchmarks, a new SOTA for 7B R1-Zero-like training, in 27h on 8×A100.

*Evolution:* Builds on DeepSeek-R1-Zero's RL-without-SFT paradigm and DeepSeekMath's GRPO, reacting critically to the "aha moment"/long-CoT narrative by showing pretraining and GRPO length/difficulty biases inflate response length; Dr. GRPO prefigured later RLVR correctives (DAPO-style decoupled clipping, overlong filtering).

### LIMO: Less is More for Reasoning
*2025 · rl · `rl_LIMO_2502.03387.txt` · arXiv [2502.03387](https://arxiv.org/abs/2502.03387)*

A minimal, single-stage curriculum: plain supervised fine-tuning of Qwen2.5-32B-Instruct on the 800-sample LIMO set, with no pre-SFT warmup stage, no preference/RL stage, and no distillation loop. Full-parameter SFT runs under DeepSpeed ZeRO-3 with FlashAttention-2, 16,384-token sequence cap, learning rate 5.0e-6 with cosine decay (warmup deliberately omitted for rapid adaptation), 15 epochs, batch size 64. The rationale is that a knowledge-rich pretrained backbone plus long-chain test-time compute only needs a few high-quality cognitive templates to elicit reasoning, so a multi-stage pipeline is unnecessary.

- Single-stage SFT only: Qwen2.5-32B-Instruct on 800 samples, no RL/preference/distill
- Full-param SFT, ZeRO-3 + FlashAttention-2, seq cap 16,384, 15 epochs, batch 64
- lr 5.0e-6 cosine decay; warmup deliberately omitted for rapid adaptation
- Rationale: knowledge-rich backbone + test-time compute needs few templates, not a pipeline

**Key results:** LIMO scores 63.3% AIME24 (vs o1-preview 44.6%, QwQ-32B-Preview 50.0%, NuminaMath-100k SFT 6.5%) and 95.6% MATH500, +45.8% absolute OOD gain, using ~1% of prior approaches' data; 400 samples already lift AIME24 from 16.5 to 57.5.

*Evolution:* Extends LIMA's "less is more" alignment lesson from instruction-following to mathematical reasoning, reacting against data-heavy SFT and complementing RLVR by showing minimal high-quality SFT in knowledge-rich backbones can rival far larger pipelines.

### Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning
*2025 · rl · `rl_Logic-RL_2502.14768.txt` · arXiv [2502.14768](https://arxiv.org/abs/2502.14768)*

The main recipe is single-stage rule-based RL with no SFT warmup: Qwen2.5-7B-Instruct-1M is trained directly for 3,600 steps at constant LR 4e-7 and temperature 0.7 on mixed-difficulty (3-7 person) K&K puzzles. Ablations probe staging: curriculum learning (one epoch each on progressively harder 3-to-7 person sets) is compared against mixed-difficulty training; curriculum gives slightly higher intermediate test scores but the gap is statistically negligible, so its practical necessity is not conclusive. A cold-start study shows base vs instruct checkpoints produce nearly identical RL curves (instruct only marginally better), so an SFT/instruct warm start is a bonus, not a necessity. An RFT (rejection-sampled SFT) vs RL comparison shows SFT memorizes while RL generalizes, supporting the choice of pure RL over an SFT-then-RL pipeline.

- Single-stage rule-based RL, no SFT warmup: 3,600 steps, constant LR 4e-7, temp 0.7
- Curriculum (easy→hard 3-7 person) vs mixed-difficulty: gap statistically negligible
- Cold-start: base vs instruct give nearly identical RL curves (instruct marginally better)
- RFT memorizes (LiMem rise) vs RL generalizes, supporting pure RL over SFT-then-RL

**Key results:** Qwen2.5-7B-Instruct-1M + Logic-RL on ~5K K&K puzzles: K&K average 0.19→0.89, AIME +125%, AMC +38%; REINFORCE++ beats GRPO across nearly all metrics while PPO is 138% slower.

*Evolution:* An early R1-replication study using a controlled synthetic logic corpus, showing R1-like emergent reasoning in a 7B model with no discrete "aha moment" and that curriculum/cold-start matter little, motivating later work on long-to-short CoT compression and KL-constraint relaxation.

### Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
*2025 · rl · `rl_Open-Reasoner-Zero_2503.24290.txt` · arXiv [2503.24290](https://arxiv.org/abs/2503.24290)*

Reasoner-Zero training launches large-scale RL directly on Qwen2.5 base models {0.5B, 1.5B, 7B, 32B}, bypassing any SFT or distillation warm-up, with a prompt template (think/answer tags) that elicits step-by-step reasoning. Policy and critic are both initialized from the base model (value head randomly initialized), weights not shared. The policy uses strict on-policy optimization (one update per generation), while the critic does 12 mini-batch updates per iteration. For the 32B variant an extra "annealing" stage reuses 13k prompts the model solved fewer than 4/64 times in the first 1100 steps, training 100 more steps with linear LR decay to 3e-7. A two-stage distilled path initializes from DeepSeek-R1-Distill-Qwen-14B on the same 13k hard prompts for 300 iterations.

- RL direct on Qwen2.5 base {0.5B–32B}, no SFT/distill warm-up; think/answer template
- Policy and critic both init from base (value head random), not shared
- Strict on-policy policy (1 update/gen); critic does 12 mini-batch updates/iter
- 32B annealing stage: 13k hard prompts (solved <4/64), 100 steps, linear decay to 3e-7

**Key results:** ORZ-32B matches/exceeds DeepSeek-R1-Zero-Qwen-32B on AIME2024 (48.1 vs 47.0), MATH500 (92.2 vs 91.6), GPQA (55.5 vs 55.0) using 1/10 the steps; ORZ-R1-Distill-Qwen-14B reaches AIME2024 75.2, surpassing R1-Distill-Qwen-32B (72.6).

*Evolution:* Builds on DeepSeek-R1-Zero's Reasoner-Zero paradigm (RL directly on a base model) but reacts against GRPO by reverting to vanilla PPO with a learned critic; as the first fully open-source large-scale reasoning-RL framework (0.5B–32B), it democratizes the R1-Zero scaling trend.

### Process Reinforcement through Implicit Rewards
*2025 · rl · `rl_PRIME-ProcessReinforcement_2502.01456.txt` · arXiv [2502.01456](https://arxiv.org/abs/2502.01456)*

A two-stage pipeline. (1) Full-parameter SFT warmup of Qwen2.5-Math-7B-Base on 230K action-centric-CoT data (3 epochs, lr 1e-5, AdamW, cosine annealing, warmup 0.1, batch 96) producing Eurus-2-7B-SFT. (2) RL via PRIME, initializing both policy and Implicit PRM from the SFT model (reference model = SFT). There is no dedicated PRM training stage — the PRM is updated online concurrently with the policy using only outcome labels. Default advantage estimator is RLOO (leave-one-out) with a PPO clip loss, 4 rollouts/prompt, KL coefficient 0. A "Zero" variant skips SFT and runs RL directly from the base model (7B and 32B), converging faster. Rationale: SFT instills reasoning patterns; RL refines them with verifiable plus dense rewards; using separate SFT and RL datasets broadens exploration.

- Two-stage: SFT warmup (230K action-centric-CoT, 3 epochs, cosine, warmup 0.1) → RL via PRIME
- No dedicated PRM stage; Implicit PRM updated online with policy using outcome labels only
- RLOO advantage + PPO clip, 4 rollouts/prompt, KL coeff 0; reference = SFT model
- "Zero" variant skips SFT, RL direct from base (7B/32B), converging faster

**Key results:** Eurus-2-7B-PRIME (from Qwen2.5-Math-7B-Base): 26.7% pass@1 AIME 2024 vs Qwen2.5-Math-7B-Instruct's 13.3%, +15.1% avg over the SFT model across 7 benchmarks, with 2.5× sample efficiency and +6.9% over outcome-only RLOO.

*Evolution:* Builds on implicit reward modeling and reacts to DeepSeek-R1/Kimi K1.5's conclusion that PRMs are impractical at scale, demonstrating that dense, online-updatable process rewards can be both scalable and beneficial, motivating later process-reward RL methods.

### Skywork Open Reasoner 1 Technical Report
*2025 · rl · `rl_Skywork-OR1_2505.22312.txt` · arXiv [2505.22312](https://arxiv.org/abs/2505.22312)*

All models start from DeepSeek-R1-Distill-Qwen SFT checkpoints (7B, 32B) and apply multi-stage RL with progressively increasing context length T, inspired by DeepScaleR. Skywork-OR1-Math-7B uses 4 stages (8K→16K→32K→32K, steps 0-740/740-1740/1740-2080/2080-2160; checkpoint 2160). Skywork-OR1-7B uses 2 stages (16K→32K, 0-660/660-1320; checkpoint 1320). Skywork-OR1-32B uses 2 stages (16K→24K, 0-760/760-1130; checkpoint 1000). Short early context halves inference cost (~100 hours saved over 1000 steps) while reaching the same final accuracy; later stages restore long-CoT scaling. Cross-stage online filtering discards prompts solved perfectly in the prior stage so each stage trains on challenging problems. The pipeline is SFT-distill → multi-stage RL (no extra SFT/preference stage); Math-7B used two gradient steps per training step while 7B/32B are strictly on-policy.

- Init from R1-Distill-Qwen SFT; multi-stage RL with growing context length T (DeepScaleR-style)
- Math-7B: 4 stages 8K→16K→32K→32K; 7B: 2 stages 16K→32K; 32B: 2 stages 16K→24K
- Short early context saves ~100h over 1000 steps at equal final accuracy
- Cross-stage filtering discards prompts solved perfectly in the prior stage

**Key results:** Skywork-OR1-32B reaches 82.2 AIME24, 73.3 AIME25, 63.0 LiveCodeBench (avg@4), surpassing DeepSeek-R1 and Qwen3-32B on math; average over AIME24/25/LCB improves 57.8%→72.8% (+15.0) for 32B.

*Evolution:* Builds on DeepSeek-R1's rule-reward RL and DeepScaleR's multi-stage context-length curriculum, engaging with DAPO's clip-higher trick; unlike most R1 reproductions it applies RL to already-distilled long-CoT models, and its entropy-collapse study motivates entropy-aware on-policy recipes.

## 2026

### CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation
*2026 · method · `method_CUDA-Agent_2602.24286.txt` · arXiv [2602.24286](https://arxiv.org/abs/2602.24286)*

A four-stage warm-up-then-agentic-RL curriculum counteracts the domain mismatch (CUDA is <0.01% of pretraining) that made naive RL collapse at 17 steps. Stage 1 single-turn PPO warms the base model (32k context, one network serving as both policy and value). Stage 2 initializes the actor by collecting agent trajectories from the warmed model, rejection-filtering on R>0 and clean tool-call patterns, then SFT-ing. Stage 3 value-pretrains the critic via MSE on GAE targets (γ=1, λ=0.95) over sampled trajectories. Stage 4 runs agentic PPO at 128k context for 150 steps (200 at eval), global batch 1024, actor lr 3e-6, critic lr 6e-6.

- Single-turn PPO warm-up before any multi-turn agentic training.
- Actor init via rejection-filtered SFT on positive-reward trajectories.
- Critic init via separate value pretraining on GAE targets.
- Ordering yields stable 200-step PPO with consistent reward growth.

**Key results:** CUDA Agent (Seed1.6) reaches 98.8% pass on KernelBench, 2.11x geomean speedup vs torch.compile, beating Claude Opus 4.5 / Gemini 3 Pro by ~40% on Level-3.

*Evolution:* Extends agentic-RL-for-coding by combining DAPO-style asymmetric clipping, ReAct/OpenHands loops, and Agent Skills, explicitly reacting against training-free and leakage-prone fine-tuners.

### LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification
*2026 · method · `method_LaTER_2605.07315.txt` · arXiv [2605.07315](https://arxiv.org/abs/2605.07315)*

LaTER is a single fine-tuning stage rather than a multi-stage RL pipeline; its "two-stage" notion is an inference-time latent-then-explicit paradigm, not a training ordering. The supervised objective combines token-level cross-entropy (a CoT term λ_CoT=0.5 plus a non-CoT term), cached top-k teacher self-distillation KL (λ_KL=0.25, T=1.0), and a gated latent-halting loss (base weight 0.025, dynamically gated by the CE moving average). The corpus carries difficulty metadata so early training can emphasize easier then medium/hard examples, but no explicit warmup/anneal schedule across separate stages is described.

- Single-stage fine-tuning, not stage-wise SFT→preference→RL.
- CE + cached top-k teacher self-distillation KL + gated latent-halting loss.
- Difficulty-metadata-driven data emphasis but no formal curriculum schedule.
- Evaluation contrasts CoT, CoT-SFT, training-free LaTER, and trained LaTER.

**Key results:** Trained LaTER on Qwen3-14B hits 80.0% AIME 2025 (+10.0 over CoT baseline) using 33% fewer tokens, best across 7 benchmarks.

*Evolution:* Builds on the 2025-26 latent-reasoning line, splitting labor (latent exploration for search, explicit CoT for verification) to avoid degrading symbolic tasks.

### R³: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning
*2026 · method · `method_R3-ReplayReflectionRanking_2601.19620.txt` · arXiv [2601.19620](https://arxiv.org/abs/2601.19620)*

R³ is a single RL post-training stage applied atop already-distilled DeepSeek-R1-Distill-Qwen-1.5B/7B bases; there is no separate SFT or preference phase. Training follows a difficulty-stratified loop driven by a sample buffer: an initial epoch of standard GRPO runs first, after which three mechanisms activate by query hardness. Medium-difficulty groups with collapsed advantages trigger Cross-Context Replay; persistently failing hard queries (mean historical reward below threshold τ) trigger In-Context Self-Reflection; truncated/extremely-hard responses get the SERR signal. Unlike DeepScaleR's length-scaled schedule that progressively grows response length, R³ reuses historical trajectories across rounds rather than monotonically lengthening.

- Single GRPO stage on distilled base; no SFT/preference phase.
- Initial GRPO epoch then difficulty-gated replay/reflection/SERR.
- Reuses historical trajectories rather than length-scaled annealing.
- Difficulty-stratified activation keyed to advantage collapse and reward history.

**Key results:** R³-1.5B averages 60.59 over five math benchmarks (+12.78 over base), AIME24 47.50 at 7,574 tokens vs base 28.1 at 12,270; R³-7B 67.18 average.

*Evolution:* Builds on GRPO/DAPO and DeepScaleR small-model reasoning RL plus experience-replay work, reacting to GRPO's intra-group advantage collapse on hard tasks.

### SOAR: Teaching Models to Teach Themselves — Reasoning at the Edge of Learnability
*2026 · method · `method_SOAR_2601.18778.txt` · arXiv [2601.18778](https://arxiv.org/abs/2601.18778)*

SOAR self-generates a "stepping-stone" curriculum: a teacher proposes easier synthetic problems that warm-start the student on otherwise-unlearnable hard problems. Two mixing strategies are compared—"curriculum training" (64 synthetic warmup steps then switch to real fail@128 questions) and "mixed training" (synthetic+real throughout). MATH uses curriculum (better for the Base-T baseline); HARP and OlympiadBench use mixed (more stable, avoiding spike-then-crash). Across the meta-RL trajectory, a promotion mechanism accumulates useful questions (Dbest) and advances the student baseline whenever the 3-step moving-average teacher reward exceeds τ=0.01, producing an easy-to-hard progression in question distribution.

- Teacher-generated stepping-stone problems warm-start unlearnable hard tasks.
- Curriculum (64 warmup steps) vs mixed training, chosen per-benchmark.
- Promotion mechanism advances baseline when 3-step avg teacher reward > 0.01.
- Easy-to-hard shift in question distribution over the meta-RL trajectory.

**Key results:** SOAR-PQ on Llama-3.2-3B-Instruct reaches 18.9% pass@32 on MATH (~2x / +9.3 over Hard-Only) and ~4x pass@1; recovers 75% of full-curated-MATH upper bound.

*Evolution:* Builds on asymmetric self-play (AlphaZero, Alice-Bob) and data-free LLM self-play, reacting against intrinsic-reward diversity collapse by grounding the teacher in measured student progress.

### SWE-Fuse: Empowering Software Agents via Issue-free Trajectory Learning and Entropy-aware RLVR Training
*2026 · method · `method_SWE-Fuse_2603.07927.txt` · arXiv [2603.07927](https://arxiv.org/abs/2603.07927)*

A two-stage cold-start-then-RL ordering. Stage 1 is issue-free-driven SFT on D_mixed = D_issue ∪ D_issue-free, autoregressively predicting full reasoning-action-observation trajectories to bias the initial policy toward productive action spaces. Stage 2 is entropy-aware RLVR on the SFT-initialized model. Ablations prove SFT cold-start is essential: Qwen3-32B without it needs >200 RL steps to converge, while SFT-initialized 8B/32B models gain rapidly and reach a higher ceiling. Data-mixing ablations (Qwen3-8B, 4k fixed size) find optimal performance at 25–50% issue-free ratio (~34.0% resolve vs 30.5% at 75–100%), with monotonic scaling 13.5% (0 samples) → 39.0% (full 14k) and diminishing returns past 4–8k.

- Two-stage: issue-free SFT cold-start, then entropy-aware RLVR.
- SFT cold-start cuts convergence from >200 RL steps and raises the ceiling.
- Optimal cross-stage data mixing at 25–50% issue-free ratio.
- Trajectory data scales 13.5%→39.0% resolve with diminishing returns past 4–8k.

**Key results:** SWE-Fuse-Qwen3-32B resolves 60.2% of SWE-bench Verified (65.2% at TTS@8), SOTA among open ≤32B models and +1.8% over OpenAI-o3.

*Evolution:* Extends RLVR-for-coding by attacking real-world data noise via issue-free trajectory learning and grafting entropy-adaptive clipping onto RLOO.

### SandMLE: Synthetic Sandbox for Training Machine Learning Engineering Agents
*2026 · method · `method_SandMLE_2604.04872.txt` · arXiv [2604.04872](https://arxiv.org/abs/2604.04872)*

A single trajectory-wise RL stage (GRPO) from the Base Qwen3 checkpoint; a Seed-SFT variant initializes GRPO from an SFT checkpoint built on 60 Claude-4.5-Sonnet expert trajectories. There is no multi-stage pretrain→SFT→preference ordering—SFT and RL are treated as complementary axes (format/pipeline compliance vs. higher-order reasoning). GRPO runs 100 steps, group size n=4, temperature 1.0, lr 1e-6, batch 16, clipping 0.28, KL disabled, per-step execution cap τ_max=90s, max trajectory length T_max=20 turns. Reward weights are fixed (w_format=0.1, w_execute=0.3, w_median=0.1, w_bronze=0.2, w_silver=0.2, w_gold=0.1). Test-time scaling varies T_max (peaks at 30 turns) under a 24 GPU-hour budget.

- Single GRPO stage from Base, or Seed-SFT-initialized GRPO variant.
- No multi-stage ordering; SFT and RL are complementary axes.
- Fixed reward weights across format/execution/medal tiers.
- Test-time scaling varies max trajectory length under a fixed GPU-hour budget.

**Key results:** SandMLE cuts per-rollout execution 13.7x; Qwen3-8B/14B/30B hit 22.7%/22.7%/27.3% Any Medal on MLE-Bench-Lite (+24.7–100.7% over Base).

*Evolution:* Builds on trajectory-wise RL in SWE/web search and GRPO-style RLVR, addressing MLE execution latency that had forced prior work back to SFT or off-policy async RL.

### DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
*2026 · report · `report_DeepSeek-V4_2606.19348.txt` · arXiv [2606.19348](https://arxiv.org/abs/2606.19348)*

Pre-training follows a length-and-density schedule: sequence length grows 4K→16K→64K→1M, batch ramps then holds (75.5M/94.4M tokens for Flash/Pro), LR linearly warms then cosine-decays; dense attention warms the first 1T tokens before a two-stage sparse-attention introduction (lightning-indexer warmup then full sparse) at 64K. Post-training is two-stage: (1) independent cultivation of domain specialists (math, coding, agent, instruction), each via SFT then GRPO RL, with separate specialists for three reasoning-effort modes (Non-think, Think High, Think Max) using distinct length penalties and context windows; (2) unified consolidation via multi-teacher on-policy distillation. The key change from V3.2 is that mixed-RL merging is entirely replaced by on-policy distillation.

- Length/density pretraining schedule 4K→1M with batch ramp and cosine LR.
- Two-stage sparse-attention introduction after 1T dense-attention warmup tokens.
- Per-specialist SFT→GRPO, with three reasoning-effort specialist families.
- On-policy distillation replaces V3.2's mixed-RL merging for consolidation.

**Key results:** DeepSeek-V4-Pro-Max: SimpleQA-Verified 57.9, Codeforces 3206 (23rd human), SWE-Verified 80.6%, Putnam-2025 120/120; 1M context at 27% FLOPs / 10% KV cache of V3.2.

*Evolution:* Extends V3/V3.2's MoE+MTP and R1's GRPO, replacing mixed-RL merging with full-vocabulary on-policy distillation from 10+ specialist teachers as the consolidation default.

### GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents
*2026 · report · `report_GLM-5V-Turbo_2604.26752.txt` · arXiv [2604.26752](https://arxiv.org/abs/2604.26752)*

Training is ordered as pre-training (mixed text + multimodal) → SFT (vision-language deeply integrated) → joint RL over 30+ task categories spanning perception, reasoning, and agentic capability, RL adding consistent cross-domain gains. CogViT itself follows a two-stage recipe: distillation-based masked image modeling, then contrastive image-text alignment. The central rationale is hierarchical optimization: capability is distributed across a multi-level hierarchy (element perception, GUI grounding, single-step action prediction, trajectory-level action prediction) used in both SFT and RL, rather than monolithic long-horizon training; lower-level tasks are cheaper to construct/verify and stabilize training. Cold-start trajectories seed RL, after which multi-task collaborative RL (including on-policy distillation) shares strategy patterns across domains.

- Pretrain → SFT → joint RL over 30+ task categories.
- CogViT two-stage: masked-image-modeling distillation then contrastive alignment.
- Hierarchical multi-level task hierarchy shared across SFT and RL stages.
- Cold-start trajectories seed collaborative multi-task RL with on-policy distillation.

**Key results:** GLM-5V-Turbo scores 94.8 Design2Code (beating Claude Opus 4.6), AndroidWorld 75.7, OSWorld 62.3, MMSearchPlus 30.0 (~8x over GLM-4.6V).

*Evolution:* Builds on GLM-4.5V/4.1V-Thinking multi-task RL, pushing toward native multimodal agents where perception—not just reasoning—is foundational.

### GLM-5: from Vibe Coding to Agentic Engineering
*2026 · report · `report_GLM-5_2602.15763.txt` · arXiv [2602.15763](https://arxiv.org/abs/2602.15763)*

Base training is two stages: pre-training (general language + coding) then mid-training (agentic + long-context), extending context progressively 32K (1T tokens) → 128K (500B) → 200K (50B). DSA is added via continued pre-training: a 1000-step warmup (lr 5e-3, base frozen) then 20B-token sparse adaptation. Post-training is sequential: SFT (interleaved/preserved/turn-level thinking modes; max context 202,752; INT4 QAT) → Reasoning RL (math/science/code/TIR) → Agentic RL (coding + search agents, fully asynchronous) → General RL (correctness, emotional intelligence, task quality with hybrid rule/ORM/GRM rewards and human anchors) → On-Policy Cross-Stage Distillation as the final stage, using earlier SFT/RL checkpoints as teachers (group size 1, batch 1024) to prevent catastrophic forgetting.

- Pretrain → mid-train with progressive context 32K→128K→200K.
- DSA via continued pretraining: 1000-step warmup then 20B-token sparse adaptation.
- Sequential SFT → Reasoning RL → Agentic RL → General RL → cross-stage distillation.
- Final on-policy distillation reuses earlier checkpoints as teachers against forgetting.

**Key results:** GLM-5 scores 50 on Artificial Analysis Intelligence Index v4.0 (first open-weights model), SWE-bench Verified 77.8, BrowseComp 75.9, ~20% avg gain over GLM-4.7.

*Evolution:* Builds on GLM-4.5's MoE and decoupled rollouts, adopting DeepSeek's DSA and GRPO+IcePop, pushing open-weights from vibe coding toward long-horizon agentic engineering.

### Kimi K3: Open Frontier Intelligence (Technical Report of Kimi K3)
*2026 · report · `report_Kimi-K3_2607.24653.txt` · arXiv [2607.24653](https://arxiv.org/abs/2607.24653)*

Pre-training is natively multimodal from the start (no post-hoc alignment), beginning at 8k and extending to 64k. Long-context extension follows a progressive four-stage curriculum: 8K→64K during pre-training, then 256K→1M during cooldown, concentrating costly long-sequence compute in a small budget fraction. Training uses cosine LR decay (chosen over WSD after scaling-law searches) with 1% linear warmup, weight decay 0.1, and Per-Head Muon. Post-training is three-stage: (1) SFT for cold-start; (2) RL across three domains × three reasoning-effort levels {low,high,max} yielding nine expert models; (3) multi-teacher on-policy distillation into one unified model. Reasoning-effort RL uses a stage-wise curriculum over a per-problem budget multiplier τ: train a max-budget variant with large τ first (capping overthinking), then anneal τ down for high- and low-effort experts. MXFP4/MXFP8 QAT is applied from SFT onward.

- Natively multimodal pretraining with 4-stage 8K→64K→256K→1M context curriculum.
- Cosine LR decay with 1% linear warmup, Per-Head Muon.
- SFT → 9 RL experts (3 domains × 3 effort levels) → multi-teacher distillation.
- Reasoning-effort RL anneals per-problem budget τ from large (max) down to low.

**Key results:** Kimi K3 (2.8T MoE, 104B active): BrowseComp 91.2, ProgramBench 77.8, GPQA Diamond 93.5, #1 WebDev Arena (1678 Elo), #4/580 on AAII v4.1.

*Evolution:* Builds on Kimi K2/K2.5 and DeepSeek-R1/Kimi K1.5, pushing both 3T-class pretraining and million-token agentic RL/test-time scaling simultaneously.

### Qwen3-Coder-Next Technical Report
*2026 · report · `report_Qwen3-Coder-Next_2603.00729.txt` · arXiv [2603.00729](https://arxiv.org/abs/2603.00729)*

A staged pipeline from the pretrained Qwen3-Next base. (1) Mid-training / continued pretraining shifts representations toward code and agent-centric domains, balancing mostly-natural with minimal synthetic data plus a small amount of instruction-following data for early downstream monitoring; trained on trillions of tokens at 262K context with next-token + FIM objectives and best-fit packing. (2) SFT bridges base knowledge and instructions. (3) From the SFT checkpoint, multiple expert models are trained with distinct data/recipes: Web Development, User Experience (CLI/IDE tool-format), Single-turn QA (execution-verifiable RL), and Software Engineering (multi-turn agentic RL). (4) Expert distillation consolidates all experts into one deployable model. SFT and RL prompts are kept fully disjoint to prevent stage-to-stage information leakage.

- Mid-training CPT at 262K with next-token + FIM and best-fit packing.
- SFT alignment then four specialist branches (web/UX/QA/SWE) from one checkpoint.
- Expert distillation consolidates specialists into a single deployable model.
- SFT and RL prompts fully disjoint to block stage-to-stage leakage.

**Key results:** Qwen3-Coder-Next (80B/3B active MoE) scores 70.6/71.1/71.3% on SWE-Bench Verified across SWE-Agent/MiniSWE-Agent/OpenHands, 92.7% tool-template following.

*Evolution:* Builds on Qwen2.5-Coder and SWE-Smith/Flow/Rebench executable-data scaling plus DeepSeek-R1-style execution RL, pushing agentic training onto a small-active MoE.

### Qwen3.5-Omni Technical Report
*2026 · report · `report_Qwen3.5-Omni_2604.15804.txt` · arXiv [2604.15804](https://arxiv.org/abs/2604.15804)*

Pretraining is split into three stages: S1 Encoder Alignment (LLM frozen, vision/audio encoders and adapters trained separately), S2 General Stage (all parameters unfrozen, 32k sequence length, ~4T mixed-modal tokens), and S3 Long Context Stage (sequence length extended to 262k with a higher proportion of long audio/video). Thinker post-training is three stages: Specialist Distillation (domain teachers built via independent SFT+RL then distilled into one unified model), On-Policy Distillation (using the higher-quality text-conditioned response as target for the matching audio-conditioned query to close the text/audio gap), and Interaction-Aligned RL (multi-turn trajectories with UX-oriented rewards fixing code-switching, persona drift, long-context instruction-following). Talker uses four stages: general pretraining on >20M hours of multilingual speech, Long-Context CPT at 64k, an RL stage combining DPO on human preference pairs with GSPO plus rule-based rewards, and a lightweight Speaker Fine-tuning stage.

- Pretraining S1 encoder alignment → S2 general 32k/4T tokens → S3 long-context 262k.
- Thinker: specialist distillation → on-policy distillation → interaction-aligned RL.
- OPD uses text-conditioned response as target for matching audio-conditioned query.
- Talker: pretraining → long-context CPT → DPO+GSPO RL → speaker fine-tuning.

**Key results:** Qwen3.5-Omni-Plus is SOTA across 215 audio/audio-visual benchmarks, beating Gemini-3.1 Pro on audio (FLEURS WER 6.6% vs 7.3%) with SEED-TTS WER 1.26.

*Evolution:* Builds on Qwen2.5-Omni/Qwen3-Omni Thinker-Talker design, scaling native omnimodal training to hundreds of billions of parameters toward omni-agents that act and code.

### The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence
*2026 · report · `report_MiniMax-M2-Series_2605.26494.txt` · arXiv [2605.26494](https://arxiv.org/abs/2605.26494)*

Pretraining extends context 8K→32K→192K and mixes high-quality and long-context data (code concatenation, long PDFs, document packing) into the decay phase. During decay, MTP grows from 1 to 3 modules by weight copying: the new heads first train with the main model frozen, then everything trains jointly. Post-training starts with an interleaved-thinking SFT cold start (chat, reasoning, code and cowork data built by rejection sampling), then runs multi-stage mixed-domain RL. Every RL stage draws reasoning, coding, agent and general data together. Across stages the mix shifts from reasoning/general toward agent/coding, context length grows per domain, and difficulty moves toward harder tasks. The stated aim is to avoid both forgetting and negative transfer between domains. The series iterates M2→M2.5→M2.7.

- Pretrain context 8K→32K→192K; high-quality + long-context data mixed into decay.
- MTP grown 1→3 modules during decay: frozen-trunk head training, then joint.
- Interleaved-thinking SFT cold start → multi-stage mixed-domain RL.
- Per-stage mix shifts toward agent/coding with longer context and harder tasks.

**Key results:** M2.7 (~10B active) leads Multi-SWE-bench (52.7 vs 50.3 for Opus 4.6), scores 56.2 on SWE-bench Pro and a 66.6% MLE Bench Lite medal rate; M2→M2.7 lifts BrowseComp by 33.8 points.

*Evolution:* Follows MiniMax-01/M1, keeping CISPO but dropping hybrid Lightning Attention for full attention, and moves post-training from reasoning RL toward agent-native RL over black-box scaffolds.

### Solar Open 2 Technical Report
*2026 · report · `report_Solar-Open-2_2607.20062.txt` · arXiv [2607.20062](https://arxiv.org/abs/2607.20062)*

Pre-training has four stages: (1) selective weight transfer from Solar Open 1; (2) General Pre-training on 10T tokens; (3) Intensive Pre-training on the best 1T tokens, re-filtered at a higher quality threshold; (4) Length Expansion (~865B tokens) to 1M context with long documents, repo-level code, and injected reasoning/agent data. Stage 4 causes a performance drop, so four intermediate checkpoints are merged (the merge scores 0.745 on a 50+-metric PLM average). Post-training also has four stages: SFT covering reasoning and agent tasks, multi-domain RL with verifiable rewards on STEM-centered tasks, a Specialist stage that trains twelve domain teachers (SFT + domain RL) in parallel, and Multi-teacher On-Policy Distillation (MOPD) that consolidates them into one model. Office tasks use an expertise curriculum that progressively strips scaffolding from prompts.

- Pretrain: weight transfer → 10T general → 1T intensive (stricter filter) → 1M length expansion.
- Four-checkpoint merge offsets the Stage-4 performance drop.
- Post-train: SFT → STEM RLVR → 12 parallel specialists → MOPD consolidation.
- Office-task curriculum progressively removes prompt scaffolding.

**Key results:** Solar Open 2 leads comparably sized open-weight models on MMLU-Pro (86.2) and LiveCodeBench v6 (92.4), tops the Korean average (85.4), and scores 86.8 on Ko-GDPval (Solar Open 1: 3.4); trails on SWE-Bench Verified (70.4 vs 73.0-73.8).

*Evolution:* Scales Solar Open 1 into a Korean sovereign agentic model, adopting the specialist-then-on-policy-distillation consolidation of DeepSeek-V4/Nemotron/MiMo with exact full-vocabulary KL.

### From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement
*2026 · rl · `rl_RLSVR_2607.23802.txt` · arXiv [2607.23802](https://arxiv.org/abs/2607.23802)*

Not a staged pipeline but an RL self-play loop from Qwen3 models (Qwen3-4B-Instruct-2507 as the actor) run for 100 iterations. The key schedule alternates a Performing stage and a Detection stage, updating only one at a time. A hysteresis gate controls the switch: Detection→Performing when EMA detection accuracy is ≥0.9 and the N/A rate ≤0.1; back to Detection when the error rate is ≥0.4 or the N/A rate ≥0.5; each phase lasts at least 5 updates. Updating both roles jointly instead degrades math (5-benchmark average 42.4→35.3), while alternation raises it to 50.8.

- Single self-play RL loop, 100 iterations, no SFT/preference stages.
- Performing and Detection stages alternate; only one updates at a time.
- Hysteresis gate on EMA accuracy / N/A / error rates; minimum 5 updates per phase.
- Joint updating hurts math (42.4→35.3); alternation reaches 50.8.

**Key results:** SpyRL beats R-Zero and Absolute Zero on every summarization and creative-writing benchmark, lifts Qwen3-4B math average 41.4→50.4, and beats a Qwen3.5-27B rubric-as-reward baseline (59.3% win on WritingPrompts) with no external verifier.

*Evolution:* Extends proposer-solver self-play (Absolute Zero, R-Zero) to information-asymmetric multi-agent games, arguing verifiability can be engineered for open-ended generation rather than taken as a fixed task property.

### Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents
*2026 · report · `report_Qwen-UI-Agent_2607.28227.txt` · arXiv [2607.28227](https://arxiv.org/abs/2607.28227)*

Training has three stages, SFT → Action RL → Online RL, wrapped in a data flywheel. Bootstrapping uses multiple rounds of rejection sampling (following MAI-UI) to build the initial SFT corpus; each later iteration evaluates the model, diagnoses failures, synthesizes targeted tasks and retrains. In SFT, one expert per domain (mobile, desktop, web, DeepSearch) trains mainly on its own domain plus a controlled cross-domain mix, and the experts are merged into one checkpoint. A small share of verified-correct, self-sampled general/math/code/tool data preserves general ability; questions the starting model already solves work better than ones it fails. Online RL uses a model-adaptive curriculum: an active pool holds intermediate-success tasks, unsolved tasks sit in a low-budget monitoring pool until they start succeeding, and mastered tasks are monitored and reactivated if performance drops.

- SFT → Action RL → Online RL inside an evaluate-diagnose-synthesize flywheel.
- Per-domain SFT experts with controlled cross-domain mix, then merged.
- Self-sampled, already-solved general data preserves general ability.
- Online RL pools: active (intermediate success), monitoring (unsolved), mastered (reactivated on regression).

**Key results:** The 27B model sets mobile SOTA at 82.1% MobileWorld (vs Seed 2.1 Pro 73.2) and 92.2% MobileWorld-Real, reaching 73.6% WebArena and 79.5% OSWorld-Verified; Action RL lifts task SR >7% while cutting reasoning tokens 21.3%.

*Evolution:* Continues MAI-UI and the 2026 native-GUI-agent wave (UI-TARS-2, GUI-Owl-1.5, EvoCUA), with an agent-driven flywheel pointing toward agents that partly build their own training curricula.

### Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering
*2026 · rl · `rl_Frontis-MA1_2607.28568.txt` · arXiv [2607.28568](https://arxiv.org/abs/2607.28568)*

The pipeline is base model → execution-grounded full-parameter SFT warm start → online RL from sandbox execution feedback → deployment inside the OpenMLE-Evo search harness. SFT collection is budget-adaptive: each task stops sampling at an accepted-example quota or when its execution budget runs out, shifting verification compute toward sparse-success tasks. SFT runs 3 epochs at LR 3e-5 with cosine decay and 0.1 warmup, global batch 128, 32K context (SLIME + Megatron-LM). RL samples operators with fixed probabilities (Draft 0.50, Improve 0.17, Debug 0.17, Crossover 0.16) and selects parent states by F(p) = norm(reward) + norm(child-reward variance) + norm(visit-cooling), so training follows informative states rather than a fixed curriculum.

- SFT warm start → online sandbox RL → deployment in the evolutionary harness.
- Budget-adaptive SFT sampling shifts compute toward sparse-success tasks.
- SFT: 3 epochs, LR 3e-5 cosine, 0.1 warmup, batch 128, 32K context.
- RL parent selection by reward, child-reward variance and visit-cooling replaces a fixed curriculum.

**Key results:** Under the same OpenMLE-Evo harness, post-training lifts Medal Average 39.39%→60.61% (Qwen3.6-35B-A3B); with OpenMLE-Evo-Max it reaches 71.21%, above GPT-5.5+Codex (68.18%).

*Evolution:* Unites evolutionary MLE search (AIDE/AIRA), executable MLE environments and MLE-agent RL by training the same operators the inference-time evolution composes, as an open testbed toward AI4AI self-improvement.

### Qwen-CUA: Native Computer Use for (almost) Everything
*2026 · report · `report_Qwen-CUA_2608.02352.txt` · arXiv [2608.02352](https://arxiv.org/abs/2608.02352)*

Training is iterative over several runs. In each run, failed SFT queries and weak domains drive teacher rerolls, new human trajectories and targeted data, which together form a refreshed SFT mixture. Every SFT run restarts from the same mid-training checkpoint rather than the previous agent, so gains transfer through data rather than through optimization drift. The new SFT model then calibrates the RL pool: from eight trial rollouts per query, tasks with 0 < successes < 8 are kept, and SAPO RL on that pool yields the next-iteration model. In the final RL run, held-out validation rose 0.734→0.770 and checkpoint 40 (update 800 of 1,000) was selected.

- Iterative SFT-refresh → RL-calibration loop across generations.
- Each SFT run restarts from the same mid-training checkpoint (data carries gains, not drift).
- RL pool keeps tasks with 0 < successes < 8 of eight trials under the new SFT model.
- OSWorld-Verified across generations: 38 → 46 → 62 → 73 → 86.2.

**Key results:** Qwen-CUA reaches 86.2 OSWorld-Verified, above GPT-5.5 (78.7) and Claude Opus 4.8 (83.4), and cuts RedTeamCUA attack success 36.6→16.4; Qwen-CUA-Max reaches 87.6.

*Evolution:* Continues the native screenshot-only agent line (UI-TARS, OpenCUA, Qwen3-VL) on CUA-Gym and SAPO, distinguished by an SFT-refresh/RL-calibration loop that always restarts SFT from mid-training.

### Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA
*2026 · method · `method_Macaron-MoL_2608.09819.txt` · arXiv [2608.09819](https://arxiv.org/abs/2608.09819)*

Every model starts from a frozen pretrained base (GLM-5.2 744B for Venti, Qwen3.6-35B-A3B for Tall); only LoRA specialists are trained. The recursive self-improvement loop has three stages per generation: Discovery builds a harder task bank; Expansion runs tasks under a fixed model-plus-harness configuration and searches prompt/skill/tool/hook edits in "language space"; Update applies GRPO to the adapters from selected trajectories and registers the accepted harness configuration as the next generation's. Base, specialists and harness are released on separate schedules. The four specialists share one recipe — AdamW, LR 5e-6, batch 4, 4 epochs, linear-warmup cosine with 0.1 warmup — except Agent (1 epoch) and GenUI (batch 2, 1 epoch). Only one snapshot is released; gains across generations are not measured.

- Frozen base; only LoRA specialists train.
- Per-generation loop: Discovery → Expansion (harness search) → Update (GRPO on adapters).
- Base, adapters and harness versioned on separate schedules.
- Shared specialist recipe: LR 5e-6, 4 epochs, 0.1 warmup cosine (Agent/GenUI 1 epoch).

**Key results:** Macaron-V1-Venti leads compared frontier models on ChatBench (58.3), UI4A-Bench (87.8 vs 75.9 for Opus 4.8) and TerminalBench 2.1 (87.6); Tall beats its base on all seven matched rows; MoL stores ~74% fewer parameters than four merged-base copies.

*Evolution:* Shifts post-training from monolithic checkpoints toward an agent system with separately versioned base, adapters and harness, coupling harness search with adapter RL; compounding gains across generations are not yet shown.

### Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning
*2026 · report · `report_Palmyra-x6_2608.16620.txt` · arXiv [2608.16620](https://arxiv.org/abs/2608.16620)*

A single offline post-training stage on an existing base, GLM-5.2 (744B total / ~40B active MoE), with no pretraining, no preference stage and no on-policy RL. Development used a matrix of 12 SFT/ASFT runs varying the objective, K (0.1 or 0.02), LR (5e-7 or 1e-6), batch (16 or 32), context (64K with CP2, or 32K) and data mix (12-dataset reasoning-trace, 10-dataset, a panel-pruned 240-trajectory subset, 8/9-dataset short-context mixes, and a broad mix at doubled batch), plus a zero-LR sanity run. The final run changes exactly two things — a newer base and the Muon optimizer — and trains one epoch with a 0.1 warmup fraction from LR 1e-8 and cosine decay 5e-7→5e-8.

- Single anchored-SFT stage on GLM-5.2; no preference or RL stage.
- 12-run ablation matrix over objective, K, LR, batch, context and data mix.
- Zero-LR sanity run included as a control.
- Final: 1 epoch, Muon, 0.1 warmup from 1e-8, cosine 5e-7→5e-8.

**Key results:** Palmyra x6 beats the prior Writer Agent default on all six benchmarks (+0.320 MCP-Atlas, +0.305 FinanceBench), tops BFCL Core (0.785) and has the highest six-benchmark mean (0.765) against five frontier models.

*Evolution:* Post-trains an existing open frontier MoE rather than training end to end, combining Anchored SFT with self-distillation and data minimalism to avoid eroding base capabilities and skipping RL entirely.

### Apodex 1.1: Scaling Agentic Intelligence for Complex Work
*2026 · report · `report_Apodex-1.1_2608.23283.txt` · arXiv [2608.23283](https://arxiv.org/abs/2608.23283)*

Post-training is unified SFT followed by agentic RL inside a controlled capability-development loop. SFT variants are trained per major domain (general, agentic, coding) and model-soup merged into a single behavioral initialization for RL. RL then runs over environment and coordination trajectories, with batches mixing localized pivot continuations and unhinted full tasks. After each round, failed trajectories, benchmark errors and runtime failures are abstracted into capability gaps; a Task Pipeline turns these into new tasks and worlds that re-enter family-specific verification, and expert and user feedback decide the next allocation of training effort. The authors call this "self-evolution" but restrict it to this managed engineering loop. Pre-training and base-model details are not described.

- Per-domain SFT variants (general/agentic/coding) model-soup merged as RL init.
- Agentic RL over environment and coordination trajectories.
- RL batches mix pivot continuations with unhinted full tasks.
- Round-by-round gap → new tasks/worlds → re-verification loop steers training effort.

**Key results:** With Agent Team, Apodex 1.1 leads its tables on FrontierFinance (54.3 vs 49.2) and FrontierScience-Research (63.3), improves APEX-Agents 16.5→38.5 and reaches GDPVal 78.8; trails on SWE-bench Verified (77.7 vs 92.2).

*Evolution:* Extends Apodex 1.0's Agent Team toward long-horizon working agents, treating environment construction and multi-agent coordination as explicit scaling axes.

### A.X K2 Technical Report
*2026 · report · `report_A.X-K2_2608.30181.txt` · arXiv [2608.30181](https://arxiv.org/abs/2608.30181)*

Pre-training follows a WSD schedule. Stage 1 trains 6.4T tokens at 4K context, ramping to a 2.2e-4 peak LR and 64M-token batch; Stage 2 trains 1.4T tokens of reasoning/STEM/synthetic data while the LR cosine-decays to 7e-5; Stage 3A extends to 32K (~0.22T, RoPE base 1e4→1e6), Stage 3B to 128K, and Stage 3C performs SGA sparse-attention adaptation on the same mix (3B+3C ~0.14T). Instruction-format data enters during pre-training, and the last six checkpoints are merged (WSM-style). Post-training runs an indexer SFT (0.4M samples, indexer KL 0.1), then main SFT on packed 128K sequences (LR 1.5e-5, batch 128) with the indexer frozen. Multi-stage RL follows: every stage jointly trains instruction following, preference and agentic tool use, safety joins in later stages, and the mix is re-weighted at each boundary around the weakest behaviors.

- WSD pretrain: 6.4T @4K → 1.4T reasoning/STEM decay → 32K → 128K → sparse-attention adaptation.
- Instruction data in pretraining; last six checkpoints merged.
- Indexer SFT → main 128K SFT with frozen indexer.
- Multi-stage joint RL; safety added late; mix re-weighted toward weakest behaviors.

**Key results:** A.X K2 leads compared open-weight models on AIME26 (97.1), Apex (45.8, from 1.0), KMMLU-Pro (80.5) and tau2-Telecom (98.0), reaching IMO 2025 gold threshold on ~8.5T tokens; trails on Terminal Bench (36.0) and BrowseComp (9.3).

*Evolution:* SK Telecom's successor to A.X K1, adding DeepSeek-V3.2/GLM-5-style indexer sparse attention with a simplified sparse-only warmup and replacing dual-track training with paired-response Think-Fusion and CISPO.

### NeoHorse-1: Towards Recursive Self-Improvement via Agentic Post-Training with Routing Harness
*2026 · method · `method_NeoHorse-1_2609.08183.txt` · arXiv [2609.08183](https://arxiv.org/abs/2609.08183)*

SFT runs as a three-stage routing-guided curriculum: each stage holds roughly a third of the examples, ordered by a routing score (hard tier index k_i, or soft score s_i = Σ_k k·π_i,k over C0-C3), progressively adding higher-demand examples while reserving some low-score examples for later stages. Every example is used once per pass, with no optimizer or LR-schedule reset between stages. Routing-guided on-policy distillation follows with the same three-stage context allocation. Outside training, a capability-guided allocation step evaluates each checkpoint on a decontaminated stratified suite, builds a model-deficiency profile, and shifts the next mixture toward weak regions; the paper runs only one pass of this evaluation-selection-update loop.

- Three-stage SFT curriculum ordered by router capability-demand score.
- Low-score examples partly reserved for later stages; no optimizer/LR reset.
- Routing-guided on-policy distillation reuses the three-stage allocation.
- Deficiency-profile mixture update, run for a single loop iteration.

**Key results:** Agentic post-training raises the 10-benchmark macro average 58.94→64.87 (4B) and 65.60→69.04 (9B); routing-harness data beats Toucan by +6.26 avg under a matched recipe.

*Evolution:* Extends trajectory-SFT lines (FireAct, AgentTuning, Agent-FLAN) with on-policy distillation, using router capability-demand predictions as a cheap difficulty signal for curriculum ordering; compounding across loops is left open.

### An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics
*2026 · rl · `rl_Nemotron-IMO_2609.10712.txt` · arXiv [2609.10712](https://arxiv.org/abs/2609.10712)*

Two independent specialists are branched from the Nemotron-3-Ultra 550B-A55B general-availability checkpoint rather than chained. One is long-context SFT: 426K max sequence, ~1.6K optimizer steps at global batch 64, peak LR 1.5e-5 with warmup over 1,024 samples and cosine decay to 2e-6, with step 1300 selected on 133 proof problems. The other is proof-generation RL started directly from GA. At test time a staged inference curriculum follows: round-1 ensemble generation, up to eight generate-verify-refine rounds, then a separate high-compute final-selection stage.

- Parallel SFT and RL specialists branched from GA, not chained.
- SFT: 426K context, ~1.6K steps, LR 1.5e-5 cosine to 2e-6.
- RL proof generation starts directly from the GA checkpoint.
- Inference: ensemble round 1 → up to 8 generate-verify-refine rounds → final selection.

**Key results:** The natural-language system scored 30/42 at IMO 2026 (gold cutoff 29) within ~707M tokens and 1,464 GPU-hours; splitting round-1 budget across checkpoints beats doubling one (RL128+SFT64 159 vs RL256 139).

*Evolution:* Follows the IMO 2025 natural-language gold results and DeepSeekMath-V2's generate-verify-refine line, contributing a fully open recipe, data and compute accounting.

### DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression
*2026 · report · `report_DeepSeek-V4.1-Flash_2609.19969.txt` · arXiv [2609.19969](https://arxiv.org/abs/2609.19969)*

Pretraining runs from scratch with sparse attention at 64K, extends to 1M at 34T tokens, and cosine-decays the LR 2.6e-4→2.6e-5 over 28T-40T tokens. DeepSeek-ViT is trained beforehand with SigLIP on 47B pairs, then autoregressively on 236B tokens, and a DSpark drafter is trained on the frozen backbone after pretraining. Post-training follows the "standard" DeepSeek-V4 paradigm: SFT, then large-scale asynchronous RL, then a final full-vocabulary on-policy distillation stage over all domains from more than 40 heterogeneous teachers. RL is scaled across compute and scaffolds (Claude Code versions, OpenCode, Pi, DeepSeek Harness), and successive RL runs are re-initialized by merging checkpoints from different scaffolds or configurations. The Hierarchical Sparse Indexer, FP4 KV QAT and bounded-replay simulation are introduced during post-training.

- Sparse-attention pretraining from scratch at 64K, extended to 1M at 34T tokens.
- ViT pretrained separately; DSpark drafter trained on the frozen backbone.
- SFT → async RL → full-vocabulary OPD from 40+ teachers.
- Successive RL runs re-initialized from cross-scaffold checkpoint merges.

**Key results:** With 16B active parameters, V4.1-Flash lifts DeepSWE 54.4→74.2 over V4-Flash and matches or beats Opus-5 on DeepSWE, Terminal-Bench 2.1 (89.1) and AutomationBench, with ~1/4 the global KV cache.

*Evolution:* Refines DeepSeek-V4's sparse attention and OPD recipe, claiming post-training algorithms barely matter at this stage and gains come from scaling verifiable tasks, environments and scaffold diversity.

### Qwen3.8-Omni: Towards Native Omni-Modal Agents
*2026 · report · `report_Qwen3.8-Omni_2609.25611.txt` · arXiv [2609.25611](https://arxiv.org/abs/2609.25611)*

Thinker pretraining runs four stages, all at a 262,144-token native length. S1 freezes the LLM (initialized from Qwen3.8-Next) and aligns the vision, AuT and Spatial AuT encoders, training the adapters first; S2 unfreezes everything on the 2.5T mix; S3 freezes the backbone and warms up the Qwen Sparse Attention (QSA) indexer with dense-attention supervision; S4 enables QSA and jointly trains backbone and indexer. Post-training has two stages: multi-teacher distillation of specialists into one student, then unified RL. Context is extended 256K→1M after post-training. The Talker has its own five stages: pretraining, CPT, per-language experts merged via MOPD, speaker fine-tuning, then RL.

- Thinker pretrain: encoder alignment → full unfreeze (2.5T) → QSA indexer warmup → joint sparse training.
- Post-train: multi-teacher specialist distillation → unified RL.
- Context extended 256K→1M after post-training.
- Talker: pretrain → CPT → MOPD-merged language experts → speaker FT → RL.

**Key results:** Against Qwen3.5-Omni-Plus it raises the 29-eval audio/audio-visual/agent average by >25% while matching text-only Qwen3.8-Flash (SWE-bench Pro 63.3 vs 62.5); agentic evidence gathering lifts OmniVideoBench 63.4→67.8 with ~45.7% fewer tokens.

*Evolution:* Succeeds Qwen3.5-Omni, moving the omni line toward long-horizon agentic productivity and following the 2026 pattern of consolidating specialist teachers via multi-teacher distillation before unified RL.

### Hunyuan-A13B Technical Report
*2026 · report · `report_Hunyuan-A13B_2609.27284.txt` · arXiv [2609.27284](https://arxiv.org/abs/2609.27284)*

Pretraining is a 20T-token foundation stage (LR warmup to 3e-4, cosine decay to 3e-5 over 13.5T tokens, 4K context), then a 300B-token fast-annealing stage (3e-5→8e-6, 8K context), then long-context extension to 32K and 256K with NTK-aware RoPE (alpha 50 and 1000). Post-training has four sequential steps: reasoning-oriented SFT, reasoning-oriented RL, all-scenarios SFT (reasoning samples mixed with general-domain data), then all-scenarios RL. Reasoning RL itself runs in two context phases, 24K then 32K.

- Pretrain 20T @4K → 300B fast anneal @8K → 32K/256K NTK-aware extension.
- Reasoning SFT → reasoning RL → all-scenarios SFT → all-scenarios RL.
- All-scenarios SFT mixes reasoning samples with general-domain data.
- Reasoning RL staged by context: 24K then 32K.

**Key results:** With 13B active parameters, Hunyuan-A13B (slow-thinking) scores 87.3 on AIME2024, ahead of Qwen3-A22B (85.7) and DeepSeek-R1 (79.8), and leads BFCL v3 (78.3); the base beats Hunyuan-Large on 12/14 benchmarks with 1/4 the active parameters.

*Evolution:* Builds on Hunyuan-Large/TurboS, borrowing DAPO's KL-free GRPO and DeepScaleR's context-length staging; internally dated mid-2025 despite its 2609 arXiv ID.

### Scaling properties of same-family on-policy distillation
*2026 · method · `method_OPD-Scaling-SameFamily_2609.32722.txt` · arXiv [2609.32722](https://arxiv.org/abs/2609.32722)*

Each model follows SFT → GRPO RL (teachers) or SFT → OPD (students). The grid has 25 Vanilla-OPD teacher-student pairs (weak-to-strong, same-base, strong-to-weak) and 17 Delta-OPD pairs, each OPD run capped at 10 epochs (580 updates). Design-choice studies compare pure OPD, an off-policy SFT cold start followed by OPD, and pure OffPD. Bootstrapped chains apply RL only to the 0.5B model and then distill upward stage by stage (e.g. 0.5B→1.5B→3B→7B→14B), each stage's best OPD checkpoint becoming the next teacher (5 Vanilla and 3 Delta chains).

- Teachers: SFT → GRPO; students: SFT → OPD.
- 25 Vanilla + 17 Delta OPD pairs, ≤10 epochs (580 updates) each.
- Compares pure OPD vs off-policy SFT cold start + OPD vs pure OffPD.
- Bootstrapped chains distill upward 0.5B→…→14B, best checkpoint as next teacher.

**Key results:** Every weak-to-strong student peaks above its teacher; the joint law extrapolates to held-out sizes within 0.7/0.4 points; off-policy cold start costs 6.6/15.7/19.4 points for 3B/7B/14B; every bootstrapped chain peaks below direct OPD (76.8 vs 77.7 at 14B).

*Evolution:* Carries Gao et al.'s overoptimization scaling-law method to on-policy distillation, making weak-to-strong OPD predictable and overturning Burns et al.'s bootstrapping gain in this setting (one family, single seeds).

### WEFT: Scaling Tool-Use Post-Training for General-Purpose Agents
*2026 · code · `code_WEFT_2609.36887.txt` · arXiv [2609.36887](https://arxiv.org/abs/2609.36887)*

The pipeline is SFT then RL from the base models (Qwen3-8B, Qwen3-14B, Qwen3.5-35B-A3B; RL results reported for 8B and 14B). Data construction is iterative: execution-driven self-evolution runs up to three rounds in which rollouts are attributed, components revised and fresh rollouts re-collected. Training mixtures vary task disclosure (Agentic full brief vs SimUser incremental requests; vague/partial/full explicitness) and harness (ReAct, then OpenClaw, then Hermes) while holding objectives, dependencies, initial state and completion conditions fixed.

- Two stages: SFT → RL from Qwen3/Qwen3.5 bases.
- Up to three rounds of execution-driven data self-evolution before training.
- Mixtures vary task disclosure and explicitness with fixed task semantics.
- Harness diversity across ReAct, OpenClaw and Hermes.

**Key results:** WEFT-8B/14B beat every matched-size environment-scaling baseline on BFCL V4, tau2-Bench and Claw-Eval (14B +6.41/+2.23/+12.27 over Agent-World-14B); trajectory-level GRPO regresses tau2-Bench vs SFT while atomic-turn GRPO improves all three.

*Evolution:* Responds to environment-scaling work (EnvScaler, ScaleEnv, Agent-World) by repairing tools, tasks and verifiers and adding harness diversity, with finer-grained atomic-turn credit than GiGPO/PivotRL.

### MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement
*2026 · report · `report_MiMo-V2.6_2610.11959.txt` · arXiv [2610.11959](https://arxiv.org/abs/2610.11959)*

Pretraining is two-stage: text first, then omni-modal joint training with an in-house ViT (4T+ image tokens) and an audio encoder, with context extended 32K→256K. Flash sees 48T tokens (26T text + 22T omni) and Pro sees 30T (27T + 3T). Agent-centric mid-training then runs at 256K and finally 1M context, switches hidden matrices from AdamW to the Muown optimizer, applies MXFP4 QAT, and adds reward-hacking self-correction examples. A short SFT follows, then a single large mixed-task RL run ("You Only RL Once", about 30 steps), and finally MOPD2 multi-teacher on-policy distillation merges domain teachers, including SFT teachers for hard-to-verify open domains.

- Pretrain text → omni joint (32K→256K); Flash 48T, Pro 30T tokens.
- Agentic mid-training 256K→1M with optimizer switch to Muown and MXFP4 QAT.
- Short SFT → one mixed-task RL run (~30 steps) → MOPD2.
- MOPD2 includes SFT teachers for hard-to-verify open domains.

**Key results:** ~$2.6M of RL compute lifts MiMo-V2.6-Pro's DeepSWE 58.4→72.6 (Opus 5: 74.0) and held-out-harness pass@1 ~50%→66%; confirmed hacking stays <2% of trajectories; the open 9B distill + RL reaches 66.2 SWE-bench Verified.

*Evolution:* Builds on MiMo-V2-Flash's MOPD and RL infrastructure, moving from verifiable-reward RLVR toward compute spent on agentic graders with systematic leakage and reward-hacking defenses.

### AutoEnvScaling: Automating the Data Flywheel with Terminal Agents
*2026 · code · `code_AutoEnvScaling_2610.14163.txt` · arXiv [2610.14163](https://arxiv.org/abs/2610.14163)*

The curriculum is an online data flywheel. The solver trains on an active pool (256 tasks in the shared-model run); every 16 solver updates the pool is re-checked against the updated solver, out-of-band tasks are dropped, and the proposer builds replacements from the newest rollouts. With frontier proposers (GPT-5.6-sol, Claude Opus 5), weights are fixed and the proposer improves only its harness (memory, skills, tools) across rounds. In the self-improvement setting, Qwen3.6-35B-A3B is first cold-started by SFT on 600 high-quality proposer trajectories (valid rate 32.3%→86.8%), then RL-trained as both proposer and solver. HiL-Bench training uses SFT on 75 ask-for-help trajectories before up to 150 RL steps.

- Active 256-task pool re-banded against the solver every 16 updates.
- Proposer replaces dropped tasks from the newest rollouts.
- Frontier proposers improve harness only; weights fixed.
- Self-improvement: proposer SFT cold start (32.3%→86.8% valid) → joint proposer-solver RL.

**Key results:** Terminal-task generation raises the valid-environment rate up to 25x at up to 75% lower cost; with Qwen3.6-35B-A3B as both roles, Terminal-Bench 2.1 rises 12.2 points (6.1 above Tmax), and without proposer training training collapses.

*Evolution:* Builds on terminal-environment synthesis (Tmax, TermiGen) and proposer-solver self-play (Absolute Zero, R-Zero), letting an agent run the environment-design workflow end to end as a step toward self-built training data.

### Apertus 1.5: From Open Text Models to Open Multimodal Reasoners
*2026 · report · `report_Apertus-1.5_2610.15542.txt` · arXiv [2610.15542](https://arxiv.org/abs/2610.15542)*

Training starts from Apertus 1.0 checkpoints. Continued pretraining (4T tokens for 8B, 2T for 70B) has four stages: (1) unpaired image/audio/text with all-modality targets; (2) paired image-text and audio-transcript data with text-only targets; (3) a cooldown on higher-quality math/code/translation data plus ~3B instruction tokens; (4) long-context extension 8K→32K→65K→131K→262K (50-63B tokens per step, with synthetic CWE/DocOrder/needle tasks). Post-training runs SFT (16K) → long-context SFT at 262K → RLVR → DPO → SDPO charter alignment (8B only). The long-context SFT is needed because RULER@262K fell from 54.8 to 8.3 after SFT. DPO is online only for the 8B, offline then online for the 70B. Thinking is a chat-template flag, not a separate model.

- CPT: unpaired multimodal → paired (text targets) → quality cooldown → 8K→262K extension.
- SFT → long-context SFT → RLVR → DPO → SDPO (8B).
- Long-context SFT restores RULER@262K lost in SFT (54.8→8.3).
- 70B uses offline then online DPO; 8B online only.

**Key results:** Apertus 1.5 8B leads all compared 8B-class models on instruction following (IFEval 87.1, Multi-IF 83.7); thinking lifts MATH-500 51.2→71.4; the 70B improves over Apertus 1.0 everywhere but trails Qwen3.5 and Gemma 4.

*Evolution:* Upgrades fully open Apertus 1.0 in place to image and audio via early fusion, borrowing OLMo 3 prompts, DAPO clipping and online DPO, and adding SDPO self-distillation for charter alignment.

### Efficiently Learning To Reason or Not to Reason: Root-token Policy Optimization for Adaptive Thinking
*2026 · rl · `rl_RPO-AdaptiveThinking_2610.17767.txt` · arXiv [2610.17767](https://arxiv.org/abs/2610.17767)*

Three stages: original LRM → SFT for mode selection → RPO. The SFT loop iteratively refreshes correctness-filtered self-distilled data and keeps training only while default-template accuracy stays within a tolerance ε of the original model; one iteration usually sufficed. SFT is framed as a controllability enabler rather than a capability stage, because over-training degrades the solver (R1-Distill-1.5B MATH500 drops 83.2→75.1 after 1 epoch). RPO then runs fewer than 60 update steps on a balanced easy/hard prompt mix.

- Original LRM → mode-selection SFT → RPO.
- SFT loop stops when accuracy leaves an ε band around the original model.
- SFT kept minimal: one epoch drops R1-Distill-1.5B MATH500 83.2→75.1.
- RPO: <60 updates on a balanced easy/hard prompt mix.

**Key results:** RPO uses ~0.02x AdaptThink's training compute (10,241 vs 522,240 PFLOPs at 8B) and reaches 51.3 average Pass@1 at 20.5% MATH500 Think% on R1-1.5B; at an equal ~10K-PFLOP budget full-trajectory GRPO stays at 90.2% Think%.

*Evolution:* Follows the learn-when-to-think line (AdaptThink, Thinkless, ARM), pushing token-selective RL to a single root routing token as a controllable layer on top of SFT.

### LLM-jp-4 Technical Report
*2026 · report · `report_LLM-jp-4_2610.19331.txt` · arXiv [2610.19331](https://arxiv.org/abs/2610.19331)*

Each model is trained from scratch on ~12T tokens. Pre-training Phase 1 (6.3T tokens, LLM-jp Corpus v4) builds broad knowledge, with sampling weights set from 13 small-scale 7.7B/400B-token runs. Phase 2 (4.2T tokens, Corpus v4.1) emphasizes math and code (The Stack v2, MMWPM-OSS), chosen via a 2^(5-2) fractional-factorial study. Mid-training (1.26T tokens) uses a 50:50 mix of instruction corpus and Phase 2 pretraining data at half the pretraining LR, extending context 4K→16K→64K (0.84T + 0.21T + 0.21T tokens). Post-training is SFT (2 epochs, ~10.83B tokens) then DPO (2 epochs), with no RL. All stages use WSD schedules with 1-sqrt decay to zero; Phase 1 decay branches are forked from the stable run.

- Phase 1 6.3T broad → Phase 2 4.2T math/code, weights set by small-scale and factorial studies.
- Mid-training 1.26T: 50:50 instruction/pretrain mix at half LR, 4K→16K→64K.
- SFT (2 epochs) → DPO (2 epochs); no RL.
- WSD with 1-sqrt decay throughout; decay branches forked from the stable run.

**Key results:** LLM-jp-4-32B-A3B roughly doubles llm-jp-eval (0.307→0.605), matching Qwen3-8B (0.608) and gpt-oss-20b (0.604), and beats gpt-oss-20b on MT-Bench JA (7.82 vs 7.33); still trails on English, GPQA-EN and AIME.

*Evolution:* Successor to LLM-jp-3/3.1 as a fully open Japanese sovereign model, rebuilding license-contaminated datasets and using SFT+DPO distilled from gpt-oss-120b without RL.

### Priming Base Models for Agentic Post-Training: An Open Midtraining Recipe
*2026 · method · `method_Open-Midtraining-Recipe_2610.22019.txt` · arXiv [2610.22019](https://arxiv.org/abs/2610.22019)*

Midtraining is inserted between pretraining (8K context) and post-training: hundreds of billions of tokens at 256K context, ideally before the final 1M long-context stage, so that much agentic learning moves out of SFT and SFT can stay short to preserve later RL trainability. Traces are capped at 10-15% of the blend to avoid SFT-like overfitting, and replay is fixed at 20% (40% gave no clear benefit). LR warms up briefly then stays constant (5e-5 for Lightning, 1e-5 for Super). In experiments, runs start from the 1M long-context checkpoint and go straight to SFT, short (33B tokens) or full (100B tokens; ~430K agentic trajectories, LR 2e-5 cosine to 5e-6). A ~9B-token long-context stage restores Super's RULER@1M 33.0→81.0, and a 20-step GRPO check follows full SFT on Lightning-8K.

- 256K midtraining placed between 8K pretraining and the final 1M stage.
- Agent traces capped at 10-15% of the blend; 20% replay.
- Brief warmup then constant LR; SFT kept short to preserve RL trainability.
- ~9B-token long-context stage restores RULER@1M 33.0→81.0.

**Key results:** On Nemotron 3.5 Lightning, 500B midtraining tokens add 3.8/7.0/3.5 points on SWE-bench Verified/Multilingual/Pro after identical SFT; on Super, 280B tokens lift DeepSWE 11.7→15.7 and improve every harness tested.

*Evolution:* Opens up the midtraining stage that Qwen3-Coder-Next, GLM-5 and Kimi-Dev describe only at a high level, moving natural SWE data and traces out of long SFT; whether it raises what RL can add remains open.

### Self-Evolving Coding Agents via Harness-Policy Optimization
*2026 · code · `code_Self-HPO_2610.22791.txt` · arXiv [2610.22791](https://arxiv.org/abs/2610.22791)*

Self-HPO alternates H-steps (harness search with the policy frozen) and P-steps (RL with the harness frozen) on the schedule Base → H1 → P1 → H2 → P2, stopping when the hard-task set stabilizes, which happened after two rounds. Each H-step evaluates K=6 candidates once per search task, each revising the current best harness from all accumulated experience. Each P-step runs 50 RL updates starting from the previous policy. The evolving hard-task search set acts as an adaptive curriculum for harness search, keeping it pointed at the updated policy's failures.

- Alternating harness search (policy frozen) and RL (harness frozen).
- Schedule Base → H1 → P1 → H2 → P2; stops when hard-task set stabilizes.
- H-step: K=6 candidates; P-step: 50 RL updates from the previous policy.
- Evolving hard-task set beats a frozen initial set (71.6 vs 70.1).

**Key results:** Self-HPO lifts Qwen3.5-35B-A3B's three-benchmark average 57.8%→64.8% (Verified 61.0→69.5), beating RLVR by 2.9 points; evolved harness alone +3.5, trained policy alone +3.4, joint +8.5.

*Evolution:* Joins frozen-policy harness optimization and fixed-harness policy RL, making the RL-trained policy its own next harness proposer with search tasks drawn from what RL left unsolved.

### Muse Glimmer 30B: An On-Device Agentic Model
*2026 · report · `report_Muse-Glimmer_2610.24628.txt` · arXiv [2610.24628](https://arxiv.org/abs/2610.24628)*

Pretraining is logit co-distillation from Muse Spark's top-256 logits, using only a KL loss for the first 95% of training and only CE for the final 5% to correct top-k truncation bias; the final checkpoint is uniformly averaged with three earlier checkpoints spaced 600 steps apart. Mid-training Phase 1 (domain infusion) adds synthetic reasoning traces at a reduced LR. Phase 2 (agentic) extends context 32K→128K, adds tool-use and agent trajectories, and switches to CE because teacher logits are less reliable on that data. Post-training runs SFT (about one epoch per dataset), on-policy distillation from Muse Spark, then multi-objective RL over reward models, verifiers, task success and RLHF. Inserting OPD between SFT and RL consistently gives better downstream RL than going straight from SFT to RL.

- Pretrain: KL-only distillation for 95%, CE-only for final 5%; 4-checkpoint average.
- Mid-train: reasoning-trace infusion at reduced LR → agentic phase 32K→128K on CE.
- SFT → on-policy distillation → multi-objective RL.
- OPD before RL beats going straight from SFT to RL.

**Key results:** At high reasoning Muse Glimmer beats Gemma4-31B and Qwen3.6-27B on MCPAtlas (75.5 vs 54.2/62.5) and SWE-Bench Pro (51.2); trails Qwen3.6-27B on OSWorld-Verified (65.9 vs 75.6); 4-bit EvoPress keeps 75.6 of 75.7 average accuracy.

*Evolution:* Meta's on-device follow-up to Llama 4 distills frontier Muse Spark, pairing pretraining logit co-distillation with Qwen3-style on-policy distillation before RL in a dense model.

### Arctic-SWE: Revisiting the Long Horizon Agent Training by Modeling-System Co-Design
*2026 · code · `code_Arctic-SWE_2610.24914.txt` · arXiv [2610.24914](https://arxiv.org/abs/2610.24914)*

There is no SFT stage: RL starts directly from Qwen3.5-4B/9B. The recipe is first stress-tested by prolonged overfitting on 160 solvable R2E-Gym tasks (32 tasks × 5 rollouts), then trained on 3,606 unchanged R2E-Gym tasks for 800 updates, then on a 7,500-row curated deck (the reported 9B run uses 12,389 rows including strengthened suites). Tasks are drawn without replacement from one global shuffled deck; easy eviction retires a task once a group solves it 8/8, and in the R2E-Gym run hard eviction retires it after three consecutive 0/8 groups. Optimization uses AdamW at constant LR 1e-6, beta2 0.98, with no warmup or decay.

- RL-only from Qwen3.5-4B/9B; no SFT.
- Overfitting stress test → 800 updates on R2E-Gym → curated deck.
- Single shuffled deck without replacement; easy (8/8) and hard (3× 0/8) eviction.
- Constant LR 1e-6, no warmup or decay.

**Key results:** On unchanged R2E-Gym data Arctic-SWE lifts Qwen3.5-4B 47.4%→62.0% SWE-bench Verified, matching FrogNano (61.5%) with ~1/6 the rollouts; with curated data the 9B reaches 65.7/39.6/70.2 on Verified/Pro-v1/Pro-v2.

*Evolution:* Reacts to hard-to-reproduce SWE-agent RL baselines (R2E-Gym, DeepSWE, SkyRL) by assembling known pieces into a reproducible async recipe validated by an RL-overfitting stress test.

### Self-Credit: Credit Assignment for Long-Horizon Agentic RL via Intra-group Rescaling
*2026 · rl · `rl_Self-Credit_2610.28388.txt` · arXiv [2610.28388](https://arxiv.org/abs/2610.28388)*

A single RL stage from the base models (Qwen3-8B, Qwen3-32B, Gemma4-E4B, thinking mode) with no SFT warm-up. The only schedule is an annealing curriculum on the repeated-failure penalty weight w_f, decayed linearly from 1.5 to 1.0: strong penalties early discourage repeatedly unsuccessful retrieval, and weaker ones later allow exploration. Training runs 115 steps on BrowseComp-Plus and 100 steps for deep research, with 16 prompts × 16 rollouts (256 trajectories) per step split into 8 optimizer updates.

- Single RL stage, no SFT warm-up.
- Repeated-failure penalty w_f annealed linearly 1.5→1.0.
- Strong early penalty curbs repeated failed retrieval; weaker late penalty allows exploration.
- 115 / 100 steps; 256 trajectories per step over 8 optimizer updates.

**Key results:** On BrowseComp-Plus Self-Credit lifts Qwen3-8B/32B by 37.0/36.2 points over zero-shot (to 40.6/47.0), beating the strongest RL baselines by 8.5/5.6; with Gemma4-E4B it beats the best baselines by 4.9 on GAIA (39.2).

*Evolution:* Combines compaction-augmented agentic RL with GiGPO-style group turn-level credit matched by retrieved evidence, showing a policy's own summaries can supply process supervision for long-horizon search agents.

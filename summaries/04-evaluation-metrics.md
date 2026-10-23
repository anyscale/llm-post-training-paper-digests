# 04 — Evaluation & Metrics

*Post-training summaries, generated solely from the full-text files in [`../texts/`](../texts/). Papers are sorted by arXiv-ID year (2022→2026), then by corpus order within each year. Each entry synthesizes **one lens only** for that paper; the chronological cross-lens narrative — older trends, how the field evolved, and why the newer methods were proposed — lives in [`EVOLUTION_OVERVIEW.md`](EVOLUTION_OVERVIEW.md).*

**Lens:** the benchmarks and metrics used (AIME, MATH, HumanEval, SWE-bench, Arena, win rate, pass@k, avg@k, accuracy, reward, KL, entropy, …) and the headline results/numbers.

**Coverage:** 148 of the 148 papers contribute substantive content on this lens; papers for which this lens was not a focus are omitted here and appear under their relevant topic files.

---

## 2022

### CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning
*2022 · code · `code_CodeRL_2207.01780.txt` · arXiv [2207.01780](https://arxiv.org/abs/2207.01780)*

CodeRL evaluates code generation on APPS, MBPP (zero-shot and finetuned), and CodeXGLUE using pass@k (k=1,5,1000), n@k (filtered subsets via example-test filtering), plus BLEU-4, CodeBLEU, and EM for CodeXGLUE tasks. The actor-critic RL setup over CodeT5-770M is validated through ablations on return variants and critic-error prediction. Test-time filtering (CodeT-style CS) is shown to add gains especially at pass@1000, with the critic reaching >75% error-prediction accuracy.

- APPS (CodeRL+CodeT5-770M): 2.69% pass@1, 6.81% pass@5, 20.98% pass@1000 (all new SOTA); 8.48% 1@k and 12.62% 5@k matching AlphaCode-1B's k=50000 with only k=1000.
- Zero-shot MBPP: 63.0% pass@80 and 81.8% pass@1000, beating GPT-137B's 61.4% with a far smaller model.
- Ablations: relative+critic returns (Model D) and combined Lce+Lrl are optimal; CS boosts pass@1000.

**Key results:** CodeRL+CodeT5-770M sets APPS SOTA at 2.69% pass@1, 6.81% pass@5, 20.98% pass@1000, and 8.48% 1@k / 12.62% 5@k. Zero-shot on MBPP it reaches 63.0% pass@80 and 81.8% pass@1000, beating GPT-137B (61.4%) with a far smaller model.

*Evolution:* CodeRL extends the REINFORCE/actor-critic line for sequence generation (Ranzato 2016, Bahdanau 2016) and CodeT5 pretraining into code, reacting to the neglect of unit-test signals and the limits of NTP-only SFT. In 2022 it anticipated the later trend of execution-feedback and reward-model-driven post-training for code, showing that test outcomes and critic scores can shape both training and test-time decoding.

### STaR: Bootstrapping Reasoning With Reasoning
*2022 · data · `data_STaR_2203.14465.txt` · arXiv [2203.14465](https://arxiv.org/abs/2203.14465)*

STaR is evaluated on three reasoning domains: per-digit arithmetic accuracy, CommonsenseQA dev accuracy, and GSM8K test accuracy, supplemented by a Prolific human evaluation of rationale quality. Iterative self-bootstrapping with rationalization is compared against few-shot CoT, direct finetuning, and much larger models. Gains are reported across outer-loop iterations, with few-shot CoT near zero on hard arithmetic but a single rationalization iteration providing large jumps.

- Arithmetic: 89.5% overall after 16 iterations vs 76.3% for a 10k-example no-rationale baseline; OOD solves on 9-10 digit sums; one rationalization lifts 2-digit from <1% to 32%.
- CommonsenseQA dev: STaR+rationalization 72.5% vs GPT-J direct-finetune 60.0%, few-shot CoT 36.6%, matching a 30x larger GPT-3 direct-finetuned (73.0%); +35.9% over few-shot, +12.5% over direct.
- GSM8K test: 10.7% (with) / 10.1% (without) rationalization vs 5.8% direct-finetune, 3.1% few-shot CoT.
- Human eval: STaR rationales preferred above few-shot 30% more often (p=.039) and above human rationales 74% more often (p<.001).

**Key results:** GPT-J (6B) + STaR with rationalization reaches 72.5% on CommonsenseQA dev, matching a 30x larger GPT-3 direct-finetuned (73.0%) and beating GPT-J direct-finetuning (60.0%) by +12.5%. Arithmetic reaches 89.5% overall after 16 iterations (vs 76.3% baseline) with OOD generalization to 9-10 digit sums. GSM8K test accuracy rises to 10.7% from 5.8% (direct finetune) and 3.1% (few-shot CoT).

*Evolution:* STaR builds on chain-of-thought prompting (Wei et al. 2022), scratchpads (Nye et al. 2021), and Expert Iteration (Anthony et al. 2017), turning few-shot CoT into an iterative self-bootstrapping fine-tuning loop that uses answer-correctness as a reward. In the 2022 context it helped launch the self-improving-reasoning line (later STaR*, Quiet-STaR, V-STaR/RFT, ReST) that distills model-generated reasoning traces as training signal, anticipating verifier-reward RL and self-play trends in post-training.

### SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions
*2022 · data · `data_SelfInstruct_2212.10560.txt` · arXiv [2212.10560](https://arxiv.org/abs/2212.10560)*

Self-Instruct is assessed on two axes: broad instruction-following via the SUPERNI benchmark (119 tasks × 100 instances, zero-shot, ROUGE-L) and user-oriented instruction quality on a curated set of 252 novel instructions judged by expert authors on a 4-level A/B/C/D scale with reported inter-rater agreement. Baselines span T5-LM (11B), GPT3 (175B), T0/Tk-INSTRUCT (11B), and InstructGPT001/002/003. Data-size and distillation-quality ablations are also reported.

- SUPERNI: GPT3SELF-INST 39.9 vs vanilla GPT3 6.8 (+33.1 absolute), T0 33.1, GPT3+T0-training 37.9, InstructGPT001 40.8; with SUPERNI training added, GPT3SELF-INST+SUPER NI reaches 51.6 (best).
- 252 user-oriented instructions: Cohen's kappa 0.57-0.58 (4-class), 0.75 (binary), Spearman rho 0.81; counting A+B as valid, GPT3SELF-INST beats T0/SUPERNI-tuned baselines by a large margin, trailing InstructGPT001 by only ~5%.
- Distilling outputs via InstructGPT003 adds ~+10% on the user-oriented eval.

**Key results:** GPT3SELF-INST (GPT3 175B finetuned on 52K self-generated instructions) improves over vanilla GPT3 by +33.1 absolute ROUGE-L on SUPERNI (39.9 vs 6.8), nearly matching InstructGPT001 (40.8). On 252 user-oriented instructions it beats public-dataset-tuned baselines by a large margin and lands only ~5% behind InstructGPT001. Distilling the outputs via InstructGPT003 adds ~+10% on the user-oriented eval.

*Evolution:* Building on the 2022 instruction-tuning wave (T0, SUPERNI/FLAN, InstructGPT) and LM-based data generation, SELF-INSTRUCT reacts against the human-annotation cost and opacity of commercial instruction-tuned models, pioneering self-bootstrapped synthetic instruction data that directly enables later open self-instruct-style models (Alpaca, Baize) and motivates distillation/reward-based refinement of synthetic instruction corpora.

## 2023

### AgentTuning: Enabling Generalized Agent Abilities for LLMs
*2023 · code · `code_AgentTuning_2310.12823.txt` · arXiv [2310.12823](https://arxiv.org/abs/2310.12823)*

AgentTuning measures agent skill on AgentBench (6 held-in tasks: ALFWorld, WebShop, Mind2Web, KG, OS, Database; 6 held-out: SciWorld, MiniWoB++, WebArena, HotpotQA, ReWOO, Digital Card Game) with task-specific metrics (Success Rate, Reward, Step SR, F1) aggregated into normalized weighted averages, while general ability is tracked on MMLU, GSM8K, HumanEval, and MT-Bench to detect regressions. The design isolates generalization to unseen agent tasks from held-in performance.

- AgentLM-70B: held-in overall 2.55 (vs Llama-2-70B 2.11, GPT-4 2.13) and held-out overall 1.40 (+176% over Llama-2-70B's 0.51), comparable to GPT-3.5's 1.49; general overall 0.96 (+1% vs Llama-2).
- Held-out gains scale with size: 7B +76%, 13B +57%, 70B +176%.
- General tasks: MMLU 59.5, GSM8K 59.7, HumanEval 28.7, MT-Bench 7.26; error analysis shows sharply reduced invalid-action and repetition failures.

**Key results:** AgentLM-70B achieves held-out overall 1.40 (+176% over Llama-2-70B), roughly matching GPT-3.5 (1.49), and held-in overall 2.55, approaching GPT-4 (2.13). General ability is preserved: overall general score 0.96 vs Llama-2's 0.95 (+1%), e.g. MMLU 59.5 and GSM8K 59.7.

*Evolution:* AgentTuning (Oct 2023) builds on ReAct (CoT+action prompting), Self-Instruct, FLAN-style instruction tuning, and GPT-4 distillation, and directly reacts to the AgentBench finding that open LLMs trailed GPT-3.5/GPT-4 on agent tasks. As one of the earliest multi-task agent-trajectory SFT works, it anticipates the later wave of agentic post-training and tool-use RL that scales trajectory data and moves from SFT toward reinforcement over environment rewards.

### RLTF: Reinforcement Learning from Unit Test Feedback
*2023 · code · `code_RLTF_2307.04349.txt` · arXiv [2307.04349](https://arxiv.org/abs/2307.04349)*

RLTF evaluates on APPS and MBPP using pass@k (k=1,5,10,100,1000) with nucleus sampling (top-p 0.95; temperature 0.6 on APPS, 1.2 on MBPP), over a CodeT5-770M base (also CodeGen 2.7B). Ablations compare online vs offline frameworks, feedback granularity, decoding temperature, and base-model size, with Critic Sampling as a test-time refinement.

- APPS (all difficulties), CodeT5+RLTF: pass@1 1.45%, pass@5 3.78%, pass@10 5.21%, pass@1000 19.92%, beating CodeRL (1.32/3.37/17.84) and PPOCoder (1.32/3.37/17.77) and larger GPT/Codex/AlphaCode baselines; with Critic Sampling pass@1 3.27% and pass@5 7.80%.
- MBPP zero-shot: pass@1 71.3 vs CodeRL 68.1 and PPOCoder 68.2, ahead of all fine-tuned GPT models up to 137B.
- Ablations: online+RLTF is best, fine-grained feedback gives the largest gain, temperature 1.0 outperforms 0.2/0.6; benefits grow with base size (CodeGen 2.7B gains ~1% pass@10).

**Key results:** CodeT5-770M + RLTF on APPS: pass@1 1.45%, pass@5 3.78%, pass@1000 19.92% (all difficulties), SOTA among CodeT5-based RL methods, beating CodeRL and PPOCoder; with Critic Sampling pass@1 rises to 3.27%. On MBPP zero-shot, CodeT5+RLTF scores pass@1 71.3 vs CodeRL 68.1 and PPOCoder 68.2. Ablation: online framework + RLTF is optimal; fine-grained feedback contributes the largest single gain; benefits grow with base-model size (CodeGen 2.7B gains ~1% pass@10).

*Evolution:* RLTF builds on CodeRL (2022) and PPOCoder (2023) by moving code-LLM RL from an offline, coarse-reward regime to an online loop with compiler-parsed, line-level reward shaping -- a 2023 precursor to later execution-feedback / RLVR-style code training that rewards verifiable unit-test signals. Its reliance on hand-crafted Python error categories foreshadows the later need for language-agnostic, automated reward shaping.

### What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
*2023 · data · `data_DEITA_2312.15685.txt` · arXiv [2312.15685](https://arxiv.org/abs/2312.15685)*

DEITA evaluates alignment via MT-Bench (GPT-4 judge, 8 subtasks), AlpacaEval win-rate, the Open LLM Leaderboard (ARC, HellaSwag, MMLU, TruthfulQA, averaged), and a blinded human pairwise study (win/tie/lose, 77% inter-annotator agreement). The framing isolates data-selection effects by holding model/recipe fixed while varying dataset size and composition, including a +DPO stage comparison.

- DEITA-Mistral-7B6K: 7.22 MT-Bench / 80.78% AlpacaEval; DEITA-Mistral-7B10K: 7.32 / 81.67% (SOTA among open-source SFT at 7B/13B).
- DEITA-Mistral-7B6K+DPO: 7.55 MT-Bench / 90.06% AlpacaEval, comparable to Zephyr-beta trained on ~30x more data.
- Leaderboard: +DPO lifts Mistral-7B average by ~5 points and surpasses Zephyr; human eval puts DEITA-LLaMA1-13B6K roughly on par with Vicuna-13B-v1.3 (125K) using 20x less data.
- Data scaling: 3K DEITA-selected samples match full 300K training (100x reduction).

**Key results:** DEITA-Mistral-7B6K+DPO (6K SFT + 10K DPO) reaches 7.55 MT-Bench and 90.06% AlpacaEval, comparable to Zephyr-beta trained on ~30x more data. DEITA-Mistral-7B10K (SFT only) hits 7.32 MT-Bench / 81.67% AlpacaEval, SOTA among open-source SFT 7B/13B models. Data scaling: 3K DEITA-selected samples match full 300K training (100x reduction).

*Evolution:* DEITA builds on the 2023 'less is more for alignment' trend (LIMA) and WizardLM's Evol-Instruct, formalizing automatic data selection across complexity, quality, and diversity for SFT. It anticipates the data-centric turn in post-training, where curated SFT/DPO data rather than raw scale drives alignment, motivating later data-efficient and distillation-based alignment recipes.

### Direct Preference Optimization: Your Language Model is Secretly a Reward Model
*2023 · data · `data_DPO_2305.18290.txt` · arXiv [2305.18290](https://arxiv.org/abs/2305.18290)*

DPO is evaluated across three controlled tasks plus a human study, with models up to 6B parameters: sentiment steering (IMDb, GPT-2-large) measured on the reward-vs-KL frontier under a ground-truth sentiment classifier; summarization (TL;DR, GPT-J 6B) and OOD transfer (CNN/DailyMail) measured by GPT-4 win rate vs reference summaries; and dialogue (Anthropic-HH, Pythia-2.8B) measured by GPT-4 win rate vs preferred test responses. Best-of-N is included as a strong non-RL baseline.

- IMDb: DPO achieves the best reward-KL frontier, dominating PPO and even oracle PPO-GT at every KL.
- TL;DR: DPO ~61% GPT-4 win rate at temp 0.0 vs PPO 57% at its best temp, more temperature-robust, beating Best-of-N; OOD CNN/DailyMail DPO 0.36/0.31 vs PPO 0.26/0.23 (temp 0/0.25).
- Dialogue (Anthropic-HH): DPO is the only efficient method beating the dataset's chosen responses, matching Best-of-128.
- Human study on TL;DR: GPT-4 agrees with humans about as often as humans agree with each other; humans preferred DPO (temp 0.25) over PPO (temp 0) 58% of the time.

**Key results:** TL;DR summarization: DPO ~61% GPT-4 win rate vs reference (temp 0.0), beating PPO's 57% and Best-of-N; OOD CNN/DailyMail DPO 0.36 vs PPO 0.26. IMDb sentiment: DPO achieves the best reward-KL frontier, dominating PPO and oracle PPO-GT. Human eval: DPO (temp 0.25) preferred over PPO (temp 0) 58% of the time.

*Evolution:* DPO (2023) builds on the RLHF pipeline of Christiano (2017), Stiennon (2022) and InstructGPT/Ouyang (2022), reacting against the complexity and instability of PPO-based RLHF by deriving the KL-constrained optimum in closed form. It replaced the reward-model-plus-RL recipe with a single cross-entropy loss, enabling a family of simpler preference-optimization methods (IPO, KTO, SLiC) and becoming a default alignment stage in later post-training stacks.

### A General Theoretical Paradigm to Understand Learning from Human Preferences
*2023 · data · `data_IPO_2310.12036.txt` · arXiv [2310.12036](https://arxiv.org/abs/2310.12036)*

IPO offers no standard LLM benchmarks; evaluation is restricted to tiny synthetic contextual-bandit problems (2- and 3-action, no context) comparing IPO vs DPO learning curves across the regularization parameter τ. Policies are softmax(θ) over 3 logits optimized for 18,000 Adam steps (lr 0.01, mini-batch 9) in JAX/flax/optax, each setting run over 10 seeds with mean and 95% CIs on a 4-core/32GB VM. Findings are qualitative about collapse behavior.

- 2-action asymptotic case with p*(y1≻y2)=1, uniform π_ref: IPO yields π*(y1)=σ(0.5τ^{-1}) (→1/2 as τ→∞, →1 as τ→0); DPO yields π*(y1)=1 for all τ.
- 3-action synthetic: DPO collapses to a deterministic policy whenever one action dominates (D1) or never wins, ignoring π_ref; IPO stays close to π_ref under strong regularization and only approaches the deterministic optimum as τ→0.
- No accuracy/win-rate/pass@k numbers are reported.

**Key results:** Qualitative contributions only. In the 2-action asymptotic case with p*(y1≻y2)=1 and uniform π_ref, IPO yields π*(y1)=σ(0.5τ^{-1}) (→1/2 as τ→∞, →1 as τ→0), while DPO yields π*(y1)=1 for all τ. On 3-action synthetic sets DPO ignores π_ref and collapses to deterministic; IPO remains τ-controlled near π_ref.

*Evolution:* This 2023 theory paper builds on DPO (Rafailov et al., 2023) and RLHF (Christiano et al., 2017; Ouyang et al., 2022), unifying them under ΨPO and diagnosing DPO's overfitting under deterministic preferences. It motivates later BT-free direct alignment variants (e.g., KTO and IPO follow-ups) toward more robust preference optimization.

### LIMA: Less Is More for Alignment
*2023 · data · `data_LIMA_2305.11206.txt` · arXiv [2305.11206](https://arxiv.org/abs/2305.11206)*

LIMA relies primarily on a controlled human pairwise preference study over 300 test prompts comparing against Alpaca 65B, DaVinci003, Bard, Claude (April), and GPT-4 (April), with a GPT-4 annotator run reproducing trends (tie-discounted agreement ~78-82%). It supplements preference with absolute analysis of 50 outputs, a 30-prompt safety check, and 7B ablations using ChatGPT 6-point Likert helpfulness and multi-turn additions.

- Human pairwise (equal-or-preferred): GPT-4 43%, Claude 46%, Bard 58%, DaVinci003 65%; outright beats Alpaca 65B (52K examples); GPT-4 prefers LIMA over itself 19%.
- Absolute analysis (50 outputs): 50% excellent, 88% meet prompt requirements; safety 80% safe on 30 sensitive prompts.
- 7B ablations: Stack Exchange 3.83 vs wikiHow 3.49 (ChatGPT Likert); filtered vs unfiltered ~0.5 gap; 2K→32K quantity scaling plateaus.
- Multi-turn: adding 30 dialogue chains raises excellent-turn share 45.2%→76.1% and cuts failures 15/42→1/46.

**Key results:** LIMA (65B LLaMa, SFT on 1,000 examples, no RLHF) is equal-or-preferred to GPT-4 43%, Claude 46%, Bard 58%, DaVinci003 65%, and beats Alpaca 65B (52K examples) in human pairwise preference; 50% of outputs are rated excellent and 88% meet the prompt. 7B ablations: quality/diversity each add ~0.5 ChatGPT-Likert points while 2K->32K quantity scaling plateaus.

*Evolution:* Building on instruction tuning (Wei, Mishra), RLHF (Ouyang, Bai), and Self-Instruct/Alpaca distillation, LIMA (2023) reacts against million-example alignment by arguing alignment is 'superficial'-mostly style-so a curated 1,000-example SFT set rivals RLHF products. It crystallized the data-quality-over-quantity movement that later drove high-quality SFT curation and informed debates on how much alignment truly requires RLHF.

### Magicoder: Empowering Code Generation with OSS-Instruct
*2023 · data · `data_Magicoder-OSS-Instruct_2312.02120.txt` · arXiv [2312.02120](https://arxiv.org/abs/2312.02120)*

Magicoder uses pass@1 across six code benchmarks: HumanEval/HumanEval+ and MBPP/MBPP+ (EvalPlus adds 80×/35× tests), MultiPL-E (Java, JavaScript, C++, PHP, Swift, Rust), DS-1000 (completion and insertion), and APPS (300-problem subset). Decoding is greedy for HumanEval(+)/MBPP(+)/APPS and temperature 0.2, top-p 0.5-0.95 with multiple samples for MultiPL-E and DS-1000. Baselines include ChatGPT/GPT-3.5 Turbo, GPT-4 Turbo, WizardCoder, StarCoder, CodeLlama-Python, CodeT5+, CodeGen, Mistral, and DeepSeek-Coder.

- MagicoderS-CL-7B: HumanEval 70.7 (matching ChatGPT 72.6) and HumanEval+ 66.5 (beating ChatGPT 65.9); Magicoder-CL-7B 60.4 / 55.5.
- MagicoderS-DS-6.7B: 76.8 HumanEval / 70.7 HumanEval+, surpassing DeepSeek-Coder-Instruct-6.7B with 8× fewer finetuning tokens.
- DS-1000: MagicoderS-CL-7B 37.5 vs WizardCoder-SC-15B 29.2; MultiPL-E rivals WizardCoder-CL-34B with 7B parameters.

**Key results:** MagicoderS-CL-7B surpasses ChatGPT on HumanEval+ pass@1 (66.5 vs 65.9) with only 7B parameters. MagicoderS-DS-6.7B achieves 76.8 pass@1 on HumanEval and beats DeepSeek-Coder-Instruct-6.7B on HumanEval(+)/MBPP(+) using 8× fewer finetuning tokens. On DS-1000, MagicoderS-CL-7B scores 37.5 vs WizardCoder-SC-15B's 29.2.

*Evolution:* OSS-Instruct extends Self-Instruct (2023) and reacts against Evol-Instruct/WizardCoder (2023), whose narrow seed-task sets and fixed heuristics inherit LLM bias; it instead grounds synthetic instruction generation in the abundance of open-source code (The Stack/starcoderdata). In the 2023 wave of synthetic data-driven code instruction tuning, it demonstrates that real-code inspiration plus orthogonal complexity evolution can let a 7B model rival ChatGPT, motivating later data-centric scaling to larger base models and stronger teacher LLMs.

### Scaling Relationship on Learning Mathematical Reasoning with Large Language Models
*2023 · data · `data_RFT-rejection-sampling_2308.01825.txt` · arXiv [2308.01825](https://arxiv.org/abs/2308.01825)*

This study uses a single benchmark, GSM8K, with maj1@1 (greedy-decode accuracy) and maj1@100 (sample 100, majority-vote) as the metrics, plus 8-shot ICL accuracy and pre-training loss as predictors of downstream performance. RFT (rejection sampling fine-tuning) is compared against SFT and prior open augmentation methods (CoRE 41.4, FCS+PCS), with FLOPs/GPU-hour cost estimates across pre-train, SFT, RFT-inference, and RFT stages.

- maj1@1 gains from RFT: LLaMA-7B 35.9→49.3 (+13.4); LLaMA2-7B 41.6→50.3 (+8.7); LLaMA-13B 43.0→52.1 (+9.1); LLaMA2-13B 50.0→55.4 (+5.4).
- RFT adds ~5-6 points maj1@1 (~4 points maj1@100) for 7B/13B but nothing for 33B/65B/70B.
- RFT-U13B beats prior open methods and trails only large proprietary models (GPT-4 92.0, PaLM2 80.7).
- Scaling: SFT accuracy is log-linear in supervised data with diminishing returns for lower-loss bases; pre-training loss is ~negatively linear with SFT/ICL accuracy and a better indicator than parameter count.

**Key results:** LLaMA-7B RFT-U13B reaches 49.3% maj1@1 on GSM8K vs 35.9% SFT (+13.4); LLaMA2-13B reaches 55.4% (+5.4 over SFT). RFT helps weaker (higher pre-training-loss) base models most and adds nothing for 33B/65B/70B, which overfit training paths.

*Evolution:* A 2023 scaling-relationship study building on STaR (Zelikman 2022), Cobbe et al. verifiers, and self-consistency, replacing trained verifiers/MCTS with simple rejection-sampling deduplication of correct CoT paths. It prefigures and motivates the later rejection-sampling/RLAIF data-augmentation wave (e.g., used in LLaMA2 alignment) and correctness-reward RLVR methods for math, while arguing that lowering pre-training loss remains the fundamental lever.

### Statistical Rejection Sampling Improves Preference Optimization
*2023 · data · `data_RSO_2309.06657.txt` · arXiv [2309.06657](https://arxiv.org/abs/2309.06657)*

RSO uses four evaluators: a Proxy Reward Model (T5-XXL win rate vs SFT target), a Gold Reward Model (PaLM 2-S trained on the same data), AutoSxS (PaLM 2-L few-shot pairwise, tie-aware), and human evaluation (Amazon Mechanical Turk side-by-side with pointwise 1-5 quality and a best-choice vote, 3 raters per task). Experiments cover Reddit TL;DR and AnthropicHH, with RSO variants (sigmoid-norm, rso-sample-rank) compared against DPO on T5-large and T5-XXL.

- T5-large (sigmoid-norm, rso-sample-rank) on TL;DR: 92.37% Proxy / 84.40% Gold / 71.86% AutoSxS vs DPO 84.35 / 76.09 / 58.65.
- AnthropicHH: 86.94 / 69.26 / 40.98 vs DPO 51.63 / 67.72 / 24.01.
- T5-XXL: RSO improves AutoSxS over DPO by +1.1% (TL;DR) and +33.1% (AnthropicHH); human raters choose RSO >2x more often than DPO.
- Ablations find β=0.5, γ=0.05 optimal and first-round-rank best for rso-8-sample.

**Key results:** RSO (T5-large, sigmoid-norm, rso-sample-rank) reaches 84.40% Gold Reward and 71.86% AutoSxS on Reddit TL;DR vs DPO 76.09/58.65, and 69.26/40.98 on AnthropicHH vs 67.72/24.01. At T5-XXL scale, RSO improves AutoSxS over DPO by +1.1% (TL;DR) and +33.1% (AnthropicHH); human raters choose RSO >2x more often than DPO.

*Evolution:* RSO reacts to DPO and SLiC (both 2023), arguing that DPO's MLE is mismatched because preference pairs are not sampled from the optimal policy, and that best-of-N rejection sampling (AnthropicHH, Llama2) is an unregularized special case. By using reward-guided statistical rejection sampling to approximate on-policy data, it bridges offline preference optimization and online RLHF, foreshadowing later iterative-DPO and best-of-N / on-policy preference methods that emphasize sampling from the learned policy.

### Enhancing Chat Language Models by Scaling High-quality Instructional Conversations
*2023 · data · `data_UltraChat_2305.14233.txt` · arXiv [2305.14233](https://arxiv.org/abs/2305.14233)*

UltraChat evaluates with ChatGPT-automated judging (preferred over human eval in pilots for stability) on the Vicuna benchmark plus 300 GPT-4-generated questions spanning commonsense, world knowledge, physics, biology, math, reasoning, and writing at varied difficulty. Two protocols are used: pairwise comparison (1-10 scoring, randomized order, Win/Tie/Lose) and independent 1-10 per-segment scoring; TruthfulQA multiple-choice accuracy is also reported.

- UltraLLaMA-13B: 9.02 overall (vs Vicuna-13B 8.96, ChatGPT 9.12); Table 1 average 9.023 (highest open-source).
- Pairwise win rate up to 85% over open-source baselines and 13% above Vicuna.
- TruthfulQA multiple-choice accuracy 0.54, tying Vicuna-13B.

**Key results:** UltraLLaMA-13B scores 9.02 overall (ChatGPT-judged, 1-10) vs Vicuna-13B 8.96 and ChatGPT 9.12 on the curated eval set. Pairwise win rate up to 85% over open-source baselines, 13% above Vicuna. TruthfulQA multiple-choice accuracy 0.54, tying Vicuna-13B.

*Evolution:* Builds on the 2023 Self-Instruct/Alpaca/Vicuna wave of distilling ChatGPT into instruction-tuning data, reacting to the observation that small curated sets (e.g., LIMA) stalled below Vicuna. UltraChat's large-scale synthetic multi-turn SFT corpus later seeded open alignment pipelines such as UltraFeedback.

### UltraFeedback: Boosting Language Models with Scaled AI Feedback
*2023 · data · `data_UltraFeedback_2310.01377.txt` · arXiv [2310.01377](https://arxiv.org/abs/2310.01377)*

UltraFeedback reports reward-model preference-prediction accuracy (OpenAI WebGPT, OpenAI Summarization, Anthropic Helpful, Stanford SHP), best-of-n by AlpacaEval win rate vs text-davinci-003, PPO judged head-to-head by GPT-4 on AlpacaEval/Evol-Instruct/UltraChat, capability retention on nine exact-match benchmarks, and AI-human judge agreement plus a GPT-4-rated critique-quality study.

- UltraRM: fine-grained UltraRM ~71% avg preference accuracy (+4.2 over mixed baselines); UltraFeedback-only beats baselines by >6.3%.
- Best-of-n: AlpacaEval win rate vs text-davinci-003 rises from 76.53% (n=1) to 91.54% (n=16).
- UltraLM-13B-PPO: top average win rate 69.7% on AlpacaEval/Evol-Instruct/UltraChat, +16.8 over base UltraLM-13B and above LLaMA2-70B-Chat.
- Capability: nine exact-match benchmarks show only ~1-point gain; AI-human agreement 59.7% (individuals) / 68.6% (majority); UltraCM critique quality 5.92 (1-7), near gpt-3.5-turbo.

**Key results:** UltraRM trained only on UltraFeedback beats open-source reward models by >6.3% avg preference-prediction accuracy; the fine-grained mixed variant reaches ~71% avg. UltraLM-13B-PPO attains the highest average win rate (69.7%) on AlpacaEval/Evol-Instruct/UltraChat, +16.8% over base UltraLM-13B and surpassing LLaMA2-70B-Chat. Best-of-16 sampling lifts AlpacaEval win rate vs text-davinci-003 from 76.53% to 91.54%.

*Evolution:* Extends RLAIF (Constitutional AI; Lee et al. 2023) and SFT data-engineering lessons (UltraChat, Evol-Instruct) by scaling GPT-4 AI feedback into a large, diverse, fine-grained open preference dataset, proving scaled AI feedback can build strong open chat models. Its released dataset and reward model became a staple substrate for later 2023–2024 preference work (e.g., Zephyr/DPO-style alignment), lowering the data barrier for open-source RLHF.

### WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions
*2023 · data · `data_WizardLM-EvolInstruct_2304.12244.txt` · arXiv [2304.12244](https://arxiv.org/abs/2304.12244)*

WizardLM combines automatic benchmarks (OpenLLM MMLU/ARC/HellaSwag/TruthfulQA, HumanEval pass@1, GSM8k 4-shot pass@1) with GPT-4 judges (AlpacaEval, MT-Bench, and the authors' WizardEval of 218 instructions) and a blind pairwise human ranking by 10 annotators across Relevance, Knowledgeable, Reasoning, Calculation, and Accuracy, with win-rate and Kappa agreement. Ablations isolate data scale (250k vs 70k), seed source (ShareGPT vs Alpaca), and evolver model.

- WizardLM-13b averages 58.96 vs Vicuna-13b 54.60 and Alpaca-13b 43.44; large code/math gains: HumanEval 24.0 vs 12.5, GSM8k 37.15 vs 24.34; WizardEval 89.1 vs 86.9.
- WizardLM-70b reaches 71.33 avg (ChatGPT-3.5 is 76.15) with GSM8k 70.61, HumanEval 42.1, WizardEval 99.7.
- Ablations: 250k > 70k data, ShareGPT seed > Alpaca seed, and an open-source Llama-2-70B-Chat evolver substitutes for ChatGPT.
- Model quality rises monotonically with evolved-instruction complexity across rounds C0-C4.

**Key results:** WizardLM-13b beats Vicuna-13b and Alpaca-13b on average (58.96 vs 54.60 vs 43.44) with HumanEval 24.0 (vs 12.5) and GSM8k 37.15 (vs 24.34); WizardLM-70b reaches 71.33 avg (approaching ChatGPT-3.5's 76.15) and 99.7 on WizardEval. Model quality rises monotonically with evolved-instruction complexity across rounds C0–C4.

*Evolution:* Builds on the early-2023 Self-Instruct/Alpaca/Vicuna line of AI-generated open-domain instruction data, reacting to the observed difficulty skew in human-created ShareGPT. It anticipates the later synthetic-data-scaling and difficulty-aware-curriculum trend in post-training (e.g., WizardMath/Evol series) by showing that LLM-evolved, complexity-controlled SFT data can surpass both human and Self-Instruct data without RLHF.

### ZEPHYR: Direct Distillation of LM Alignment
*2023 · data · `data_Zephyr_2310.16944.txt` · arXiv [2310.16944](https://arxiv.org/abs/2310.16944)*

Zephyr measures intent alignment via MT-Bench (160 multi-turn questions, 8 domains, GPT-4 1-10 scoring) and AlpacaEval (805 single-turn prompts, GPT-4 win-rate vs text-davinci-003), and checks for regressions on the Open LLM Leaderboard (ARC, HellaSwag, MMLU, TruthfulQA). Ablations compare dDPO+dSFT against dSFT variants and dDPO-without-SFT; noted limitations include GPT-4 evaluator bias toward verbose/distilled responses and weak math/coding.

- Zephyr-7B: MT-Bench 7.34 (beating Llama2-Chat-70B's 6.86) and AlpacaEval 90.60% win rate (within 2 std devs of Llama2-Chat-70B's 92.66%).
- Open LLM Leaderboard (leads 7B): ARC 62.03, HellaSwag 84.52, MMLU 61.44, TruthfulQA 57.44; matches ~40B models.
- Ablations: dDPO+dSFT (7.00/86.07) beats dSFT-1 (6.64/85.65), dSFT-2 (6.19/78.54), and dDPO-without-SFT (4.76/30.76).

**Key results:** Zephyr-7B (Mistral-7B + dSFT on UltraChat + dDPO on UltraFeedback) reaches MT-Bench 7.34, surpassing Llama2-Chat-70B (6.86), with AlpacaEval 90.60% win rate. It leads all 7B models on the Open LLM Leaderboard (ARC 62.03, HellaSwag 84.52, MMLU 61.44, TruthfulQA 57.44). Trained in 2-4 hours on 16 A100s with no human annotation and no on-policy sampling.

*Evolution:* Zephyr builds on the self-instruct/dSFT lineage (Alpaca, Vicuna) and the InstructGPT alignment recipe, but replaces the costly human-feedback-plus-PPO stage (as in Llama2-Chat) with Rafailov et al.'s DPO over GPT-4-scored AI feedback, showing a 7B model can match 70B RLHF chat models. In the 2023 open-model wave it popularized the distilled SFT-then-DPO-on-AIF pipeline and motivated later preference-optimization and AI-feedback-distillation work on small aligned models.

### Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations
*2023 · rl · `rl_Math-Shepherd_2312.08935.txt` · arXiv [2312.08935](https://arxiv.org/abs/2312.08935)*

Math-Shepherd evaluates on GSM8K (full test), MATH (MATH500 for verification, full set for RL), and an out-of-distribution Hungarian national final exam (33 questions, 100 pts). Verification is measured via best-of-N (N up to 256, mean of 3 sampling groups); RL uses greedy-decoding accuracy. Baselines include self-consistency (majority voting), ORM, ORM-PPO, and Rejective Sampling Fine-tuning (RFT), and the auto-annotated PRM dataset (~440k solutions, ~4× PRM800K) is compared against human-annotated PRM800K.

- Verifier (LLaMA2-70B PRM): LLaMA2-70B-MetaMath 93.2% GSM8K / 44.5% MATH500; DeepSeek-67B-MetaMath 93.3% / 48.1% (open-source SOTA without tools).
- RL (Mistral-7B step-by-step PPO): GSM8K 77.9%→84.1%, MATH 28.6%→33.0%; combined with verification 89.1% GSM8K / 43.5% MATH.
- PRM beats ORM and self-consistency at all sizes (7B/13B/70B), is more data-efficient (~4% over ORM at 10k instances), and generalizes OOD (PRM beats ORM by 9 points on the Hungarian exam); auto-annotated PRM surpasses human-annotated PRM800K on MATH.

**Key results:** Mistral-7B + step-by-step PPO with Math-Shepherd: GSM8K 77.9%→84.1%, MATH 28.6%→33.0% (greedy); combined with verification, 89.1% GSM8K and 43.5% MATH. DeepSeek-67B-MetaMath + Math-Shepherd verification reaches 93.3% GSM8K and 48.1% MATH—state of the art for open-source models without tools. The auto-annotated PRM dataset (~440k solutions, ~4x PRM800K) yields a verifier that surpasses human-annotated PRM800K on MATH.

*Evolution:* Building directly on the human-annotated PRM line of Uesato et al. (2022) and Lightman et al.'s PRM800K (2023) and on Cobbe et al.'s ORM verifier training (2021), Math-Shepherd makes process supervision scalable by automating step labels via MCTS-style completion, then plugs that PRM into step-by-step PPO. In late-2023 context it demonstrated that high-quality reward-model data could be generated without humans, anticipating the 2024 wave of RLVR/process-reward work and iterative RLHF where verifiers are bootstrapped from the generator itself.

## 2024

### AgentGym: Evolving Large Language Model-based Agents across Diverse Environments
*2024 · code · `code_AgentGym_2406.04151.txt` · arXiv [2406.04151](https://arxiv.org/abs/2406.04151)*

Evaluation centers on the AGENTEVAL benchmark suite (1,160 instructions) with per-environment success rate/reward plus interactive-round counts as the efficiency metric, run on Llama-2-Chat-7B (also 13B and DeepSeek-Coder-1.3B) trained on 8x A100-80GB. Baselines span GPT-3.5/4-Turbo, Claude-3, DeepSeek-Chat, Llama-2-Chat, AgentLM-7B/13B/70B, and the BCbase/BClarge AGENTTRAJ clones. AGENTEVOL surpasses BClarge and rivals GPT-4-Turbo on many tasks while cutting interactive rounds, though closed-source SOTA still leads on BIRD and SciWorld.

- WebShop 76.5 vs 73.5 (GPT-4-Turbo); ALFWorld 88.0 vs 83.0
- BabyAI 82.7 vs 74.19; TextCraft 64.0 vs 60.0; Tool-Weather 70.0 vs 65.0
- Evolution runs 4 iterations, ~20h on 8x A100-80GB, K=1 sample per instruction

**Key results:** AGENTEVOL on Llama-2-Chat-7B beats BClarge on WebShop (76.5 vs 73.5), ALFWorld (88.0 vs 83.0), BabyAI (82.7 vs 74.19), and TextCraft (64.0 vs 60.0), matching/surpassing GPT-4-Turbo on several environments.

*Evolution:* An early-2024 push toward multi-environment agent self-evolution beyond isolated tasks, complementing behavioral-cloning agent-tuning and foreshadowing the later surge of RL/post-training for generalist agents.

### AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials
*2024 · code · `code_AgentTrek_2412.09605.txt` · arXiv [2412.09605](https://arxiv.org/abs/2412.09605)*

Evaluation spans text and vision agent benchmarks: WebArena task success rate (OOD self-hosted sites), ScreenSpot Web grounding accuracy (text/icon/widget), and Multimodal-Mind2Web Element Accuracy, Operation F1, and Step Success Rate across cross-task/cross-website/cross-domain splits. The VLM evaluator reaches 84% human agreement. Fine-tuned Qwen2.5/Qwen2-VL models surpass GPT-4o/GPT-4 on WebArena and ScreenSpot Web, and scaling data from 20% to 100% lifts cross-domain Step SR from 39.5% to 45.0%.

- WebArena: Qwen2.5-32B+AgentTrek 22.40 vs GPT-4o 13.10, GPT-4 14.41
- ScreenSpot Web: Qwen2-VL-7B+AgentTrek 67.4 avg (text 81.7/icon 51.5) vs 30.7 baseline, SeeClick 44.7, CogAgent 50.7
- Multimodal-Mind2Web: AgentTrek+Mind2Web best all splits (cross-domain Step SR 52.6 vs 35.1)

**Key results:** Qwen2.5-32B-Instruct+AgentTrek scores 22.40 on WebArena (beating GPT-4 14.41); Qwen2-VL-7B+AgentTrek reaches 67.4 ScreenSpot Web avg vs 30.7 baseline, at $0.551/verified trajectory.

*Evolution:* Builds on synthetic-trajectory GUI-agent work and reacts to the cost of human-annotated datasets, establishing a low-cost automated data-synthesis paradigm for later scalable GUI-agent training.

### Agentless: Demystifying LLM-based Software Engineering Agents
*2024 · code · `code_Agentless_2407.01489.txt` · arXiv [2407.01489](https://arxiv.org/abs/2407.01489)*

Evaluation is on SWE-bench Lite (300), Lite-S (249), and Verified (500), reporting % Resolved, Avg $ Cost, Avg # Tokens, and % Correct Location at file/function/line. Agentless (GPT-4o) leads open-source approaches on Lite while staying cheap, and is best among GPT-4o-based tools on Verified. Combined file localization hits 81.7%. Ablations isolate the re-ranking lever: majority voting alone yields 77 fixes, +regression 81, +reproduction tests 96, against an all-samples upper bound of 126 (42.0%).

- Lite: 96/300 (32.00%) at $0.70, 78,166 tokens
- Lite-S: 84/249 (33.73%); Verified: 194/500 (38.80%)
- Compared against 26 agent-based baselines

**Key results:** Agentless (GPT-4o) resolves 96/300 (32.00%) on SWE-bench Lite at $0.70 (highest open-source), 84/249 (33.73%) on Lite-S, and 194/500 (38.80%) on Verified (best GPT-4o-based).

*Evolution:* Reacts against the 2024 surge of complex autonomous coding agents, showing a simple prompting pipeline matches/beats them at a fraction of the cost and helping motivate cleaner SWE-bench variants.

### Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions
*2024 · code · `code_Marco-o1_2411.14405.txt` · arXiv [2411.14405](https://arxiv.org/abs/2411.14405)*

Evaluation uses the MGSM multilingual grade-school math benchmark (English and Chinese) with accuracy and Test@N (Test@1/8/32). No standard coding or agentic benchmarks are reported. Marco-o1-MCTS (step) beats the Qwen2-7B-Instruct base on MGSM-En, and the mini-step-32 variant improves MGSM-Zh, though Marco-o1-CoT alone underperforms the base on MGSM-Zh due to English-only CoT data. A qualitative case study compares slang/colloquial translation against Google Translate.

- MGSM-En: Marco-o1-MCTS (step) 90.40% Test@1 vs 84.00% Qwen2-7B (+6.17%)
- MGSM-Zh: Marco-o1-MCTS (mini-step 32) 82.40% vs 76.80% (+5.60%)
- Test@32 reaches 99.60% (En) / 96.80% (Zh)

**Key results:** Marco-o1-MCTS (step): 90.40% on MGSM-En vs 84.00% Qwen2-7B-Instruct (+6.17%); Marco-o1-MCTS (mini-step 32): 82.40% on MGSM-Zh vs 76.80% (+5.60%); Test@32 reaches 99.60%/96.80%.

*Evolution:* An early open attempt to demystify o1 by combining CoT SFT with AlphaZero-style MCTS and self-reflection, extending reasoning models to open-ended/multilingual tasks and motivating later learned-reward RL.

### SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
*2024 · code · `code_SWE-agent_2405.15793.txt` · arXiv [2405.15793](https://arxiv.org/abs/2405.15793)*

Evaluation is on SWE-bench full (2,294) and Lite (300) plus HumanEvalFix (164/language), using % Resolved/pass@1 (all human unit tests pass after the patch), per-instance $ Avg. Cost (capped at $4), and pass@k up to 6. SWE-agent w/ GPT-4 Turbo far exceeds the prior best non-interactive RAG baseline and shows large relative gains over a shell-only interface; file-localization F1 is 59.05% vs 45.47% for BM25+Claude. Ablations cover editor, search, viewer window size, and context management.

- Full SWE-bench: 12.47% (286/2,294) w/ GPT-4 Turbo; Claude 3 Opus 10.46%
- Lite: 18.00% (vs 11.0 shell-only, +64% relative); pass@6 35%
- HumanEvalFix pass@1: 87.7% Python, 89.7% JS, 87.9% Java

**Key results:** SWE-agent w/ GPT-4 Turbo resolves 12.47% (286/2,294) of full SWE-bench and 18.00% of Lite, far above the 3.8% prior best RAG baseline, with 87.7% pass@1 on HumanEvalFix-Python.

*Evolution:* Transposes HCI-style interface-design principles to LM agents, arguing interaction design—not weight changes—drives coding-agent gains, and seeds the SWE-bench leaderboard ecosystem.

### ToolACE: Winning the Points of LLM Function Calling
*2024 · code · `code_ToolACE_2409.00920.txt` · arXiv [2409.00920](https://arxiv.org/abs/2409.00920)*

Primary benchmarks are BFCL-v3 (4,951 cases: 3,951 single-turn, 1,000 multi-turn) with AST, Executable, Relevance, Irrelevance, and Overall metrics, and API-Bank (314 dialogues, 753 calls) with Call, Retrieval+Call, Plan+Retrieval+Call accuracies; general capability is probed via MMLU, HumanEval, GSM8K, CommonSenseQA. ToolACE-8B (LLaMA-3.1-8B-Instruct + LoRA) ranks 3rd overall on BFCL-v3, essentially matching GPT-4-turbo/4o and beating all open-source models on API-Bank. Matched-sample training ablations show ToolACE data dominates xLAM/ToolLLM.

- BFCL-v3: 59.13 overall (Non-live AST 89.27, Exec 90.07, Relevance 85.37, Irrelevance 83.81)
- API-Bank: Call 75.94, Retrieval+Call 47.41, matching GPT-4-turbo (72.43/39.26)
- Matched 25k training: 58.19 vs 40.51 (xLAM), 24.90 (ToolLLM)

**Key results:** ToolACE-8B ranks 3rd on BFCL-v3 at 59.13 overall, competitive with GPT-4-turbo (59.49)/4o (59.29); on API-Bank scores 75.94 (Call)/47.41 (Retrieval+Call), matching GPT-4-turbo and beating all open-source models.

*Evolution:* Extends synthetic-data-for-tool-use work with evolutionary API synthesis, capability-aware loss-based complexity targeting, and dual-layer verification, showing an 8B specialist can rival GPT-4.

### WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
*2024 · code · `code_WebRL_2411.02337.txt` · arXiv [2411.02337](https://arxiv.org/abs/2411.02337)*

Evaluation is on WebArena-Lite (165 test cases across 5 sites) using task success rate (SR). WebRL lifts open LLMs far past their bases and past proprietary APIs and the prior open-LLM SOTA AutoWebGLM; per-site highs include GitLab 46.7% and CMS 54.3% for Llama-8B. The trained 8B ORM verifies success at ~80% accuracy, above GPT-4/Captioner+GPT-4/GPT-4V (~70-73%). Ablations confirm the replay buffer, KL-constrained update, and curriculum each contribute, with stability checked across 4 seeds and 3 generation prompts.

- Llama-3.1-8B: 4.8% -> 42.4% SR
- Llama-3.1-70B: 12.7% -> 49.1%; GLM-4-9B: 6.1% -> 43%
- vs GPT-4-Turbo 17.6%, GPT-4o 13.9%, AutoWebGLM 18.2%

**Key results:** Llama-3.1-8B + WebRL: 4.8% to 42.4% SR on WebArena-Lite (vs GPT-4-Turbo 17.6%); Llama-3.1-70B reaches 49.1% and GLM-4-9B 43%; the 8B ORM verifies success at ~80% accuracy, exceeding GPT-4 verifiers.

*Evolution:* Builds on DigiRL/AWR actor-critic online RL and evol-instruct task generation, adding a self-evolving curriculum and KL-anchored off-policy updates; shows open LLMs can surpass proprietary APIs on web tasks.

### HybridFlow: A Flexible and Efficient RLHF Framework
*2024 · code · `code_veRL-HybridFlow_2409.19256.txt` · arXiv [2409.19256](https://arxiv.org/abs/2409.19256)*

The metric is RLHF throughput (tokens/sec) = total prompt+response tokens / one iteration time, averaged over 5 iterations after 10 warm-up, on 16 machines/128 A100-80GB GPUs, across Llama 7B-70B actor/critic/reference/reward models. Compared against DeepSpeed-Chat v0.14.0, OpenRLHF v0.2.5, and NeMo-Aligner v0.2.0. HybridFlow delivers 1.53x-20.57x throughput gains, with transition overhead cut dramatically by its 3D-HybridEngine. Smaller generation tensor-parallelism reduces generation latency, and strong-scaling efficiency reaches 66.8%.

- PPO: 3.67x (up to 7.84x) vs DS-Chat, 3.25x (up to 5.93x) vs OpenRLHF, 12.52x (up to 20.57x) vs NeMo-Aligner; 70B avg 9.64x
- Transition overhead reduced up to 89.1% (78.2s) vs OpenRLHF; transition time -55.2% (11.7s) avg
- Smaller gen TP (t_g=2 for 7B, 4 for 13B) cuts generation latency 60.3%/36.4%

**Key results:** PPO on 7B-70B Llama: HybridFlow beats DS-Chat 3.67x (up to 7.84x), OpenRLHF 3.25x (up to 5.93x), NeMo-Aligner 12.52x (up to 20.57x); 3D-HybridEngine cuts transition overhead up to 89.1% (78.2s) vs OpenRLHF.

*Evolution:* Unifies single-controller dataflow and multi-controller efficiency, reacting to first-generation RLHF systems' hardcoded placements; released as veRL, it enabled the later shift toward verifiable-reward RL (GRPO/RLVR).

### DataComp-LM: In search of the next generation of training sets for language models
*2024 · data · `data_DataCompLM_2406.11794.txt` · arXiv [2406.11794](https://arxiv.org/abs/2406.11794)*

Evaluation uses 53 downstream base-model tasks via LLM-Foundry (no finetuning): QA, commonsense, math, textbook knowledge, open-ended generation. Three headline metrics are MMLU 5-shot, CORE centered accuracy (22 low-variance tasks rescaled 0=random, 1=perfect), and EXTENDED centered accuracy over all 53. Instruction tuning is measured by AlpacaEval2.0 LC win-rate and 5-shot GSM8K CoT; long-context is probed with multi-document QA. DCLM-BASELINE 7B on 2.6T tokens matches Mistral-7B/Llama3-8B at far less compute, and filtering-model choice swings 7B/280B MMLU from 35% to 44%.

- DCLM-BASELINE 7B (2.6T): 63.7% MMLU, 57.1 CORE, 45.4 EXTENDED (+6.6pp MMLU over MAP-Neo, 40% less compute)
- 7B-1x (280B): MMLU 50.8; fastText (OH-2.5+ELI5) CORE 41.0 vs 35.7 Wikipedia positives
- DCLM-IT: AlpacaEval2.0 LC 16.6, 5-shot GSM8K 2.1->52.5

**Key results:** DCLM-BASELINE 7B on 2.6T tokens reaches 63.7% MMLU 5-shot, +6.6pp over MAP-Neo with 40% less compute and within ~2-3pp of Llama3-8B at 6.6x less compute; DCLM-IT reaches 16.6 AlpacaEval2.0 LC and lifts GSM8K 2.1->52.5.

*Evolution:* Systematizes data-centric LM research at 240T-token/7B scale as a reaction to closed-data models, with the fastText OH-2.5+ELI5 recipe, cooldown model-soup, and multi-scale benchmark seeding later open datasets.

### KTO: Model Alignment as Prospect Theoretic Optimization
*2024 · data · `data_KTO_2402.01306.txt` · arXiv [2402.01306](https://arxiv.org/abs/2402.01306)*

Primary metric is GPT-4-0613 winrate vs the SFT target (helpfulness/harmlessness/conciseness) with 90% binomial CIs, plus human evaluation on 256 OpenAssistant test prompts. Closed-ended benchmarks include MMLU (0-shot EM), GSM8K (8-shot CoT EM), HumanEval (0-shot pass@1), BigBench-Hard (3-shot CoT EM), TydiQA (1-shot GP F1), and AlpacaEval 2. HALOs (DPO, offline PPO) beat non-HALOs significantly at 13B+ (p<0.05); KTO matches/exceeds DPO from 1B-30B and can run with one-y-per-x (72% less data) and still win.

- Zephyr-β-SFT/UltraFeedback GSM8K: 40.0->53.5 (+13.5) under KTO
- Human winrate vs SFT: KTO 72.9%±5.3 vs DPO 62.1%±5.7 (p<0.05)
- GPT-4 winrate: 65.2%±3.6 vs 60.0%±3.7; one-y-per-x KTO 0.631 vs DPO 0.600

**Key results:** KTO matches/exceeds DPO across 1B-30B; on Zephyr-β-SFT/UltraFeedback GSM8K rises 40.0->53.5 (+13.5) over DPO; human-eval winrate vs SFT targets KTO 72.9% vs DPO 62.1% (p<0.05); one-y-per-x KTO (72% less data) still beats DPO (0.631 vs 0.600).

*Evolution:* Reframes alignment through prospect theory and the HALO family, arguing the loss's inductive bias—not just data/reward modeling—drives success, and lowers the data-collection barrier via binary feedback.

### MAGPIE: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing
*2024 · data · `data_Magpie_2406.08464.txt` · arXiv [2406.08464](https://arxiv.org/abs/2406.08464)*

Models are evaluated on AlpacaEval 2 (805 prompts; GPT-4-Turbo-1106 and Llama-3-8B-Instruct baselines), Arena-Hard (500 queries; GPT-4-0314 baseline), and WildBench (1024 real-user tasks) using win rate (WR) and length-controlled win rate (LC) with a GPT-4 judge. Downstream ability is measured on the Open LLM Leaderboard (MMLU, ARC, HellaSwag, TruthfulQA, WinoGrande, GSM8K, MMLU-Redux) and trustworthiness via TrustLLM. MAGPIE SFT alone beats all SFT baselines and even UltraChat-SFT + UltraFeedback-DPO; the MagpieLM-8B-Chat variant tops <10B open models.

- Llama-3-8B-Base + MAGPIE-Pro SFT + MAGPIE-Pro-DPO: AlpacaEval2 LC 50.10%, WR 53.53%
- Surpasses both Llama-3-8B-Instruct and GPT-4-Turbo using only ~400K data vs >10M
- MagpieLM-8B-Chat: AlpacaEval2 LC 58.18, ArenaHard 48.54, WildBench 42.70

**Key results:** Llama-3-8B-Base + MAGPIE-Pro-300K-Filtered SFT + MAGPIE-Pro-DPO reaches AlpacaEval2 LC 50.10% vs GPT-4-Turbo and WR 53.53% vs official Llama-3-8B-Instruct, surpassing both with ~400K data; MagpieLM-8B-Chat ranks #1 <10B (LC 58.18, ArenaHard 48.54, WildBench 42.70).

*Evolution:* Extends Self-Instruct/Evol-Instruct/UltraChat by removing seed-question dependence, extracting an aligned model's own instruction distribution via its chat template to democratize alignment data.

### ORPO: Monolithic Preference Optimization without Reference Model
*2024 · data · `data_ORPO_2403.07691.txt` · arXiv [2403.07691](https://arxiv.org/abs/2403.07691)*

Benchmarks and metrics: AlpacaEval1.0 and AlpacaEval2.0 (GPT-4/GPT-4-turbo judge, win rate vs text-davinci-003 and vs GPT-4), MT-Bench (GPT-4 multi-turn category scores), IFEval (instruction-level prompt/instruction, strict/loose), reward-model win rate vs SFT/PPO/DPO using RM-1.3B, and lexical diversity via Gemini-Pro embeddings. Mistral-ORPO-β (7B) surpasses Zephyr-β and Llama-2-Chat (13B) with single-epoch training on UltraFeedback alone, and the margin over DPO grows with scale.

- Mistral-ORPO-β (7B): 12.20% AlpacaEval2.0, 7.32 MT-Bench, 66.19% IFEval inst-loose
- Mistral-ORPO-α (7B): 11.33% AlpacaEval2.0, 7.23 MT-Bench, 61.63% IFEval inst-loose
- Llama-2+ORPO 9.44% AlpacaEval2.0; Phi-2+ORPO 6.35%; win-rate vs DPO 70.9% (OPT-1.3B/HH-RLHF)

**Key results:** Mistral-ORPO-β (7B): 12.20% AlpacaEval2.0, 7.32 MT-Bench, 66.19% IFEval instruction-level loose—surpassing Zephyr-β and Llama-2-Chat (13B) with single-epoch UltraFeedback training; ORPO win rate vs DPO reaches 70.9% at OPT-1.3B, with margin growing with scale.

*Evolution:* Builds on DPO's reference-free spirit and unlikelihood training, reacting against the unstable SFT->RLHF/PPO pipeline and the cost of a frozen reference model; helps popularize single-stage, reference-free alignment.

### Scaling Laws for Data Filtering—Data Curation cannot be Compute Agnostic
*2024 · data · `data_ScalingLawsFiltering_2404.07177.txt` · arXiv [2404.07177](https://arxiv.org/abs/2404.07177)*

Evaluation is zero-shot on ImageNet-1k plus average accuracy over 18 DataComp tasks: six ImageNet-OOD variants (V2, R, A, Sketch, O, ObjectNet), six VTAB datasets, Food-101, Pascal VOC 2007, Stanford Cars, and MSCOCO/Flickr retrieval, using top-1 accuracy, Mean per Class Recall (selected VTAB), and Mean Recall@1 (retrieval). The key finding is that the Pareto-optimal filtering strength shifts with compute, and the proposed scaling laws accurately extrapolate to ViT-B/16 and ViT-B/32 trained at 3B-34B compute where prior fits had large errors.

- At 128M samples: LAION filtering +7.5% avg accuracy over 18 tasks vs no-filtering
- Beyond ~450M samples: unfiltered common crawl overtakes LAION-filtered data
- Pareto-optimal: Top 10% at 32M, Top 20% at 100-350M, Top 30% beyond 350M

**Key results:** At 128M compute, LAION filtering gives +7.5% avg zero-shot accuracy over 18 tasks vs no-filtering, but beyond ~450M samples unfiltered common crawl wins; the scaling laws predict Pareto-optimal filtering across 32M-640M compute and extrapolate accurately to ViT-B/16/B/32 at 3B-34B compute.

*Evolution:* Extends Kaplan/Chinchilla scaling laws to heterogeneous web data and contrastive CLIP training, reacting against compute-agnostic filtering practices and motivating later compute-adaptive curation.

### Self-Rewarding Language Models
*2024 · data · `data_SelfRewarding_2401.10020.txt` · arXiv [2401.10020](https://arxiv.org/abs/2401.10020)*

Two evaluation axes. Instruction following: GPT-4 head-to-head win/tie/loss over 256 IFT test prompts, human pairwise eval (50 instructions, 3 annotators), AlpacaEval 2.0 win rate vs GPT-4 Turbo over 805 prompts, MT-Bench (GPT-4 grades), and 9 NLP benchmarks. Reward modeling: pairwise accuracy, 5-best %, exact-match %, and Spearman/Kendall tau vs human rankings on Open Assistant eval. The self-reward itself improves across iterations (not just instruction following), and AlpacaEval 2.0 climbs steadily to surpass Claude 2, Gemini Pro, and GPT-4 0613, while NLP benchmarks mostly hold steady as an alignment tax.

- AlpacaEval 2.0 win rate: 9.94% (M1) -> 15.38% (M2) -> 20.44% (M3)
- Reward-modeling pairwise accuracy: 65.1% (SFT) -> 78.7% -> 80.4% -> 81.7% (M3); Spearman 0.253 -> 0.349
- MT-Bench 6.85 -> 7.25; M3 beats SFT head-to-head 62.5%

**Key results:** Llama 2 70B Self-Rewarding M3 reaches 20.44% AlpacaEval 2.0 win rate over GPT-4 Turbo, beating Claude 2 (17.19%), Gemini Pro (16.85%), GPT-4 0613 (15.76%); reward-modeling pairwise accuracy rises each iteration (65.1% SFT -> 81.7% M3), confirming the self-reward improves.

*Evolution:* Folds the reward model into the policy so it co-improves across DPO iterations, building on Iterative DPO/Self-Instruct/LLM-as-a-Judge and motivating later scalable self-alignment and iterative-RLAIF.

### SimPO: Simple Preference Optimization with a Reference-Free Reward
*2024 · data · `data_SimPO_2405.14734.txt` · arXiv [2405.14734](https://arxiv.org/abs/2405.14734)*

Three open-ended chat benchmarks are used: AlpacaEval 2 (805 prompts; LC and raw WR win rate vs GPT-4 Turbo), Arena-Hard v0.1 (500 queries; WR vs GPT-4-0314), and MT-Bench (80 questions; 1-10 single-answer grading), with downstream tasks via the HuggingFace Open Leaderboard (MMLU, ARC, HellaSwag, TruthfulQA, Winograd, GSM8K) and real-user evaluation on Chatbot Arena. The authors note MT-Bench has poor separability and Arena-Hard is the hardest. SimPO beats DPO across settings while cutting runtime ~20% and GPU memory ~10%.

- Gemma-2-9B-it-SimPO: 72.4% AlpacaEval 2 LC, 59.1% Arena-Hard WR
- Ranks 1st among <10B models on Chatbot Arena (lifting Gemma-2-9B-it 36th -> 25th overall)
- Beats DPO by up to +6.4 AlpacaEval 2 LC and +7.5 Arena-Hard WR; beats best baseline by 3.6-4.8 LC

**Key results:** Gemma-2-9B-it-SimPO reaches 72.4% LC win rate on AlpacaEval 2 and 59.1% WR on Arena-Hard, ranking 1st among <10B models on Chatbot Arena; SimPO outperforms DPO by up to 6.4 pts (AlpacaEval 2 LC) and 7.5 pts (Arena-Hard) while cutting DPO runtime ~20% and GPU memory ~10%.

*Evolution:* Builds on DPO and contemporaneous reference-free work like ORPO, fixing the mismatch between DPO's log-ratio reward and the average-log-likelihood used at generation and dropping the reference model; a widely adopted lightweight DPO replacement.

### Tülu 3: Pushing Frontiers in Open Language Model Post-Training
*2024 · data · `data_Tulu3_2411.15124.txt` · arXiv [2411.15124](https://arxiv.org/abs/2411.15124)*

Tülu 3 ships a dedicated evaluation suite, Tülu 3 Eval, via the OLMES toolkit, split into development and unseen sets spanning knowledge, reasoning, code, instruction-following, and safety. The suite is used both to compare against closed frontier models and to isolate the contribution of the RLVR stage on small models.

- Dev set: MMLU (0-shot CoT), PopQA (15-shot), TruthfulQA (MC2), BigBenchHard (3-shot CoT), DROP (F1), MATH (4-shot flex EM), GSM8K (8-shot EM), HumanEval/+ (pass@10), IFEval (prompt-loose pass@1), AlpacaEval 2 (LC winrate), plus a 6-task safety average (XSTest, HarmBench, DAN, JailbreakTrigger, WildJailbreakTest, WildGuardTest).
- Unseen set: MMLU-Pro, GPQA, AGIEval-English, Deepmind Mathematics, BigCodeBench, IFEval-OOD, HREF.
- RLVR gains on 8B: GSM8K 84.3→87.6, MATH 42.0→43.7, IFEval 81.1→82.4.

**Key results:** Tülu 3 70B averages 76.2 on Tülu 3 Eval, surpassing Llama 3.1 70B Instruct (74.1), GPT-4o-mini (69.6), and Claude 3.5 Haiku (75.3). RLVR improves the 8B model on GSM8K (84.3→87.6), MATH (42.0→43.7), and IFEval (81.1→82.4). Tülu 3 405B reaches 80.7 avg (with safety), competitive with GPT-4o (81.6) and DeepSeek-V3 (75.9).

*Evolution:* Tülu 3 extends the open post-training line of Tülu 2/Zephyr-β toward closed-style multi-stage recipes (cf. Llama 3.1), borrowing persona-driven synthetic data, DPO/SimPO, and PPO/RLHF, and formalizing verifiable-reward RL (cf. STaR/VinePPO) as RLVR. Its fully open data/recipe and RLVR stage anticipate the 2024–25 wave of RL-with-verifiable-rewards methods (e.g., GRPO, DeepSeek-R1) and reproducible open post-training.

### DeepSeek-V3 Technical Report
*2024 · report · `report_DeepSeek-V3_2412.19437.txt` · arXiv [2412.19437](https://arxiv.org/abs/2412.19437)*

DeepSeek-V3 is benchmarked against DeepSeek-V2.5, Qwen2.5-72B-Instruct, LLaMA-3.1-405B, Claude-3.5-Sonnet-1022, and GPT-4o-0513 across knowledge, math, code, long-context, open-ended, and reward-modeling axes. The breadth of suites is notable, mixing EM/F1, pass@1, agent tasks, and judge-based win rates.

- Knowledge: MMLU 88.5, MMLU-Pro 75.9, MMLU-Redux (EM); reasoning: GPQA-Diamond 59.1, AIME 2024 39.2, MATH-500 90.2, CNMO 2024.
- Code: HumanEval-Mul, LiveCodeBench (pass@1 and CoT) 40.5, Codeforces 51.6 percentile, SWE-bench Verified 42.0 (agentless), Aider.
- Open-ended: Arena-Hard 85.5 (first open-source >85%), AlpacaEval 2.0 70.0 LC winrate judged by GPT-4-Turbo; RewardBench 87.0 (89.6 with maj@6); plus SimpleQA/C-SimpleQA, FRAMES, LongBench v2, IFEval.

**Key results:** DeepSeek-V3 (671B-total/37B-active MoE) achieves AIME 2024 39.2 Pass@1, MATH-500 90.2 EM, and 85.5 win rate on Arena-Hard (first open-source model above 85%), competitive with GPT-4o and Claude-3.5-Sonnet. Full training costs only 2.788M H800 GPU-hours (~$5.576M), including just ~5K GPU-hours for post-training.

*Evolution:* Building on DeepSeek-V2's MLA/DeepSeekMoE and GRPO recipe, V3 pioneers distilling long-CoT reasoning from the R1 series into a standard aligned MoE model via SFT/RL-generated data, anticipating the 2025 wave of distilling strong reasoning models into efficient base models. It also validates FP8 large-scale training and auxiliary-loss-free MoE balancing, pushing open-source efficiency and quality toward closed-source frontier models.

### Gemma 2: Improving Open Language Models at a Practical Size
*2024 · report · `report_Gemma2_2408.00118.txt` · arXiv [2408.00118](https://arxiv.org/abs/2408.00118)*

Gemma 2 reports a large automated benchmark sweep plus human LMSYS Chatbot Arena Elo and side-by-side win-rate evaluations, alongside memorization and safety probes. The headline metric is Chatbot Arena Elo, which positions small instruct models against well-known reference points.

- Automated: MMLU, GSM8K, ARC-c/e, HellaSwag, Winogrande, AGIEval, DROP, BBH, MATH, HumanEval, MBPP, PIQA, SIQA, Boolq, TriviaQA, NQ.
- Human: LMSYS Chatbot Arena Elo; side-by-side safety and instruction-following win rates vs GPT-4o; 500-scenario multi-turn ratings (1–5, avg 8.4 turns).
- Elo: 27B-IT 1218 (beats Llama-3 70B-IT 1206), 9B-IT 1187 (matches GPT-4-0314 1186), 2.6B-IT 1126 (beats GPT-3.5-Turbo-0613 1116). Pretrained 27B: MMLU 75.2, GSM8K 74.0. Memorization <0.1%; safety via RealToxicity, TruthfulQA, BBQ, CrowS-Pairs, Winobias.

**Key results:** Gemma 2 27B-IT reaches LMSYS Chatbot Arena Elo 1218, beating Llama-3 70B-IT (1206); 9B-IT Elo 1187 matches GPT-4-0314, and 2.6B-IT (1126) beats GPT-3.5-Turbo-0613. Distillation lifts a 2B model from 60.3 to 67.7 average on 3 benchmarks when distilled from a 7B teacher over 500B tokens.

*Evolution:* Builds on the knowledge-distillation tradition (Hinton 2015) and Gemini 1.5's distillation, repurposing it as a pre-training substitute for next-token prediction to push small models far past compute-optimal token counts, reacting to Llama-3/Qwen scaling trends. In the 2024 small-model wave it helps motivate teacher-distilled pre-training and weight-averaged RLHF recipes that later open models increasingly adopt.

### InternLM2 Technical Report
*2024 · report · `report_InternLM2_2403.17297.txt` · arXiv [2403.17297](https://arxiv.org/abs/2403.17297)*

InternLM2 evaluates 6 dimensions and 30 benchmarks through the OpenCompass framework, with explicit contamination checking for math. Alignment is measured by both automatic and LLM-judge/arena-style metrics, and long context is probed to 200K tokens.

- Exams: MMLU, CMMLU, C-Eval, AGIEval, GAOKAO. Language/knowledge: TriviaQA, NQ, C3, RACE-High, FLORES (BLEU). Reasoning: WinoGrande, HellaSwag, BBH, GSM8K, MATH, TheoremQA, MathBench.
- Code: HumanEval, MBPP, MBPP-CN, HumanEval-X. Long-context: L-Eval, LongBench, 200K Needle-in-a-Haystack. Tools: T-Eval, CIBench under ReAct.
- Alignment: AlpacaEval (win rate), MTBench (0–10), AlignBench, CompassArena (win rate), IFEval (accuracy); GSM8K contamination check via train/test/reference LM-loss deltas.
- InternLM2-Chat-20B: AlpacaEval 21.8, MTBench 7.9, AlignBench 6.8, CompassArena 31.4, GSM8K 79.6, MATH 32.4, HumanEval 67.7, near-perfect 200K NIAH; beats GPT-3.5 on HellaSwag (+15.6) and BBH (+26.4).

**Key results:** InternLM2-Chat-20B: AlpacaEval win rate 21.8 (SOTA among compared), GSM8K 79.6, MATH 32.4, HumanEval 67.7, MTBench 7.9, AlignBench 6.8, and near-perfect 200k Needle-in-a-Haystack, beating GPT-3.5 on reasoning. InternLM2-7B lifts GSM8K from 36.0 (base) to 70.8 and scores MMLU 65.8.

*Evolution:* COOL RLHF refines LLaMA2's separate helpful/harmless reward models into a single system-prompt-conditioned reward model and adds multi-round online patching against reward hacking, extending InstructGPT-style PPO. It reflects the 2024 open-LLM push for transparent data pipelines and staged alignment, and anticipates later iterative/online RLHF and process-reward trends.

### The Llama 3 Herd of Models
*2024 · report · `report_Llama3_2407.21783.txt` · arXiv [2407.21783](https://arxiv.org/abs/2407.21783)*

Llama 3's evaluation spans knowledge, reasoning, math, code, multilinguality, long-context, and tool use, with code scored pass@1 and a 7-point pairwise human win-rate study over ~7,000 prompts anchoring the headline quality claim.

- Benchmarks: MMLU, MMLU-Pro, IFEval; GSM8K, MATH, GPQA, ARC-Challenge; HumanEval/HumanEval+, MBPP/EvalPlus, MultiPL-E; MGSM, Multilingual MMLU (7 languages); ZeroSCROLLS, Needle-in-a-Haystack, InfiniteBench; Nexus, API-Bank, API-Bench, BFCL.
- Llama 3 405B: HumanEval pass@1 89.0, MGSM 91.6 (best in class), API-Bank 92.3, BFCL 88.5, 100% Needle-in-a-Haystack to 128K; trails GPT-4o by ~2% on Multilingual MMLU.
- Human eval: 405B roughly on par with GPT-4 (0125), beats GPT-4 on multiturn reasoning/coding but lags on Hindi/Spanish/Portuguese.

**Key results:** Llama 3 405B matches GPT-4 on human pairwise win rate (within margin of error) and is the best openly available model; HumanEval pass@1 89.0, MGSM 91.6 (best), and 100% Needle-in-a-Haystack retrieval to 128K. The 8B/70B variants are best-in-class for their sizes.

*Evolution:* It refines the Llama-2 RLHF recipe (RM + SFT + rejection sampling) and the 2023 DPO trend, deliberately favoring stable SFT/RS/DPO over PPO while scaling human preference data to millions and leaning on self-generated synthetic data across six iterative rounds. The open 405B release and the self-improving synthetic-data loops foreshadow the 2024-25 shift toward synthetic-data-driven post-training and RLVR, and the 405B model became a widely used base and distillation teacher for the open-weight ecosystem.

### MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies
*2024 · report · `report_MiniCPM_2404.06395.txt` · arXiv [2404.06395](https://arxiv.org/abs/2404.06395)*

MiniCPM uses the open UltraEval framework with vLLM inference across Chinese, English, code, math, and commonsense tasks, with QA tasks reporting the better of perplexity and generation. Chat quality is LLM-judged via MTBench, and long context via InfinityBench.

- Benchmarks: C-Eval, CMMLU (Chinese), MMLU (English), HumanEval, MBPP (code), GSM8K, MATH (math), HellaSwag, ARC-e, ARC-c (commonsense), BBH (logic).
- MiniCPM-2.4B ranks highest among SLMs on average, matches Mistral-7B in English and surpasses it in Chinese, beats Llama2-13B except on MMLU/BBH/HellaSwag; 1.2B beats Llama2-7B except HellaSwag.
- MiniCPM-2.4B-DPO raises MTBench from 6.89 (post-SFT) to 7.25, beating Llama2-70B-Chat and zephyr-7B, at small alignment-tax cost. 2.4B-128K matches Mistral-7B-Instruct-v0.2 (ABF1000w) and beats ChatGLM3-6B-128K at 2.5× smaller. MiniCPM-MoE (13.6B total/4B active) matches Llama2-34B. Int4-quantized 2.4B runs at 18 tok/s on iPhone 15 Pro.

**Key results:** MiniCPM-2.4B (2.4B non-embedding params) surpasses Mistral-7B and Llama2-13B on average across C-Eval/CMMLU/MMLU/HumanEval/MBPP/GSM8K/MATH/BBH. MiniCPM-2.4B-DPO reaches MTBench 7.25, beating Llama2-70B-Chat and zephyr-7B. The fitted scaling law yields a compute-optimal data/model ratio of ~192x versus Chinchilla's 20x, and MiniCPM-MoE (4B activated) matches Llama2-34B.

*Evolution:* Building on Kaplan (2020) and Chinchilla (Hoffmann 2022) scaling laws and muP/Tensor Program hyperparameter transfer, MiniCPM reacts to Chinchilla's 20x data/model ratio by empirically recovering a much higher (~192x) compute-optimal ratio under modern overtrained configurations akin to Llama2. Its WSD scheduler and 'anneal on high-quality data' two-stage recipe presage the annealing-with-curated-data and continuous-training practices that became common in later 2024 small-model and efficient-scaling work.

### LLM Pruning and Distillation in Practice: The Minitron Approach
*2024 · report · `report_Minitron_2408.11796.txt` · arXiv [2408.11796](https://arxiv.org/abs/2408.11796)*

Minitron evaluates base models on the LM Evaluation Harness and aligned models with extra instruction-following/code/judge metrics; runtime is measured with TensorRT-LLM in FP8 on a single H100, framing compression as both accuracy- and throughput-driven.

- Base harness: MMLU (5-shot), WinoGrande (5-shot), ARC-Challenge (25-shot), HellaSwag (10-shot), TruthfulQA (0-shot), XL-Sum English (3-shot, 20%), GSM8K (5-shot), HumanEval/MBPP pass@1 (temp 0.2, top-p 0.95).
- Instruct additions: GPQA (0-shot), IFEval (avg of prompt/instruction, loose+strict), BFCLv2 (live accuracy), corrected MT-Bench judged by GPT-4-Turbo.
- MN-Minitron-8B matches/beats Llama 3.1 8B with 40× fewer tokens (380B vs 15T); Mistral compression: GSM8K 55.7%→58.5%, HumanEval 23.8%→36.2%. Aligned models lead on MMLU, GSM8K, GPQA, IFEval, BFCLv2 but trail on HumanEval/MBPP and MT-Bench vs Gemma2.

**Key results:** MN-Minitron-8B matches Llama 3.1 8B accuracy across LM Evaluation Harness benchmarks using 40x fewer training tokens (380B vs 15T), with GSM8k improving 55.7%->58.5% and HumanEval 23.8%->36.2% over the teacher. Llama-3.1-Minitron-4B (width and depth) reaches near-teacher quality with 150x fewer tokens (94B vs 15T); width variant beats depth on MMLU (60.5 vs 58.7) and GSM8K (41.24 vs 16.8). TensorRT-LLM FP8 throughput: 2.7x (depth) and 1.8x (width) speedup over Llama 3.1 8B.

*Evolution:* Builds directly on the original Minitron (arXiv:2407.14679, 2024) pruning+distillation recipe and Gromov et al.'s contiguous-layer depth pruning, extending them to the realistic case where the teacher's pretraining data is private by introducing teacher correction and a downstream-task depth saliency metric. It exemplifies the 2024 shift toward cheap SLM derivation from frontier models and motivates later work on lightweight teacher adaptation and data-efficient compression of proprietary LLMs.

### Mixtral of Experts
*2024 · report · `report_Mixtral_2401.04088.txt` · arXiv [2401.04088](https://arxiv.org/abs/2401.04088)*

Mixtral is evaluated with the authors' own pipeline across commonsense, world knowledge, reading, math, code, aggregates, multilingual, long-context, and bias, with instruct models scored by MT-Bench and LMSys Arena Elo.

- Commonsense: HellaSwag, WinoGrande, PIQA, SIQA, OpenbookQA, ARC-E/C, CommonsenseQA. Knowledge: NQ, TriviaQA. Reading: BoolQ, QuAC. Math: GSM8K (maj@8), MATH (maj@4). Code: HumanEval, MBPP pass@1.
- Aggregates: MMLU 5-shot 70.6%, BBH, AGIEval. Multilingual French/German/Spanish/Italian. Long-context: 100% passkey retrieval, proof-pile perplexity. Bias: BBQ, BOLD.
- Instruct: MT-Bench 8.30, LMSys Arena Elo 1121 — ahead of Claude-2.1 (1117), GPT-3.5-Turbo (1117), Gemini Pro (1111), Llama-2-70b-chat (1077). Mixtral 8x7B matches/beats Llama 2 70B and GPT-3.5: MMLU 70.6%, GSM8K 74.4% (8-shot) vs Llama 2 70B's 69.6%, MATH 28.4% vs 13.8%.

**Key results:** Mixtral 8x7B (13B active / 47B sparse) matches or beats Llama 2 70B and GPT-3.5 with 5× fewer active params: MMLU 70.6%, GSM8K 74.4% (8-shot maj@8) vs Llama 2 70B's 69.6%, MATH 28.4% vs 13.8%. Mixtral 8x7B – Instruct reaches MT-Bench 8.30 and LMSys Arena Elo 1121, the best open-weights model as of Dec 2023, ahead of Claude-2.1 (1117), GPT-3.5-Turbo (1117), Gemini Pro (1111), and Llama-2-70b-chat (1077).

*Evolution:* Mixtral builds on the GShard/Shazeer sparse-MoE line and the Mistral 7B architecture, and adopts DPO (2023) in place of PPO-based RLHF for instruction tuning. As one of the first open-weights sparse-MoE LLMs to match dense 70B quality at 5× lower active compute (Jan 2024), it validated open MoE as a cost-efficient path and motivated the wave of open MoE releases that followed.

### Phi-4 Technical Report
*2024 · report · `report_Phi-4_2412.08905.txt` · arXiv [2412.08905](https://arxiv.org/abs/2412.08905)*

Phi-4 mixes OpenAI simple-evals (temp 0.5), internal-framework metrics, a proprietary PhiBench of team-authored questions, long-context HELMET, live November-2024 AMC exams, and RAI defect/grounding probes. The headline is that a 14B model surpasses its GPT-4o teacher on STEM reasoning.

- simple-evals: MMLU, GPQA-diamond, MATH, HumanEval, MGSM, SimpleQA (F1). Internal: MMLU-pro, HumanEval+, ArenaHard, IFEval. PhiBench (original questions). Long-context HELMET (Recall/SubEM, RAG, nDCG@10 re-rank, ICL/F1, GPT-4o-scored QA/Summ). Nov-2024 AMC-10/12 (avg /150 over 100 runs). RAI: DR1/DR3 defect rates, grounding.
- phi-4 (14B): MMLU 84.8, GPQA 56.1, MATH 80.4, HumanEval 82.6, MMLUPro 70.4, ArenaHard 75.4, PhiBench 56.2. Surpasses GPT-4o on GPQA (56.1 vs ~50.6) and MATH (80.4 vs 74.6); beats Qwen-2.5-14B-Instruct on 9/12 benchmarks; ~90+ on AMC, above larger frontier models. Weak spots: SimpleQA, DROP, IFEval (strict instruction-following).

**Key results:** phi-4 (14B): MATH 80.4 and GPQA 56.1, surpassing its teacher GPT-4o (74.6 / ~50.6) and beating Qwen-2.5-14B-Instruct on 9/12 benchmarks. November 2024 AMC-10/12 average ~90+/150, above GPT-4o and Llama-3.3-70B at far lower inference cost than long-CoT models.

*Evolution:* Extends the Phi family's "textbooks are all you need" synthetic-data thesis beyond GPT-4 distillation, showing curated synthetic data plus token-level preference optimization (PTS) can let a 14B model surpass its teacher on STEM reasoning. Reacts to the late-2024 long-CoT wave (O1, DeepSeek-R1, QwQ) by matching much of their reasoning at ~10x lower cost, motivating token-level rather than trajectory-level preference signals for small models.

### Qwen2.5 Technical Report
*2024 · report · `report_Qwen2.5_2412.15115.txt` · arXiv [2412.15115](https://arxiv.org/abs/2412.15115)*

Qwen2.5 evaluates base, instruct, reward-model, and long-context variants across an unusually wide suite, including live benchmarks, in-house multilingual automatic evals, and a 1M-token passkey task. A key empirical finding is that reward-model benchmarks fail to predict downstream RL quality.

- Base: MMLU, MMLU-Pro, MMLU-redux, BBH, ARC-C, TruthfulQA, Winogrande, HellaSwag, GPQA, TheoremQA, GSM8K, MATH, HumanEval(+), MBPP(+), MultiPL-E, multilingual (Multi-Exam/Understanding/Mathematics/Translation, MGSM, Flores-101).
- Instruct adds: LiveBench 0831, LiveCodeBench 2305–2409, IFEval (strict-prompt), Arena-Hard, MT-Bench, plus in-house English/Chinese/multilingual evals (IF/Knowledge/Comprehension/Coding/Math/Reasoning; AMMLU/JMMLU/KMMLU/IndoMMLU/TurkishMMLU/okapi-MMLU, MGSM8K-extended, BLEnD).
- Reward models scored on RewardBench, RMB, PPE, Human-Preference-Chinese; long context on RULER, LV-Eval, LongBench-Chat, 1M-token passkey.
- Qwen2.5-72B-Instruct beats Llama-3.1-405B-Instruct on MMLU-redux (86.8 vs 81.6), MATH, MBPP, MultiPL-E, LiveCodeBench, Arena-Hard, MTBench at ~6× smaller; Qwen2.5-Plus wins 9/13 vs 72B; Qwen2.5-RM-72B leads PPE and Human-Preference-Chinese.

**Key results:** Qwen2.5-72B-Instruct matches or exceeds Llama-3.1-405B-Instruct (~6x larger) on MMLU-redux 86.8 vs 81.6, plus MATH, MBPP, MultiPL-E, LiveCodeBench, Arena-Hard, and MTBench. Pre-training scaled 7T->18T tokens; post-training = >1M SFT examples, ~150K DPO pairs, then GRPO online RL. Qwen2.5-Turbo reaches 100% on the 1M-token passkey retrieval task; Qwen2.5-RM-72B leads on PPE and Human-Preference-Chinese.

*Evolution:* Qwen2.5 extends the Qwen2/Qwen2.5-Math/Coder lineage and the standard SFT->RLHF recipe with a deliberate offline-DPO-then-online-GRPO two-stage RL design and aggressive 18T-token data scaling, placing it in the 2024 open-weight frontier wave (Llama-3, Gemma2) racing GPT-4o. Its finding that RM-benchmark scores fail to predict downstream RL quality, together with the scaled post-training foundation, motivates better reward-model evaluation and enables later specialized and reasoning models (QwQ, Math, Coder, multimodal).

### Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models — The Story Goes On
*2024 · report · `report_Skywork-Math_2407.08348.txt` · arXiv [2407.08348](https://arxiv.org/abs/2407.08348)*

Skywork-Math evaluates under a zero-shot chain-of-thought framework (as in MetaMath) with top-1 accuracy and a strict regex answer matcher (near-100% precision, lower recall), plus difficulty-, subject-, and bilingual-slice analyses and leakage filtering.

- Primary: GSM8K (1,319 grade-school items) and MATH (5,000 competition items), top-1 accuracy.
- Analyses: per-difficulty (MATH Levels 1–5), per-subject (Algebra, Geometry, …), bilingual EN→ZH transfer (GPT-4-translated benchmarks, English-only training), distractor-injection robustness (CMATH-style, 1–5 distractors), dense vs sparse-MoE (Mixtral-8x7B), 30- vs 10-gram leakage filtering.
- Skywork-Math-Mistral-7B: MATH 51.2%, GSM8K 83.9%, surpassing an early GPT-4 on MATH and beating 70B open models. DeepSeekMath-7B variant: 49.88% / 81.50%; LLaMA2-7B variant: 47.66% / 72.86%.

**Key results:** Skywork-Math-Mistral-7B (SFT only on 2.5M synthetic GPT-4 data) achieves 51.2% on MATH and 83.9% on GSM8K, SOTA among models <10B and surpassing an early GPT-4 on MATH. Stage 2's 0.4M hard problems lift MATH Level 3-5 accuracy markedly, and 2.1M+0.4M(hard) beats 7.5K+0.4M(hard), validating the easy-to-hard curriculum.

*Evolution:* Builds on 2023-24 synthetic-SFT trends (MetaMath, WizardMath/Evol-Instruct, Xwin-Math) and pushes back against the LIMA "less is more" and math-as-emergent-ability beliefs by showing that scaling synthetic SFT data on common 7B models rivals 120B-token continual pre-training (DeepSeekMath). It anticipates later code-integrated and iterative reasoning pipelines (which it names as future work) and motivates larger, harder synthetic-data scaling for math.

### Improve Mathematical Reasoning in Language Models by Automated Process Supervision
*2024 · rl · `rl_OmegaPRM_2406.06592.txt` · arXiv [2406.06592](https://arxiv.org/abs/2406.06592)*

OmegaPRM evaluates math reasoning via PRM-weighted majority voting (% problems solved) at k=64 and accuracy-vs-sample curves on MATH500 and GSM8K, and separately measures PRM step-classification accuracy across label types. Efficiency is quantified as auto-annotations per unit compute.

- Metric: PRM-weighted majority voting (% problems solved) at k=64, plus accuracy-vs-sample curves. Baselines: majority vote, +Math-Shepherd, +PRM800K, +OmegaPRM.
- Gemini Pro: MATH500 51%→69.4%, GSM8K 86.4%→93.6%. Gemma2 27B: MATH500 42.3%→58.2%, GSM8K 74.0%→92.2%.
- PRM step-classification accuracy: pointwise soft label 70.1% vs hard 63.3% vs pairwise 64.2%.
- Efficiency: at equal compute OmegaPRM yields 15M points vs 200K for brute-force Monte Carlo — a 75× speedup.

**Key results:** Gemini Pro: 51%->69.4% on MATH500 and 86.4%->93.6% on GSM8K; Gemma2 27B: 42.3%->58.2% on MATH500 and 74.0%->92.2% on GSM8K, all via OmegaPRM-weighted majority voting. 1.5M auto-annotations collected at 75x the efficiency of brute-force Monte Carlo.

*Evolution:* Builds on PRM800K's human process supervision and Math-Shepherd/MiPS's per-step Monte Carlo automation, importing AlphaGo Zero's MCTS to make step-level reward data cheap and large-scale. As a 2024 PRM-data contribution it predates and motivates the later wave of process-reward-model and RLVR work that needs abundant automated step-level credit assignment.

### ReFT: Reasoning with Reinforced Fine-Tuning
*2024 · rl · `rl_ReFT_2401.08967.txt` · arXiv [2401.08967](https://arxiv.org/abs/2401.08967)*

ReFT's primary metric is value accuracy (final-answer correctness) on GSM8K, SVAMP, MathQA-MCQ, and MathQAnumeric, for both N-CoT and P-CoT, with inference-time majority voting (self-consistency over 100 samples) and outcome-reward-model reranking. Ablations probe reward design, KL strength, and value-model sharing, and a reward-hacking case is documented.

- Primary: value accuracy on GSM8K, SVAMP, MathQA-MCQ, MathQAnumeric (N-CoT and P-CoT).
- CodeLLAMA-7B: GSM8K N-CoT 53.30 vs SFT 43.59 (+9.7), P-CoT 75.28 vs 63.68 (+11.6); averaged +6.7 N-CoT / +7.4 P-CoT. With reranking, P-CoT reaches 81.2 on GSM8K, above GPT-3.5-turbo (78.0) with a 7B model.
- Ablations (GSM8K P-CoT): removing partial reward 74.40; KL beta=0 collapses to 0; separate value model 75.15. Reward hacking observed on MathQA-MCQ N-CoT (A–E answer space lets wrong CoTs earn reward 1). Small models (Galactica-125M, Codeparrot-small, Codegen-350M) also beat their SFT counterparts.

**Key results:** CodeLLAMA-7B + ReFT reaches 75.28 P-CoT accuracy on GSM8K vs 63.68 for SFT (+11.6), and 81.2 with reward-model reranking, surpassing GPT-3.5-turbo (78.0) using only a 7B model. Average gains over SFT across GSM8K/SVAMP/MathQA are +6.7 N-CoT and +7.4 P-CoT for CodeLLAMA, with comparable gains for Galactica-6.7B.

*Evolution:* ReFT is an early-2024 demonstration that outcome-only RL (PPO with answer-derived rewards and no trained reward model) on the same SFT data can substantially beat SFT for math reasoning, anticipating the GRPO/RLVR direction later popularized by DeepSeekMath and DeepSeek-R1. It builds on RLHF/PPO alignment, expert iteration, and self-training, and explicitly motivates later work on process-based rewards and reward-hacking-robust RL for reasoning.

### Technical Report on Slow Thinking with LLMs: II — Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems
*2024 · rl · `rl_STILL-2_2412.09413.txt` · arXiv [2412.09413](https://arxiv.org/abs/2412.09413)*

STILL-2 evaluates on three benchmarks with greedy decoding (max 32K tokens), reporting accuracy and the gain over the Qwen2.5-32B-Instruct CoT backbone. Baselines include o1-preview, DeepSeek-R1-Lite-Preview, QwQ-32B-preview, GPT-4o, and Claude 3.5 Sonnet.

- Benchmarks: MATH-OAI (500 competition problems, relatively easier), AIME2024 (30 hard olympiad problems, high variance), GPQA (198 graduate-level multiple-choice in physics/chemistry/biology).
- Backbone Qwen2.5-32B-Instruct scores 80.0/13.3/43.4 (MATH-OAI/AIME/GPQA); 3.9k-distillation SFT reaches 90.2/46.7/55.1 (+12.8/+251.1/+27.0), approaching o1-preview (85.5/44.6/72.3) on math.
- With 1.1k distilled seed + 1.8k explored instances, SFT reaches 89.8/40.0/56.1. Ablations: removing hard problems hurts AIME most; DPO aligning only the thought (no SFT loss) yields the best average (61.0).

**Key results:** STILL-2 (Qwen2.5-32B-Instruct + 3.9k distilled SFT) reaches 90.2% MATH-OAI and 46.7% AIME2024 (vs 80.0/13.3 backbone), approaching o1-preview's 85.5/44.6 on math. Exploration+self-improvement from only 1.1k seed instances lifts AIME from 33.3% to 40–46.7%.

*Evolution:* STILL-2 revises the authors' own Nov-2024 reward-guided tree-search report, rejecting test-time search and domain-specific reward models in favor of distillation from newly-open R1/QwQ plus self-improvement. Within the 2024 o1-replication wave, it evidences that small distilled long-CoT data plus rejection-sampling/DPO can elicit cross-domain slow thinking, motivating later RL-based and scaled self-improvement recipes.

### Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
*2024 · rl · `rl_TestTimeCompute_2408.03314.txt` · arXiv [2408.03314](https://arxiv.org/abs/2408.03314)*

This study evaluates on MATH test accuracy (500 questions), reporting pass@1 and accuracy as a function of generation budget (up to 256 sampled solutions), with majority voting or PRM/ORM-weighted best-of-N selection and 5-level difficulty bins.

- Metric: MATH test accuracy (500 questions), pass@1 and accuracy-vs-generation-budget (up to 256 samples); selection via majority voting or best-of-N weighted with PRM/ORM. Difficulty binned into 5 levels from base-model pass@1 (oracle) or averaged PRM final-answer score (predicted), two-fold cross-validated.
- Compute-optimal scaling matches/beats best-of-N with up to 4× less compute for revisions (64 vs 256 generations) and PRM search (16 vs 64). PRM beats ORM; ~38% of correct answers are flipped to incorrect by naive revision.
- FLOPs-matched trade: test-time compute of PaLM 2-S* vs a ~14× larger pretrained model using X=6·N·D_pretrain and Y=2·N·D_inference at R=D_pretrain/D_inference = 0.16, 0.79, 22. Test-time wins on easy/medium (and sometimes hard when R≪1), but pretraining wins on the hardest. ReST-optimized revision model degrades.

**Key results:** Compute-optimal test-time scaling outperforms best-of-N by up to 4x on MATH for both PRM search (16 vs 64 generations) and revisions (64 vs 256). In a FLOPs-matched evaluation, test-time compute with PaLM 2-S* beats a ~14x larger pretrained model on easy/medium MATH questions (and sometimes hard, when R<<1), but pretraining wins on the hardest questions. The unsupervised MC-rollout PRM outperforms ORM, and sequential revisions beat parallel sampling but flip ~38% of correct answers without verifier/majority selection.

*Evolution:* Building on process reward models (PRM800k/Lightman), unsupervised PRMs (Math-Shepherd), and Recursive Introspection-style revision models, this 2024 work is an early systematic study of inference-time (test-time) compute scaling and its tradeoff against pretraining FLOPs. It anticipates the 'reasoning model' wave (e.g., o1-style adaptive inference compute) and motivates distilling test-time outputs back into the base model for iterative self-improvement, while cautioning that test-time and pretraining compute are not 1-to-1 exchangeable on the hardest problems.

## 2025

### ACECODER: Acing Coder RL via Automated Test-Case Synthesis
*2025 · code · `code_ACECODER_2502.01718.txt` · arXiv [2502.01718](https://arxiv.org/abs/2502.01718)*

Evaluation spans HumanEval(+), MBPP(+), BigCodeBench (completion/instruct, full/hard), and LiveCodeBench V4, using pass@1 (greedy), sample-average, and oracle pass@N, with Best-of-N selection (top-p, temp 1.0, N=16/32/64) choosing the highest-RM-scoring sample. Reward models are separately scored on RM-Bench (Code/Chat/Math/Safety/Easy/Normal/Hard/Avg). ACECODE-RM-32B tops RM-Bench at 76.1 avg and beats NVIDIA-Nemotron-340B-Reward by 7.5 on coding; the 7B variant lifts Llama-3.1-8B-Ins by +8.4 avg (Best-of-N) and the 32B by +10.7. R1-style RL from Qwen2.5-Coder-7B-Base adds +25% HumanEval-plus and +6% MBPP-plus in 80 steps (48 H100-h), raising BigCodeBench-Instruct-Full from 40.2 to 43.2, near DeepSeek-R1-Distill-Qwen-32B (43.9).

- Code benchmarks: HumanEval(+), MBPP(+), BigCodeBench, LiveCodeBench V4; RM-Bench for reward models.
- ACECODE-RM-32B: 76.1 RM-Bench avg, +7.5 coding over Nemotron-340B-Reward.
- RL rule rewards: +25% HumanEval-plus, +6% MBPP-plus in 80 steps (48 H100-h).

**Key results:** ACECODE-RM-32B tops RM-Bench at 76.1 avg (beating Nemotron-340B-Reward by 7.5 on coding) and boosts Llama-3.1-8B by +8.4 avg Best-of-N; R1-style RL from Qwen2.5-Coder-7B-Base gives +25% HumanEval-plus and +6% MBPP-plus in 80 steps.

*Evolution:* Building on DeepSeek-R1's verifiable-reward RL-from-base recipe, ACECODER supplies the missing test-grounded reward signal that general RMs lack, motivating later code-RL work showing short RL runs from a base coder can rival distilled reasoning models.

### Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training
*2025 · code · `code_Agent-R_2501.11425.txt` · arXiv [2501.11425](https://arxiv.org/abs/2501.11425)*

Backbone is Llama-3.1-8B-Instruct, evaluated over up to 100 rounds (50 in revision-recovery). Metrics are average final reward on WebShop and SciWorld and success rate on TextCraft, with baselines spanning closed models (GPT-3.5/4-Turbo/4o, Claude-3 Haiku/Sonnet, DeepSeek-Chat), open agents (Llama2-Chat-13B, AgentLM-7B/13B, Agent-FLAN), ETO, and a Direct-Revision ablation. Agent-R (iter3) averages 70.71 (WebShop 63.91, SciWorld 70.23, TextCraft 78.00), beating Direct-Revision (62.36), ETO (65.12), and GPT-4o (+5.59% over baselines); in revision-recovery it reaches 46.75 vs 35.67. Ablations show revision trajectories beat self-generated optimal ones, average revision length shrinks across iterations (TextCraft 8.3→4.7→2.6), and repeat-action loops decline.

- Tasks: WebShop, SciWorld (avg final reward), TextCraft (success rate); up to 100 rounds.
- Agent-R iter3: 70.71 avg (63.91/70.23/78.00), +5.59% over baselines and ahead of GPT-4o.
- Revision-recovery 46.75 vs Direct-Revision 35.67; revision length shrinks 8.3→2.6 on TextCraft.

**Key results:** Llama-3.1-8B + Agent-R (iter 3) averages 70.71 across WebShop/SciWorld/TextCraft, beating Direct-Revision (62.36), ETO (65.12), and GPT-4o; revision length drops across iterations (TextCraft 8.3→2.6), indicating earlier error detection.

*Evolution:* Extending MCTS-driven agent training and self-correction RL, Agent-R reacts against behavior cloning from all-correct expert trajectories that cannot recover errors, motivating automatic step-level reflection data and iterative self-play SFT.

### CODE I/O: Condensing Reasoning Patterns via Code Input-Output Prediction
*2025 · code · `code_CodeIO_2502.07316.txt` · arXiv [2502.07316](https://arxiv.org/abs/2502.07316)*

Four base models (Qwen 2.5 Coder 7B, DeepSeek-Coder-v2-Lite 16B MoE, LLaMA 3.1 8B, Gemma 2 27B) are evaluated on 14 benchmarks: WinoGrande, DROP (F1), GSM8K, MATH, GPQA, MMLU-STEM, CRUXEval-I/-O, BBH-EN/-ZH (3-shot), ZebraLogic, KorBench, LiveBench, and a new LeetCode-O (problem-level output-prediction accuracy). All greedy, zero-shot except BBH. CODE I/O and CODE I/O++ beat WebInstruct, OpenMathInstruct2, OpenCoder-SFT-Stage1, and Python-Edu across all four models with balanced, non-regressing gains. Headline averages: Qwen 57.7 (vs 54.8 single-stage), LLaMA 52.1 (vs 49.3), Gemma 61.5 (vs 59.5), DeepSeek-Lite 53.5 (vs 51.6); CODE I/O++ consistently exceeds CODE I/O and gains persist on non-leaked subsets.

- 14 benchmarks: WinoGrande, DROP, GSM8K, MATH, GPQA, MMLU-STEM, CRUXEval-I/O, BBH-EN/ZH, ZebraLogic, KorBench, LiveBench, LeetCode-O.
- Greedy/zero-shot (3-shot BBH); gains balanced across symbolic, logic, math, science, commonsense.
- CODE I/O++ averages: Qwen 57.7, LLaMA 52.1, Gemma 61.5, DeepSeek-Lite 53.5, beating larger baselines.

**Key results:** CODE I/O++ lifts the 14-benchmark average on all four bases vs single-stage tuning (Qwen 57.7 vs 54.8, LLaMA 52.1 vs 49.3, DeepSeek-Lite 53.5 vs 51.6, Gemma 61.5 vs 59.5), outperforming larger corpora (OpenMathInstruct2 14M, WebInstruct 11.6M).

*Evolution:* Building on code-pretraining-enhances-reasoning and execution-prediction learning, CODE I/O reacts against math-only reasoning-data scaling and positions itself as orthogonal to inference-time scaling.

### RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning
*2025 · code · `code_RAGEN_2504.20073.txt` · arXiv [2504.20073](https://arxiv.org/abs/2504.20073)*

Metrics over 256 fixed validation prompts (T=0.5, up to 5 turns) are success rate, rollout token entropy, in-group reward standard deviation (behavioral diversity), total response length, and gradient norm (stability); reward-std and entropy serve as early-warning collapse indicators while gradient-norm spikes mark irreversible collapse. StarPO-S delays or eliminates collapse vs vanilla StarPO across four tasks, and keeping 50% of rollouts avoids collapse entirely in FrozenLake-PPO. The 0.5B StarPO-S model reaches 20.70% (Sokoban) and 21.48% (FrozenLake), comparable to zero-shot GPT-4o (27.73%/26.56%) and Qwen2.5-72B-Instruct (19.53%/23.83%) with ~100x fewer parameters. Bandit generalizes 100% with reasoning vs 81.25% NoThink; BanditRev reaches 67.58% vs 56.25%. Best generalization is at 4 responses/prompt, 5-6 actions/turn, Online-1 rollouts.

- Metrics: success rate, token entropy, in-group reward-std, response length, gradient norm over 256 prompts.
- 0.5B StarPO-S: 20.70% Sokoban / 21.48% FrozenLake, ~100x smaller than GPT-4o/Qwen-72B rivals.
- 50% rollout filtering avoids collapse in FrozenLake-PPO; Bandit generalizes 100% with reasoning.

**Key results:** 0.5B StarPO-S reaches 20.70%/21.48% on Sokoban/FrozenLake, matching GPT-4o and Qwen2.5-72B with ~100x fewer params; keeping 50% highest-variance rollouts avoids collapse in FrozenLake-PPO.

*Evolution:* Extending single-turn RL-for-reasoning into multi-turn stochastic agent training, RAGEN surfaces the Echo Trap instability and motivates reasoning-aware reward shaping and turn-aware optimization for long-horizon agents.

### A Self-Improving Coding Agent
*2025 · code · `code_SICA-self-improving_2504.15228.txt` · arXiv [2504.15228](https://arxiv.org/abs/2504.15228)*

Benchmarks are SWE-Bench Verified (50-question random subset), LiveCodeBench (50 questions), two self-curated synthetic tasks (file editing scored by content closeness; symbol navigation by locating path/to/file.py:line:col), plus AIME 2024 and GPQA Diamond. Accuracy is combined with per-problem dollar cost, wall-clock time, token count, and cache-hit % into a utility score. SWE-Bench Verified accuracy rose 17%→53% over 15 iterations (peak 0.53 at iter 14); file-editing 0.82→0.91; symbol navigation 0.35→0.40 (low due to data quality); LiveCodeBench ~0.65→0.71. AIME/GPQA showed little gain since o3-mini-high alone already scores 87%/79% (agent system averaged 76%). The 15-iteration run cost about $7,000 in API spend.

- Benchmarks: SWE-Bench Verified (50-Q subset), LiveCodeBench (50), two synthetic tasks, AIME 2024, GPQA Diamond.
- Utility score blends accuracy with $-cost, wall-clock, tokens, cache-hit %.
- SWE-Bench Verified 17%→53% over 15 iters; file-editing 0.82→0.91; run cost ~$7,000.

**Key results:** SWE-Bench Verified (50-Q subset) accuracy rises 17%→53% over 15 self-improvement iterations; file-editing 0.82→0.91 and LiveCodeBench ~0.65→0.71, with total run cost ~$7,000; AIME/GPQA showed minimal gains.

*Evolution:* SICA extends ADAS to a self-referential agent editing its full Python codebase, and by demonstrating scaffolding-only (non-weight) self-improvement it motivates joint weight-and-system fine-tuning and automated benchmark generation.

### SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
*2025 · code · `code_SWE-RL_2502.18449.txt` · arXiv [2502.18449](https://arxiv.org/abs/2502.18449)*

Evaluation centers on SWE-bench Verified (500 human-verified GitHub issues), measured by pass@1 with 500 repair samples at temperature 1.0 and top-30 reproduction tests for reranking. Llama3-SWE-RL-70B scores 41.0%, SOTA among <100B models and comparable to GPT-4o+Agentless (38.8%), beating the SFT baseline (36.2%); repair-only with oracle files gives 34.8 vs 29.6 (SFT) vs 5.4 (base greedy), with format accuracy 95.6%. Five OOD benchmarks (zero-shot greedy) probe generalization: HumanEval+ 79.9 (vs 76.2 base), BigCodeBench-Hard I/C 28.4/29.1, CRUXEval-I/O 71.6/75.5 (vs 60.5/61.9), MATH-strict 73.7 (vs 63.2; SFT drops to 54.0), MMLU 86.82. Scaling repair samples 20→160 lifts 33.6→40.0 then plateaus; test samples 1→20 lift 38.8→41.0; Eval Arena thresholds give 0.05-level significance.

- Primary: SWE-bench Verified pass@1, 500 repair samples (T=1.0), top-30 test reranking.
- Llama3-SWE-RL-70B: 41.0% (SOTA <100B), vs SFT 36.2% and GPT-4o+Agentless 38.8%.
- OOD: MATH-strict 73.7 (vs 63.2 base), CRUXEval-I 71.6 (vs 60.5), HumanEval+ 79.9.

**Key results:** Llama3-SWE-RL-70B reaches 41.0% pass@1 on SWE-bench Verified (SOTA for <100B, vs SFT 36.2%); OOD MATH-strict rises to 73.7 (vs 63.2 base) and CRUXEval-I to 71.6.

*Evolution:* Extending DeepSeek-R1's rule-based-reward RL to real-world software engineering via PR-level data and GRPO, SWE-RL is the first to show emergent 'aha-moment' reasoning that generalizes to OOD math/code.

### Beyond Scaling Law: A Data-Efficient Distillation Framework for Reasoning
*2025 · method · `method_DataEfficientDistillation_2508.09883.txt` · arXiv [2508.09883](https://arxiv.org/abs/2508.09883)*

Reasoning benchmarks are AIME 2024, AIME 2025 (30 integer-answer problems each), MATH-500, GPQA Diamond, and LiveCodeBench (Feb-May 2025 snapshot, split easy/medium/hard); general/OOD tests are MMLU, CMMLU, C-EVAL, BBH, MBPP, GSM8K, MATH, Aider Polyglot (pass@2). The primary metric is pass@1 averaged over 16 runs (except GPQA Diamond and MATH-500). NTele-R1-32B-Math (DS-32B base, ~0.8k curated math examples) reaches 81.87% AIME 2024 (+16.24 over base 65.63), 77.29% AIME 2025 (+23.75 over 53.54), and 95.2% MATH-500, beating teachers QwQ-32B (76.25/67.30) and DeepSeek-R1 (79.2/70) and baselines Light-R1-32B-DS, AReal-boba-SFT-32B, Skywork-OR1-32B. Code LCB hard rises to 28.43 (from 10.39); mixed NTele-R1-32B reaches AIME 2024 83.54, LCB hard 30.94, Aider pass@2 25.8 (from 12.4).

- Benchmarks: AIME 2024/2025, MATH-500, GPQA Diamond, LiveCodeBench (easy/med/hard); OOD MMLU/CMMLU/C-EVAL/BBH/MBPP/GSM8K/Aider.
- Primary metric: pass@1 avg over 16 runs (except GPQA-D, MATH-500).
- NTele-R1-32B-Math: 81.87% AIME 2024, 77.29% AIME 2025, beating both teacher models.

**Key results:** NTele-R1-32B-Math (DS-32B, ~0.8k examples) scores 81.87% AIME 2024 and 77.29% AIME 2025, beating teachers QwQ-32B (76.25/67.30) and DeepSeek-R1 (79.2/70); mixed NTele-R1-32B doubles Aider pass@2 (12.4→25.8).

*Evolution:* DED extends data-efficient SFT and DeepSeek-R1 distillation, reacting against the reasoning 'scaling law' by importing RL-style pass-rate filtering and reframing corpus quality via token entropy and PCA latent shift.

### OpenSIR: Open-Ended Self-Improving Reasoner
*2025 · method · `method_OpenSIR_2511.00602.txt` · arXiv [2511.00602](https://arxiv.org/abs/2511.00602)*

The primary metric is Pass@1 averaged over 16 generations (temp 0.6, top-p 0.95, max 4096 tokens; 38912 for reasoning models; math_verify for extraction); Pass@k (k=8–256) on five math benchmarks probes reasoning vs memorisation. OpenSIR averages +3.35 math and up to +4.79 general reasoning across four models, the only self-play method that improves all of them, while self-play baselines give marginal/negative gains (Absolute Zero/R-Zero up to +0.89 or −1.93). Per-model gains: Llama-3.2-3B-Instruct +3.99 math avg (25.58→29.57); Llama-3.1-8B-Instruct +3.27; DeepSeek-R1-Distill-Llama-8B +3.71 math (36.31→40.56) and +4.79 general; Qwen3-8B +2.41 math, +4.41 general. OpenSIR from one seed beats GRPO on >7,000 annotated examples. Removing diversity drops math 2.73 and general 2.50, nearly halving concept coverage (5914→3328); lower solve-rate thresholds (0.5→0.1) cut validity 70.82%→42.31% and accuracy 29.57→25.97. Inter-annotator agreement: Fleiss kappa 0.82 (topic)/0.86 (validity), Kendall W 0.67 (difficulty).

- Primary: Pass@1 avg over 16 generations; Pass@k (8–256) probes memorisation vs reasoning.
- +3.35 math / up to +4.79 general across four models; only self-play method improving all.
- Diversity reward nearly doubles concept coverage (3328→5914); removal drops general by 2.50.

**Key results:** OpenSIR averages +3.35 math and up to +4.79 general reasoning across four models, beating GRPO on >7,000 annotated examples from one seed; on R1-Distill-Llama-8B it improves math 36.31→40.56 and general 21.62→26.41.

*Evolution:* OpenSIR reacts to RLVR and verifier-free self-play, diagnosing that prior self-play collapses to familiar concepts on already post-trained models, and operationalizes open-endedness via difficulty calibration and embedding-based diversity.

### Revisiting Entropy in Reinforcement Learning for Large Reasoning Models
*2025 · method · `method_PositiveAdvantageReweighting_2511.05993.txt` · arXiv [2511.05993](https://arxiv.org/abs/2511.05993)*

Evaluation uses Avg@64 and Pass@64 (64 samples/question, temp 1.0, top-p 1.0) plus token-level entropy, N-gram Diversity, SelfBLEU, Spearman correlations, and log-prob calibration plots. In-domain benchmarks are AIME 2024/2025, MATH500, AMC 2023, Minerva Math; OOD are LiveCodeBench and IF-Eval. AIME 2025 validates Qwen2.5-Math-7B and MATH500 validates Llama-3.1-8B-Instruct. Pos-Adv-Reweight (Entropy-guided) on Qwen2.5-Math-7B achieves the best average Avg@64 among entropy methods (ID 45.66, OOD 19.39) and beats Clip-Higher on 6 of 7 benchmarks; AIME 2024 rises to 34.38/73.33 vs GRPO's 28.75/63.33 and base 10.00/60.00. The entropy-performance correlation is task/metric dependent (strong negative on LiveCodeBench Avg@64), and results generalize to Llama-3.1-8B-Instruct.

- Metrics: Avg@64/Pass@64 (temp 1.0) plus token entropy, N-gram Diversity, SelfBLEU, Spearman correlations.
- ID: AIME 2024/2025, MATH500, AMC 2023, Minerva Math; OOD: LiveCodeBench, IF-Eval.
- Pos-Adv-Reweight: best avg Avg@64 (ID 45.66, OOD 19.39), beats Clip-Higher on 6/7; AIME 2024 34.38/73.33.

**Key results:** Qwen2.5-Math-7B + Pos-Adv-Reweight: AIME 2024 Avg@64 34.38 / Pass@64 73.33, best average Avg@64 (ID 45.66, OOD 19.39), beating Clip-Higher on 6 of 7 benchmarks; ~616 K-means samples match full DAPO-Math-17K.

*Evolution:* Building on the RLVR/GRPO wave and covariance-based entropy-collapse theory, it refines the positive-advantage-token insight into a per-token loss weight, motivating finer-grained entropy-targeted reweighting for agentic and long-context RL.

### Toward Training Superintelligent Software Agents through Self-Play SWE-RL
*2025 · method · `method_SSR-SelfPlay-SWERL_2512.18552.txt` · arXiv [2512.18552](https://arxiv.org/abs/2512.18552)*

Evaluation is on SWE-bench Verified (500 human-verified real GitHub issues) and SWE-Bench Pro public split (731 enterprise-level tasks), measured by resolve rate (patch passing fail-to-pass and pass-to-pass tests via a parser-driven pipeline), one attempt per problem at temperature 1.0/top-p 0.95 with no test-time scaling. SSR shows +10.4 points self-improvement on SWE-bench Verified and +7.8 on SWE-Bench Pro, consistently outperforming human-data baseline RL (same images/hyperparameters but given issue descriptions and test suites) across the entire training trajectory, generalizing to natural-language issues unseen in training. Ablations over 1231 tasks show self-play > repair-only > injection-only; removal+history > removal-only > direct-injection (direct collapses to trivial one-line bugs); solver feedback in the inject reward gives only a slight, largely negligible gain. Evaluation noise is ~2% paired standard error.

- Benchmarks: SWE-bench Verified (500) and SWE-Bench Pro public (731), resolve rate, single attempt.
- SSR: +10.4 on Verified, +7.8 on Pro, beating human-data baseline RL throughout training.
- Ablations: self-play > repair-only > injection-only; direct-injection collapses to trivial bugs.

**Key results:** SSR (CWM-sft 32B base) achieves +10.4 points on SWE-bench Verified and +7.8 on SWE-Bench Pro, consistently beating human-data baseline RL while using no human-labeled issues or test suites.

*Evolution:* SSR extends the agentic-SWE-RL wave and corpus-grounded self-play by removing human-curated issues/tests entirely, framing bug injection/repair as a self-play game over real repositories while flagging scaling instability and reward-hacking risks.

### SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
*2025 · method · `method_SWE-Bench-Pro_2509.16941.txt` · arXiv [2509.16941](https://arxiv.org/abs/2509.16941)*

The primary metric is Pass@1 resolve rate (agent patch passing human-reviewed fail2pass and pass2pass tests) on SWE-Bench Pro. On the public set (N=731) Claude Sonnet 4.5 leads at 43.6%, ahead of Sonnet 4 (42.7%), GPT-5 High (41.8%), Haiku 4.5 (39.5%), Kimi K2 Instruct (27.7%), and GPT-OSS 120B (16.2%). On the commercial set (N=276) the best stay below 20%: Opus 4.1 17.8%, GPT-5 High 15.7%, GPT-5 Medium 14.9%, Gemini 2.5 Pro Preview 10.1%, Sonnet 4 9.1%, GPT-4o 3.6%. Under a 50-turn/$2-cap, GPT-5 High scores 25.9% and Opus 4.1 22.7%. Removing requirements+interface drops GPT-5 25.9%→8.4% and Opus 4.1 22.7%→8.2%. Failure-mode analysis (GPT-5 judge) shows Opus 4.1 fails mainly on wrong solutions (35.9%) and syntax errors (24.2%), Sonnet 4 on context overflow (35.6%), Qwen 3 32B on tool errors (42.0%). Top models score ~23% here versus >70% on SWE-Bench Verified.

- Primary: Pass@1 resolve rate on SWE-Bench Pro public (731) and commercial (276).
- Public best: Sonnet 4.5 43.6%; commercial best: Opus 4.1 17.8%; ~23% here vs >70% on Verified.
- Removing human augmentations collapses GPT-5 25.9%→8.4% and Opus 4.1 22.7%→8.2%.

**Key results:** Claude Sonnet 4.5 tops the public set at 43.6% Pass@1 (N=731); the best models stay below 20% on the commercial set (Opus 4.1 17.8%, N=276), exposing a large capability gap vs SWE-Bench Verified (>70%).

*Evolution:* SWE-Bench Pro extends SWE-Bench/Verified, reacting to contamination and saturation concerns by offering a harder, contamination-resistant, enterprise-realistic yardstick that complements SWE-Bench+ and LiveBench.

### The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models
*2025 · method · `method_ValleyOfCodeReasoning_2510.06101.txt` · arXiv [2510.06101](https://arxiv.org/abs/2510.06101)*

Evaluation is on LiveCodeBench (LCB), a contamination-controlled competitive coding benchmark, using Pass@1 as the headline metric. Two auxiliary training-dynamics metrics are introduced: completion rate (share finishing within 32K tokens) and ` Whitetag`-tag occurrence rate (share beginning with ` Whitetag`). Qwen2.5-7B LCB Pass@1 goes 0.126 (baseline) → 0.055 (1K, valley) → 0.188 (10K) → 0.264 (30K); Llama3.1-8B goes 0.103 → 0.019 → 0.150 → 0.299, the latter doubling its 10K score. On Qwen2.5, additional 6K correct vs incorrect responses both lift LCB ~50% (0.185 vs 0.182), so correctness is irrelevant; easy/medium questions raise it 41% (0.179) versus only 7% for hard (0.137). The pattern holds for the 30K checkpoint (0.352 easy vs 0.296 hard). Completion and ` Whitetag`-tag rates rise log-linearly with data size but only weakly track eval gains across feature subsets.

- Benchmark: LiveCodeBench Pass@1 (contamination-controlled); auxiliary completion and ` Whitetag`-tag rates.
- Qwen2.5-7B: 0.126 → 0.055 (1K valley) → 0.188 (10K) → 0.264 (30K).
- Correctness irrelevant (+50% either way); easy/medium +41% vs hard +7%.

**Key results:** Qwen2.5-7B-Instruct LCB Pass@1: 0.126 baseline → 0.055 at 1K (valley) → 0.188 at 10K → 0.264 at 30K; Llama3.1-8B reaches 0.299 at 30K, doubling its 10K score, with easy/medium beating hard (+41% vs +7%).

*Evolution:* Building on data-efficient reasoning distillation and large-scale coding distillation, it exposes a non-monotonic 'valley' in small models, motivating study of medium-high and high (>100K) data regimes and stage-aware curation.

### λ-GRPO: Unifying the GRPO Frameworks with Learnable Token Preferences
*2025 · method · `method_lambda-GRPO_2510.06870.txt` · arXiv [2510.06870](https://arxiv.org/abs/2510.06870)*

Evaluation is on 8 math benchmarks via the SimpleRL/Qwen-Math codebase: GSM8K, MATH500, Minerva, Gaokao, OlympiadBench, College Math, AIME24, AMC23. Pass@1 (single sample) is used for the first six and Average@32 (32 rollouts, mean Pass@1) for AIME and AMC. λ-GRPO reaches average accuracy 37.8 / 43.8 / 53.5 for Qwen2.5-1.5B / 3B / 7B, beating GRPO by +1.9% / +1.0% / +1.7% and DAPO by +1.3% / +1.0% / +1.6%, with the largest gains on harder benchmarks (MATH500, Minerva, Gaokao, AIME, AMC). Training diagnostics show λ-GRPO sustains higher token-level entropy than DAPO across steps 80–160 (more diversity, less collapse) while keeping response length comparable (≈660–690 tokens), i.e. better exploration without added verbosity.

- 8 math benchmarks: GSM8K, MATH500, Minerva, Gaokao, OlympiadBench, College Math, AIME24, AMC23.
- Pass@1 (first six), Average@32 (AIME/AMC); Qwen2.5-1.5B/3B/7B averages 37.8/43.8/53.5.
- Beats GRPO by +1.9/+1.0/+1.7 and DAPO by +1.3/+1.0/+1.6; higher entropy than DAPO at similar length.

**Key results:** Qwen2.5-1.5B/3B/7B reach average accuracy 37.8/43.8/53.5 over 8 math benchmarks, +1.9%/+1.0%/+1.7% over vanilla GRPO and +1.3%/+1.0%/+1.6% over DAPO, while sustaining higher token-level entropy at similar response length with no data or compute changes.

*Evolution:* Building on GRPO and the length-bias fixes DAPO/Dr. GRPO, λ-GRPO recasts their fixed token-aggregation heuristics as special cases of a learnable token-preference framework, motivating adaptive data-driven weighting for verifiable-reward RL.

### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
*2025 · report · `report_DeepSeek-R1_2501.12948.txt` · arXiv [2501.12948](https://arxiv.org/abs/2501.12948)*

Benchmarks span MMLU/Redux/Pro, C-Eval, CMMLU, IFEval, FRAMES, DROP, GPQA Diamond, SimpleQA, C-SimpleQA, SWE-Bench Verified, Aider, LiveCodeBench (2024-08 to 2025-01), Codeforces, CNMO 2024, AIME 2024, and MATH-500, plus AlpacaEval 2.0 LC-winrate and ArenaHard via GPT-4-Turbo pairwise and Chatbot Arena Elo. Metrics include pass@1 (temp 0.6, top-p 0.95), cons@64/cons@16 majority vote, EM, 3-shot F1, and accuracy. DeepSeek-R1 scores AIME 2024 pass@1 79.8% (cons@16 86.7%) vs o1-1217 79.2% and DeepSeek-V3 39.2%; MATH-500 97.3; Codeforces 96.3rd percentile / rating 2029; LiveCodeBench 65.9; GPQA Diamond 71.5; MMLU 90.8; ArenaHard 92.3; AlpacaEval2.0 87.6. R1-Zero AIME rose 15.6%→77.9%. Distilled Qwen-32B hits AIME 72.6 / MATH 94.3; Llama-70B AIME 70.0. AIME 2025 75% (o1 80%); AMC 12 143.7/150. Total cost ~147K H800-hrs (~$294K).

- Broad suite: MMLU/Pro, GPQA Diamond, AIME 2024/2025, MATH-500, LiveCodeBench, Codeforces, SWE-Bench Verified, ArenaHard, AlpacaEval2.0.
- Metrics: pass@1 (temp 0.6/top-p 0.95), cons@16/cons@64 majority vote, EM, 3-shot F1.
- R1: AIME 2024 79.8% (cons@16 86.7%), MATH-500 97.3, Codeforces 96.3rd pct; R1-Distill-Qwen-32B AIME 72.6.

**Key results:** DeepSeek-R1 (671B MoE / 37B active) scores 79.8% pass@1 on AIME 2024 (86.7% cons@16) and 97.3 on MATH-500, matching o1-1217 (79.2%) and far exceeding DeepSeek-V3 (39.2%); on Codeforces it reaches the 96.3rd percentile (rating 2029).

*Evolution:* Building on GRPO and outcome-only RL/STaR, R1 reacts against SFT-first RLHF by showing pure RL on a strong base elicits emergent long-CoT reasoning, catalyzing the open-source reasoning-RL wave (Open-R1, SkyRL, DAPO).

### DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
*2025 · report · `report_DeepSeek-V3.2_2512.02556.txt` · arXiv [2512.02556](https://arxiv.org/abs/2512.02556)*

Benchmarks span MMLU-Pro (EM), GPQA Diamond (Pass@1), HLE text-only (Pass@1), LiveCodeBench (Pass@1-CoT), Codeforces (Rating), AIME 2025 / HMMT Feb & Nov 2025 / IMOAnswerBench (Pass@1), Terminal Bench 2.0 (Acc), SWE-Verified and SWE Multilingual (Resolved), BrowseComp/BrowseCompZh (Pass@1), tau2-bench (Pass@1), MCP-Universe (Success Rate), MCP-Mark (Pass@1), Tool-Decathlon (Pass@1), plus ChatbotArena Elo, AA-LCR, and Fiction.liveBench; eval uses temp 1.0, 128K context, thinking mode. DeepSeek-V3.2-Thinking reaches AIME 93.1, HMMT-Feb 92.5, HLE 25.1, Codeforces 2386, SWE-Verified 73.1, Terminal Bench 2.0 46.4 — comparable to GPT-5-High and below Gemini-3.0-Pro. DeepSeek-V3.2-Speciale reaches AIME 96.0, HMMT-Feb 99.2, HLE 30.6, Codeforces 2701, with gold medals at IMO, IOI, ICPC-WF, and CMO 2025. DSA cuts 128K decoding cost from $2.4M to $0.6M per million tokens.

- Suite: MMLU-Pro, GPQA-D, HLE, LiveCodeBench, Codeforces, AIME/HMMT 2025, SWE-Verified/Multilingual, Terminal Bench 2.0, BrowseComp, tau2-bench, MCP-Universe/MCP-Mark, Tool-Decathlon, ChatbotArena Elo.
- Eval: temp 1.0, 128K context, thinking mode; Pass@1/Acc/Resolved/Success Rate per task.
- V3.2-Thinking: AIME 93.1, SWE-Verified 73.1, Codeforces 2386 (≈GPT-5-High); Speciale: AIME 96.0, Codeforces 2701, gold at IMO/IOI/ICPC-WF.

**Key results:** DeepSeek-V3.2-Speciale: AIME 2025 96.0 Pass@1, HMMT Feb 99.2, Codeforces 2701, gold medals at IMO/IOI/ICPC-WF/CMO 2025; V3.2-Thinking matches GPT-5-High on reasoning (AIME 93.1) with SWE-Verified 73.1 and Terminal Bench 2.0 46.4.

*Evolution:* Building on DeepSeek-R1's RL-for-reasoning and V3/V3.1's MoE+MLA, it scales post-training RL compute beyond 10% of pre-training and synthesizes agentic environments at scale, approaching frontier parity with GPT-5 and Gemini-3.0-Pro.
### GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models
*2025 · report · `report_GLM-4.5_2508.06471.txt` · arXiv [2508.06471](https://arxiv.org/abs/2508.06471)*

Evaluated on 12 ARC benchmarks: MMLU-Pro, AIME 24, MATH-500, SciCode, GPQA, HLE, LiveCodeBench (2407-2501), SWE-bench Verified, Terminal-Bench, TAU-Bench, BFCL V3, BrowseComp; AIME is Avg@32, GPQA Avg@8, with LLM-based answer validation. GLM-4.5 scores 91.0% AIME 24, 70.1% TAU-Bench, 64.2% SWE-bench Verified, 77.8% BFCL V3, 26.4% BrowseComp, 79.1% GPQA, 72.9% LCB, 14.4% HLE, 84.6% MMLU-Pro, 98.2% MATH-500, 41.7% SciCode, 37.5% Terminal-Bench. General chat: MMLU 90.0, SimpleQA 26.4, IFEval 86.1, SysBench 81.0, MultiChallenge 52.8; safety via SafetyBench 89.9. GLM-4.5 ranks 3rd overall and 2nd on agentic tasks (as of July 28, 2025). A custom CC-Bench (52 Claude-Code tasks) gives head-to-head wins of 40.4% vs Claude Sonnet 4, 53.9% vs Kimi K2, 80.8% vs Qwen3-Coder, with 90.6% tool-calling success; human eval on 660 prompts (0-10) also reported.

- 12 ARC benchmarks: MMLU-Pro, AIME 24 (Avg@32), MATH-500, SciCode, GPQA (Avg@8), HLE, LCB, SWE-bench Verified, Terminal-Bench, TAU-Bench, BFCL V3, BrowseComp.
- GLM-4.5: AIME 24 91.0%, TAU-Bench 70.1%, SWE-bench Verified 64.2%, BFCL V3 77.8%, GPQA 79.1%.
- Ranks 3rd overall / 2nd agentic; CC-Bench wins 40.4% vs Sonnet 4, 53.9% vs Kimi K2, 80.8% vs Qwen3-Coder.

**Key results:** GLM-4.5 (355B/32B-active MoE) scores 91.0% AIME 24, 70.1% TAU-Bench, and 64.2% SWE-bench Verified, ranking 3rd overall and 2nd on agentic benchmarks; both it and GLM-4.5-Air (106B/12B) sit on the SWE-bench-vs-parameters Pareto frontier.

*Evolution:* Building on the open MoE and reasoning-RL wave and Nemotron-CC-style data bucketing, GLM-4.5 unifies agentic, reasoning, and coding via expert-iteration + self-distillation plus hybrid thinking/non-thinking modes, motivating parameter-efficient open frontier models and standardized agent RL.

### Gemma 3 Technical Report
*2025 · report · `report_Gemma3_2503.19786.txt` · arXiv [2503.19786](https://arxiv.org/abs/2503.19786)*

Evaluated on LMSYS Chatbot Arena (blind Elo), MMLU/MMLU-Pro, AGIEval, MATH, GSM8K, HiddenMath, GPQA Diamond, MBPP/HumanEval (pass@1), LiveCodeBench, Bird-SQL, SimpleQA, FACTS Grounding, IFEval, BBH, BBEH, N2C, Global-MMLU-Lite, MGSM, WMT24++, FLoRes, XQuAD, ECLeKTic, IndicGenBench; vision: MMMU, DocVQA, InfoVQA, TextVQA, AI2D, ChartQA, VQAv2, MathVista; long-context: RULER and MRCR at 32K/128K; plus memorization-rate and CBRN safety audits. Gemma-3-27B-IT reaches Chatbot Arena Elo 1338 (rank ~9, above DeepSeek-V3 at 1318 and Gemma 2 at 1220), MATH 89.0, GSM8K 95.9, HumanEval 87.8, MMLU-Pro 67.5, IFEval 90.4, comparable to Gemini-1.5-Pro; Gemma3-4B-IT rivals Gemma2-27B-IT.

- Broad eval: Chatbot Arena Elo, MMLU/Pro, MATH, GSM8K, GPQA-D, HumanEval/MBPP pass@1, LiveCodeBench, IFEval, FACTS Grounding, BBH/BBEH, multilingual (MGSM, Global-MMLU-Lite, WMT24, XQuAD).
- Vision (MMMU, DocVQA, MathVista, ChartQA, AI2D) and long-context (RULER, MRCR at 32K/128K); plus memorization and CBRN safety audits.
- Gemma-3-27B-IT: Elo 1338 (rank ~9), MATH 89.0, GSM8K 95.9, HumanEval 87.8, MMLU-Pro 67.5, IFEval 90.4.

**Key results:** Gemma-3-27B-IT scores Chatbot Arena Elo 1338 (rank ~9), MATH 89.0, GSM8K 95.9, HumanEval 87.8, MMLU-Pro 67.5, and IFEval 90.4, comparable to Gemini-1.5-Pro and ahead of larger open models like DeepSeek-V3 (1318) and Gemma 2 (1220).

*Evolution:* Building on Gemma 2's distillation-based pre-training, Gemma 3 folds in the 2024 RL-alignment toolkit (BOND/WARM/WARP) and verifiable-reward signals from DeepSeek-R1 and Tulu 3, packaging distillation-plus-RL into lightweight multimodal open models.

### Kimi K1.5: Scaling Reinforcement Learning with LLMs
*2025 · report · `report_Kimi-K1.5_2501.12599.txt` · arXiv [2501.12599](https://arxiv.org/abs/2501.12599)*

Evaluation spans text (MMLU, IF-Eval, CLUEWSC, C-Eval), reasoning (HumanEval-Mul, LiveCodeBench v4/v5, Codeforces, AIME 2024, MATH-500), and vision (MMMU, MathVision, MathVista) using EM, Pass@1, and Codeforces percentile. Long-CoT reaches 77.5 AIME, 96.2 MATH-500, 94th percentile Codeforces, 74.9 MathVista, 70.0 MMMU, matching OpenAI o1. Short-CoT reaches 60.8 AIME, 94.6 MATH-500, 81.5 HumanEval-Mul, 47.3 LiveCodeBench, outperforming GPT-4o and Claude 3.5 Sonnet by up to +550%; k1.5-shortest hits 88.2 MATH-500. Long2short RL gives the best token efficiency: 60.8 AIME with ~3,272 tokens. Ablations show curriculum sampling beats uniform, the mirror-descent method beats ReST (negative gradients help), and a smaller model with longer CoT can match a larger one.

- Text/reasoning/vision suite: MMLU, IF-Eval, C-Eval, HumanEval-Mul, LiveCodeBench v4/v5, Codeforces, AIME 2024, MATH-500, MMMU, MathVision, MathVista.
- Metrics: EM, Pass@1, Codeforces percentile.
- Long-CoT 77.5 AIME / 96.2 MATH-500 / 94th pct Codeforces (≈o1); short-CoT 60.8 AIME, beating GPT-4o by up to +550%.

**Key results:** Kimi k1.5 long-CoT: 77.5 AIME 2024 Pass@1, 96.2 MATH-500 EM, 94th percentile Codeforces, 74.9 MathVista — matching OpenAI o1; short-CoT reaches 60.8 AIME, outperforming GPT-4o/Claude 3.5 Sonnet by up to +550%.

*Evolution:* Building on RLHF and o1's long-CoT RL, K1.5 formalizes RL as online mirror descent and argues context-length scaling suffices without MCTS, value functions, or process reward models, motivating long2short distillation for bounded test-time compute.
### Kimi K2: Open Agentic Intelligence (Technical Report of Kimi K2)
*2025 · report · `report_Kimi-K2_2507.20534.txt` · arXiv [2507.20534](https://arxiv.org/abs/2507.20534)*

Benchmarks (all non-thinking mode, Avg@k where high-variance) cover coding (LiveCodeBench v6 Pass@1, OJBench, MultiPL-E, SWE-bench Verified agentless+agentic single/multi-attempt, SWE-bench Multilingual, Multi-SWE-bench, SWE-Lancer, PaperBench, TerminalBench, Aider-Polyglot), tool use (tau2-Bench Avg@4, ACEBench), math/STEM/logic (AIME 2024/2025 Avg@64, MATH-500, HMMT, CNMO, PolyMath, ZebraLogic, AutoLogi, GPQA-Diamond Avg@8, SuperGPQA, Humanity's Last Exam), general (MMLU/Redux/Pro, IFEval, Multi-Challenge, SimpleQA, LiveBench), factuality (FACTS Grounding, HHEM, FaithJudge), long-context (MRCR, DROP, FRAMES, LongBench v2), and open-ended (Arena-Hard v2.0 win rate, LMSYS Arena). Baselines: DeepSeek-V3-0324, Qwen3-235B-A22B, Claude Sonnet/Opus 4, GPT-4.1, Gemini 2.5 Flash. Kimi-K2-Instruct scores SWE-bench Verified 65.8 (71.6 multi), SWE-bench Multilingual 47.3, tau2-Bench 66.1, ACEBench 76.5, LiveCodeBench v6 53.7, AIME 2025 49.5, GPQA-Diamond 75.1, IFEval 89.8, FACTS Grounding 88.5; top-1 open-source / 5th overall on LMSYS Arena (Jul 17 2025). Safety red-teaming via Promptfoo vs DeepSeek-V3/R1 and Qwen3.

- Coverage: coding, tool use, math/STEM/logic, general, factuality, long-context, open-ended; Avg@k for high-variance tasks.
- Kimi-K2-Instruct: SWE-bench Verified 65.8 (71.6 multi), tau2-Bench 66.1, ACEBench 76.5, LCB v6 53.7, AIME 2025 49.5, GPQA-D 75.1, IFEval 89.8, FACTS 88.5.
- #1 open-source / #5 overall on LMSYS Arena (Jul 17 2025, 3000+ votes).

**Key results:** Kimi-K2-Instruct (1.04T MoE, 32B activated, non-thinking) scores 65.8 SWE-bench Verified (71.6 multi-attempt), 66.1 tau2-Bench, 53.7 LiveCodeBench v6, 49.5 AIME 2025, 75.1 GPQA-Diamond, leading open-source and closing the gap with Claude 4 Opus/Sonnet; ranked #1 open-source / #5 overall on LMSYS Arena.

*Evolution:* Building on DeepSeek-V3's sparse-MoE+MLA and K1.5's RL scaling, K2 pushes token-efficient Muon training and agentic RL with synthetic tool-use environments to trillion-parameter scale, motivating stable Muon optimizers, scalable trajectory synthesis, and self-critic alignment.

### Llama-Nemotron: Efficient Reasoning Models
*2025 · report · `report_Llama-Nemotron_2505.00949.txt` · arXiv [2505.00949](https://arxiv.org/abs/2505.00949)*

Reasoning benchmarks are AIME24, AIME25 (I/II), GPQA-Diamond, LiveCodeBench (2408-2502 and 2410-2502), MATH500; non-reasoning are IFEval (Strict), BFCL V2 Live (function calling), Arena-Hard; out-of-distribution judging is measured on JudgeBench. Evaluations run at 32k context (beyond 16k/24k training length), temp 0.6 / top-p 0.95 for reasoning-on and greedy for off, up to 16 completions reporting avg pass@1; GPQA-D is also Avg@4. LN-Ultra reasoning-on scores GPQA-D 76.0 (DeepSeek-R1 71.5), AIME24 80.8 (79.8), AIME25 72.5 (70.0), MATH500 97.0, LiveCodeBench 66.3/68.1, IFEval 88.9, Arena-Hard 87.0; RL lifts GPQA-D from 66.4 (SFT) to 76.0. Smaller variants: LN-Super (49B) GPQA-D 66.7, AIME24 67.5; LN-Nano (8B) GPQA-D 54.1, AIME24 61.3. JudgeBench overall: LN-Ultra 79.14 (best open model, > DeepSeek-R1 73.14); Artificial Analysis ranks it the most intelligent open model (April 2025).

- Reasoning: AIME24/25, GPQA-D (avg pass@1 / Avg@4), LiveCodeBench (two splits), MATH500; non-reasoning: IFEval Strict, BFCL V2 Live, Arena-Hard; judging: JudgeBench.
- Eval at 32k context, temp 0.6/top-p 0.95 (reasoning-on) or greedy, up to 16 completions avg pass@1.
- LN-Ultra: GPQA-D 76.0, AIME24 80.8 (beats DeepSeek-R1 71.5/79.8); RL lifts GPQA-D 66.4→76.0; JudgeBench 79.14 (best open).

**Key results:** LN-Ultra (253B) reaches GPQA-Diamond 76.0% and AIME24 80.8%, beating DeepSeek-R1 (71.5% / 79.8%) while fitting one 8xH100 node with 1.71x lower latency than Llama-3.1-405B; large-scale GRPO raises GPQA-D from 66.4 (SFT) to 76.0.

*Evolution:* Building directly on DeepSeek-R1's distill-then-RLVR recipe, Llama-Nemotron marries NAS/FFN-Fusion efficiency and a user-facing reasoning toggle, motivating controllable efficient reasoning and combining teacher distillation with curriculum-driven RLVR to surpass the teacher.

### Qwen3 Technical Report
*2025 · report · `report_Qwen3_2505.09388.txt` · arXiv [2505.09388](https://arxiv.org/abs/2505.09388)*

Base models are evaluated on MMLU, MMLU-Pro, MMLU-redux, BBH, SuperGPQA, GPQA, GSM8K, MATH, EvalPlus (HumanEval/MBPP/+), MultiPL-E, CRUX-O, MGSM, MMMLU, INCLUDE. Post-trained models use MMLU-Redux, GPQA-Diamond (10x avg), C-Eval, LiveBench, IFEval (strict), Arena-Hard, AlignBench, Creative Writing v3, WritingBench, MATH-500, AIME'24/'25 (64x avg of 30 Qs), ZebraLogic, AutoLogi, BFCL v3, LiveCodeBench v5, CodeForces Elo, Multi-IF, MT-AIME2024, PolyMath, MLogiQA, RULER. Thinking mode uses temp 0.6, top-p 0.95, top-k 20; non-thinking temp 0.7, top-p 0.8. Qwen3-235B-A22B (Thinking) scores 85.7 AIME'24, 81.5 AIME'25, 70.7 LiveCodeBench v5, 2056 CodeForces rating, 70.8 BFCL v3, outperforming DeepSeek-R1 on 17/23 benchmarks with 60% activated params; its AIME'24 rose 70.1→85.1 over 170 RL steps. On-policy distillation beats RL on Qwen3-8B (74.4 vs 67.6 AIME'24) at ~1/10 the GPU hours (1,800 vs 17,920) and lifts pass@64.

- Base eval: MMLU/Pro/redux, BBH, SuperGPQA, GPQA, GSM8K, MATH, EvalPlus, MultiPL-E, CRUX-O, MGSM, MMMLU, INCLUDE.
- Post-trained: GPQA-D (10x avg), AIME'24/'25 (64x avg), LiveCodeBench v5, CodeForces Elo, BFCL v3, IFEval strict, Arena-Hard, AlignBench, WritingBench, RULER, in-house CounterFactQA/LengthCtrl/ThinkFollow/ToolUse.
- Qwen3-235B-A22B (Thinking): 85.7 AIME'24, 81.5 AIME'25, 70.7 LCB v5, 2056 CodeForces; beats DeepSeek-R1 on 17/23 benchmarks.

**Key results:** Qwen3-235B-A22B (Thinking) scores 85.7 AIME'24, 81.5 AIME'25, 70.7 LiveCodeBench v5, 2056 CodeForces, outperforming DeepSeek-R1 on 17/23 benchmarks with 60% activated params; on-policy distillation beats RL on Qwen3-8B (74.4 vs 67.6 AIME'24) at ~1/10 the GPU hours.

*Evolution:* Building on Qwen2.5 and the reasoning-model wave, Qwen3 unifies chat and reasoning with a controllable thinking budget; its Strong-to-Weak distillation and unified-mode design motivate efficient small-model reasoning and agent-RL focused on inference-time scaling.
### Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning
*2025 · report · `report_Seed1.5-Thinking_2504.13914.txt` · arXiv [2504.13914](https://arxiv.org/abs/2504.13914)*

Benchmarks are AIME 2024/2025 and BeyondAIME (math, avg@32), GPQA Diamond/SuperGPQA/MMLU-PRO (science, GPQA avg@8), Codeforces avg@8 and pass@8 over 12 recent contests, LiveCodeBench v5, Aider Polyglot, SWE-bench verified, ARC-AGI, SimpleQA, Collie, IFEval. Headlines: 86.7 AIME 2024 (matches o3-mini-high), 74.0 AIME 2025, 48.0 BeyondAIME, 77.3 GPQA, 55.0 Codeforces avg@8 (near Gemini 2.5 Pro, trails o3). Human evaluation vs DeepSeek R1 on non-reasoning scenarios uses a 0-4 ordinal scale plus binary win/loss, yielding +8.0% overall win rate. Verifier accuracy on a 456-sample human-annotated test set: Seed-Verifier 82.7%, Seed-Thinking-Verifier 99.3%. Ablations report AIME avg@32 (baseline 58% vs RFT 54%) and VAPO/DAPO rankings (Qwen-32B 50%/73%, Seed-150B-MoE 60%/79%).

- Math avg@32 (AIME 2024/2025, BeyondAIME); science GPQA avg@8 (GPQA-D, SuperGPQA, MMLU-PRO); Codeforces avg@8/pass@8 over 12 contests.
- Also LiveCodeBench v5, Aider Polyglot, SWE-bench verified, ARC-AGI, SimpleQA, Collie, IFEval.
- 86.7 AIME 2024 (≈o3-mini-high), 77.3 GPQA, 55.0 Codeforces avg@8; Seed-Thinking-Verifier 99.3% accuracy on 456-sample set.

**Key results:** Seed1.5-Thinking (20B-active/200B-total MoE) scores 86.7 on AIME 2024 (matching o3-mini-high), 55.0 on Codeforces avg@8, 77.3 on GPQA, with +8.0% win rate over DeepSeek R1 on non-reasoning tasks; Seed-Thinking-Verifier reaches 99.3% judgment accuracy.

*Evolution:* Building on the o1/R1 test-time-scaling wave and the authors' VAPO/DAPO work, it makes long-CoT RL stable at MoE scale and responds to the need for harder, less gameable reasoning benchmarks and reliable reward verification.

### AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning
*2025 · rl · `rl_AceReason-Nemotron_2505.16400.txt` · arXiv [2505.16400](https://arxiv.org/abs/2505.16400)*

Evaluation follows the DeepSeek-R1 protocol (temp 0.6, top-p 0.95, max 32,768 tokens), reporting pass@1 averaged over k generations (avg@k). Math benchmarks are AIME 2024/2025 (avg@64), MATH500 (avg@4), HMMT 2025 and BRUMO 2025 from MathArena (avg@64); code benchmarks are LiveCodeBench v5 (2024-08 to 2025-02) and v6 (2025-02 to 2025-05) avg@8, Codeforces ELO/percentile via LiveCodeBench Pro, and EvalPlus (avg@4). AceReason-Nemotron-14B scores 78.6/67.4 on AIME24/25, 61.1/54.9 on LCB v5/v6, Codeforces ELO 2024, EvalPlus 85.7; the 7B scores 69.0/53.6 on AIME24/25 and 51.8/44.1 on LCB v5/v6. Math-only RL yields +14.6%/+17.2% on AIME 2025 for 7B/14B and cross-domain +6.8%/+5.8% on LCB v5. RL also raises pass@k from k=8 to k=1024 by ~10% over the SFT baseline, unlocking previously unsolvable problems.

- Math avg@64 (AIME 24/25, HMMT 2025, BRUMO 2025), MATH500 avg@4; code avg@8 (LCB v5/v6), Codeforces ELO, EvalPlus avg@4.
- DeepSeek-R1 protocol: temp 0.6, top-p 0.95, max 32,768 tokens, pass@1 as avg@k.
- AceReason-Nemotron-14B: 78.6/67.4 AIME24/25, 61.1/54.9 LCB v5/v6, Codeforces ELO 2024; math-only RL +17.2% AIME 2025.

**Key results:** AceReason-Nemotron-14B reaches 78.6/67.4 avg@64 on AIME24/25 and 61.1/54.9 avg@8 on LiveCodeBench v5/v6, surpassing DeepSeek-R1-Distill-Qwen-32B; math-only RL gives +14.6%/+17.2% on AIME 2025 (7B/14B) and cross-domain +6.8%/+5.8% on LCB v5.

*Evolution:* Building on DeepSeek-R1's GRPO-plus-rule-verification recipe, it counters the claim that distillation beats RL for small/mid models by showing RL from strong distilled 7B/14B checkpoints can match or surpass distillation, motivating sequential multi-domain RL curricula.

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale
*2025 · rl · `rl_DAPO_2503.14476.txt` · arXiv [2503.14476](https://arxiv.org/abs/2503.14476)*

The primary benchmark is AIME 2024, evaluated with avg@32 (32 repeats, temperature 1.0, top-p 0.7); pass@32 and cons@32 are also tracked. DAPO lifts Qwen2.5-32B base from near 0% to 50 on AIME 2024, beating DeepSeek-R1-Zero-Qwen-32B (47) using 50% of the training steps. Progressive ablation on AIME24 avg@32 shows naive GRPO 30 → +Overlong Filtering 36 → +Clip-Higher 38 → +Soft Overlong Punishment 41 → +Token-level Loss 42 → +Dynamic Sampling (full DAPO) 50. Training-dynamics monitoring uses mean response length, reward score, generation entropy, and mean generation probability to detect instability, entropy collapse, and overfitting (training reward often uncorrelated with validation accuracy).

- Primary: AIME 2024 avg@32 (32 repeats, temp 1.0, top-p 0.7); pass@32 and cons@32 also tracked.
- DAPO: Qwen2.5-32B base 0%→50 on AIME 2024, beating R1-Zero-Qwen-32B (47) in 50% the steps.
- Ablation: naive GRPO 30 → full DAPO 50 across five techniques; dynamics via length/entropy/gen-prob.

**Key results:** DAPO trained on the Qwen2.5-32B base reaches 50 on AIME 2024 (avg@32), surpassing DeepSeek-R1-Zero-Qwen-32B's 47 using 50% of the training steps; full DAPO (50) vs naive GRPO (30) shows the four techniques add ~20 points.

*Evolution:* Building on GRPO and DeepSeek-R1's RL-from-base recipe, DAPO reacts to the reproducibility gap left by R1's withheld training details, open-sourcing the algorithm, verl-based code, and DAPO-Math-17K dataset to democratize large-scale RLVR.

### Understanding R1-Zero-Like Training: A Critical Perspective
*2025 · rl · `rl_DrGRPO_2503.20783.txt` · arXiv [2503.20783](https://arxiv.org/abs/2503.20783)*

Benchmarks are AIME 2024, AMC, MATH500, Minerva Math, and OlympiadBench, evaluated with greedy decoding and a 3k-token budget (8k for longer-context baselines). Metrics are accuracy (averaged over 5 benchmarks), pass@8 for exploration, response/token length for efficiency, training reward, plus GPT-4o-mini judging for answering-rate and self-reflection ('Aha moment') detection via keyword+LLM cross-validation. Oat-Zero-7B (Qwen2.5-Math-7B + Dr. GRPO) reaches 43.3% AIME 2024 and 51.4% average — a new SOTA for 7B R1-Zero-like training, beating SimpleRL-Zero-7B (26.7/46.6), PRIME-Zero-7B (16.7/48.0), OpenReasoner-Zero-7B@8k (13.3/45.9), and R1-Distill-Qwen-7B@3k (10.0/28.5), in 27h on 8xA100. Oat-Zero-1.5B averages 42.1; Oat-Zero-3B (Llama-3.2-3B-NuminaQA) averages 14.8. Dr. GRPO substantially shortens incorrect-response length versus vanilla GRPO with matched-or-better accuracy, confirmed significant across 3 random seeds.

- Benchmarks: AIME 2024, AMC, MATH500, Minerva Math, OlympiadBench; greedy decoding, 3k-token budget (8k baselines).
- Metrics: avg accuracy over 5, pass@8, response/token length, training reward, GPT-4o-mini judging for answering-rate and 'Aha moment'.
- Oat-Zero-7B: 43.3% AIME 2024, 51.4% avg (7B R1-Zero SOTA), in 27h on 8xA100; Dr. GRPO shortens incorrect responses vs vanilla GRPO.

**Key results:** Oat-Zero-7B (Qwen2.5-Math-7B + Dr. GRPO on MATH L3-5) achieves 43.3% on AIME 2024 and 51.4% average over 5 math benchmarks, a new SOTA for 7B R1-Zero-like training, in 27h on 8xA100; Dr. GRPO cuts incorrect-response length while matching or improving accuracy (significant over 3 seeds).

*Evolution:* Building on DeepSeek-R1-Zero's RL-without-SFT paradigm and GRPO, Dr. GRPO reacts critically to the 'Aha moment'/long-CoT narrative by showing pretraining biases and GRPO's length/difficulty biases inflate response length, prefiguring DAPO-style correctives.
### LIMO: Less is More for Reasoning
*2025 · rl · `rl_LIMO_2502.03387.txt` · arXiv [2502.03387](https://arxiv.org/abs/2502.03387)*

Evaluation uses pass@1 in zero-shot CoT: greedy single-sample decoding for large benchmarks (MATH500, OlympiadBench, Gaokao, Kaoyan, GradeSchool, MinervaMath, GPQA) and 4 samples (temperature 0.6) with unbiased pass@1 for small ones (<50 problems: AIME24, AMC23, CHMath). Rule-based scoring handles numerical answers; an LLM-based evaluator handles complex formats; the output cap is 32,768 tokens. The in-domain suite is AIME24, MATH500, AMC23; the OOD suite is OlympiadBench, CHMath, Gaokao, Kaoyan, GradeSchool (new), Minerva, GPQA. LIMO hits 63.3% AIME24, 95.6% MATH500, 96.3% AMC23, and 78.1% overall average, beating OpenAI-o1-preview (61.1 avg) and QwQ-32B-Preview (66.9) and far above NuminaMath-100k SFT (32.3) and OpenThoughts-114k SFT (58.3), using 800 vs 100k+ samples.

- Pass@1 zero-shot CoT: greedy for large benchmarks, 4 samples (temp 0.6) with unbiased pass@1 for <50-problem benchmarks.
- In-domain (AIME24, MATH500, AMC23) and OOD (OlympiadBench, CHMath, Gaokao, Kaoyan, GradeSchool, Minerva, GPQA); rule + LLM scoring, 32,768-token cap.
- LIMO: 63.3% AIME24, 95.6% MATH500, 96.3% AMC23, 78.1% avg, beating o1-preview (61.1) and QwQ-32B-Preview (66.9) with 800 samples.

**Key results:** LIMO (Qwen2.5-32B-Instruct + 800 curated SFT samples): 63.3% AIME24 (vs o1-preview 44.6%, QwQ-32B-Preview 50.0%, NuminaMath-100k SFT 6.5%) and 95.6% MATH500, with 78.1% average across 10 benchmarks and +45.8% absolute OOD gain, using ~1% of prior approaches' data.

*Evolution:* Extending LIMA's "less is more" lesson from instruction-following to mathematical reasoning, LIMO reacts against data-heavy SFT and complements RLVR by showing minimal high-quality SFT in knowledge-rich backbones can rival far larger pipelines.

### Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning
*2025 · rl · `rl_Logic-RL_2502.14768.txt` · arXiv [2502.14768](https://arxiv.org/abs/2502.14768)*

Evaluation combines in-distribution and transfer metrics. In-distribution is K&K validation accuracy across 2-8 person difficulty, benchmarked against o3-mini-high, o1-2024-12-17, DeepSeek-R1, GPT-4o/4o-mini, NuminaMath-7B-CoT, DeepSeek-Math-7B, and Qwen2.5-Base-7B. Super-OOD transfer uses AIME 2021-2024 and AMC 2022-2023 (correct-count metric). Training-time metrics include response length, mean reward, and KL loss. Memorization is measured via Accuracy on observed problems, Consistency Ratio under statement/reorder perturbations, and LiMem = Acc*(1−CR). Starting from Qwen2.5-7B-Instruct-1M (K&K avg 0.19), Logic-RL lifts K&K average to 0.89 (+0.70), with per-difficulty gains up to +0.85 at 6-person; AIME improves 125% and AMC 38% versus the base model. Thinking-token frequency (verify, re-evaluate) and language-mixing effects are also analyzed.

- In-distribution: K&K accuracy across 2-8 person difficulty, vs o3-mini/o1/R1/GPT-4o baselines.
- Super-OOD: AIME 2021-2024, AMC 2022-2023 (correct-count); training metrics: length, reward, KL.
- Memorization probes: Accuracy, Consistency Ratio under perturbation, LiMem=Acc*(1−CR).
- Logic-RL lifts K&K avg 0.19→0.89 (+0.70); AIME +125%, AMC +38% vs base; REINFORCE++ beats GRPO.

**Key results:** Qwen2.5-7B-Instruct-1M + Logic-RL, trained on ~5K K&K puzzles, lifts K&K average accuracy 0.19→0.89, AIME +125%, AMC +38% vs base; REINFORCE++ beats GRPO across nearly all metrics, and RL generalizes (low LiMem) whereas RFT memorizes.

*Evolution:* An early R1-replication study using a controlled synthetic logic corpus, Logic-RL shows R1-like emergent reasoning in a 7B model while finding no discrete aha moment, motivating long-to-short CoT compression, KL-relaxation, and language-consistency penalties in RLVR.

### Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
*2025 · rl · `rl_Open-Reasoner-Zero_2503.24290.txt` · arXiv [2503.24290](https://arxiv.org/abs/2503.24290)*

Benchmarks are AIME 2024, AIME 2025, MATH500, GPQA Diamond (accuracy averaged over 16 responses), plus generalization on MMLU and MMLU_PRO. ORZ-32B reaches AIME2024 48.1, AIME2025 36.0, MATH500 92.2, GPQA Diamond 55.5, beating DeepSeek-R1-Zero-Qwen-32B (47.0/—/91.6/55.0) on 1/10 the steps and beating DAPO-Qwen-32B* on MATH500 (92.2 vs 71.8) and GPQA (55.5 vs 16.0). Size scaling: ORZ-0.5/1.5/7/32B give AIME2024 1.0/3.5/17.9/48.1. ORZ-R1-Distill-Qwen-14B hits AIME2024 75.2, AIME2025 60.0, MATH500 95.6, GPQA 60.4, surpassing the larger R1-Distill-Qwen-32B. On MMLU/MMLU_PRO, ORZ-32B scores 84.9/74.4, exceeding Qwen2.5-32B-Instruct. Training metrics include train reward, response length, truncate rate, average repeat score, and reflection-pattern frequency.

- Benchmarks: AIME 2024/2025, MATH500, GPQA Diamond (avg over 16 responses); generalization MMLU/MMLU_PRO.
- ORZ-32B: AIME2024 48.1, MATH500 92.2, GPQA-D 55.5, beating R1-Zero-Qwen-32B on 1/10 steps and DAPO on MATH500/GPQA.
- Size scaling 0.5/1.5/7/32B → AIME2024 1.0/3.5/17.9/48.1; ORZ-R1-Distill-Qwen-14B AIME2024 75.2 beats the larger 32B distill.

**Key results:** ORZ-32B matches/exceeds DeepSeek-R1-Zero-Qwen-32B on AIME2024 (48.1 vs 47.0), MATH500 (92.2 vs 91.6), GPQA Diamond (55.5 vs 55.0) using 1/10 the steps; ORZ-R1-Distill-Qwen-14B reaches AIME2024 75.2, surpassing the larger R1-Distill-Qwen-32B (72.6).

*Evolution:* Building on R1-Zero's Reasoner-Zero paradigm, ORZ reacts against GRPO by reverting to vanilla PPO with a learned critic, and as the first fully open-source large-scale reasoning-RL framework (code, data, weights 0.5B–32B) it democratizes R1-Zero-style scaling.

### Process Reinforcement through Implicit Rewards
*2025 · rl · `rl_PRIME-ProcessReinforcement_2502.01456.txt` · arXiv [2502.01456](https://arxiv.org/abs/2502.01456)*

Seven competition-level reasoning benchmarks are AIME 2024, AMC, MATH-500, Minerva Math, OlympiadBench, LeetCode, and LiveCodeBench (v2); the metric is pass@1 accuracy (avg@16 also reported). Rule-based outcome verifiers use exact-match for math and test-case pass rate for code. Eurus-2-7B-PRIME reaches 26.7% on AIME 2024 (vs GPT-4o 9.3%, Llama-3.1-70B-Instruct 20.0%, Qwen2.5-Math-7B-Instruct 13.3%), +15.1% average over the SFT model across the seven benchmarks (+16.7% on math, >20% on AMC/AIME), surpassing Qwen2.5-Math-7B-Instruct on seven benchmarks using only 10% of its training data. PRIME yields 2.5× sample efficiency and +6.9% over outcome-only RLOO, is 11× faster than VinePPO, and adds +3.1 points over DeepSeek-R1-Distill-Qwen-1.5B under DeepScaleR settings.

- 7 benchmarks: AIME 2024, AMC, MATH-500, Minerva Math, OlympiadBench, LeetCode, LiveCodeBench v2; pass@1 (avg@16).
- Rule-based verifiers: exact-match (math), test-case pass rate (code).
- Eurus-2-7B-PRIME: 26.7% AIME 2024 (vs Qwen2.5-Math-7B-Ins 13.3%), +15.1% avg over SFT; 2.5× sample efficiency, +6.9% over RLOO, 11× faster than VinePPO.

**Key results:** Eurus-2-7B-PRIME (from Qwen2.5-Math-7B-Base) reaches 26.7% pass@1 on AIME 2024 vs Qwen2.5-Math-7B-Instruct's 13.3%, +15.1% average over the SFT model across 7 reasoning benchmarks, beating it on 7 benchmarks with 10% of its training data.

*Evolution:* Building on implicit reward modeling and DPO-style implicit rewards, PRIME reacts to R1/K1.5's claim that PRMs are impractical at scale by showing dense, online-updatable process rewards can be both scalable and beneficial, motivating later process-reward RL.
### Skywork Open Reasoner 1 Technical Report
*2025 · rl · `rl_Skywork-OR1_2505.22312.txt` · arXiv [2505.22312](https://arxiv.org/abs/2505.22312)*

Evaluated on AIME24, AIME25 (avg@32) and LiveCodeBench 2024-08 to 2025-02 (avg@4), with max generation 32,768 tokens, temperature 1, top-p 1; ablations also use avg@8 and pass@1. Skywork-OR1-32B scores 82.2/73.3/63.0, beating DeepSeek-R1 (79.8/65.9) and Qwen3-32B (81.4) on AIME. Skywork-OR1-7B scores 70.2/54.6/47.6 and Math-7B 69.8/52.3/43.6. Average across the three benchmarks rises 57.8%→72.8% (+15.0) for 32B and 43.6%→57.5% (+13.9) for 7B. The Section 4 baseline reaches 69.2 avg@8 AIME24, 53.3 AIME25, 50.5 pass@1 LiveCodeBench after 2700 steps on 32 H800s. Policy entropy and KL divergence are tracked as training diagnostics; entropy-collapse dynamics are a central measured phenomenon.

- Benchmarks: AIME24/25 (avg@32), LiveCodeBench 2024-08 to 2025-02 (avg@4); ablations avg@8/pass@1.
- Eval: max 32,768 tokens, temp 1, top-p 1; policy entropy and KL tracked as diagnostics.
- Skywork-OR1-32B: 82.2/73.3/63.0 (AIME24/25/LCB), beating DeepSeek-R1 (79.8/65.9) and Qwen3-32B; avg rises 57.8%→72.8% (+15.0).

**Key results:** Skywork-OR1-32B reaches 82.2 AIME24, 73.3 AIME25, 63.0 LiveCodeBench (avg@4), surpassing DeepSeek-R1 and Qwen3-32B on math; average over AIME24/25/LiveCodeBench improves 57.8%→72.8% (+15.0) for 32B and 43.6%→57.5% (+13.9) for 7B.

*Evolution:* Building on DeepSeek-R1's rule-reward RL and DeepScaleR's multi-stage context-length curriculum, Skywork-OR1 applies RL to already-distilled long-CoT models and systematically studies entropy collapse, motivating entropy-aware, on-policy RL recipes.

### s1: Simple test-time scaling
*2025 · rl · `rl_s1_2501.19393.txt` · arXiv [2501.19393](https://arxiv.org/abs/2501.19393)*

Three reasoning benchmarks are used: AIME24 (30 integer-answer competition problems), MATH500 (500 competition problems), and GPQA Diamond (198 PhD-level science questions). The primary metric is greedy (temperature 0) accuracy = pass@1, via lm-evaluation-harness and vLLM (full precision for final runs to reduce nondeterminism); for test-time scaling, Control, Scaling, and Performance variants are added. s1-32B with budget forcing scores 56.7% AIME24 (vs o1-preview 44.6%), 93.0% MATH500 (vs 85.5%), 59.6% GPQA (vs 73.3%); budget forcing extrapolates AIME24 from 50% (no BF) to 57% at 6x Wait injections. Data ablations show 1K-random/diverse/longest all lag s1K (e.g., 1K-random AIME24 36.7%); 59K-full (53.3% AIME24) is not clearly better. s1.1 (r1-distilled traces) reaches 66.7% AIME24, 94.4% MATH500, 62.6% GPQA, 56.7% AIME25.

- Benchmarks: AIME24 (30), MATH500 (500), GPQA Diamond (198); greedy (temp 0) pass@1 via lm-eval-harness + vLLM full precision.
- Test-time scaling variants: Control, Scaling, Performance; budget forcing (Wait injections).
- s1-32B: 56.7% AIME24 (vs o1-preview 44.6%), 93.0% MATH500, 59.6% GPQA; budget forcing lifts AIME24 50%→57%.

**Key results:** s1-32B, SFT on only 1,000 traces for 26 min on 16 H100s, hits 56.7% AIME24 with budget forcing vs o1-preview's 44.6%, and 93.0% MATH500 vs 85.5%; budget forcing extrapolates AIME24 from 50% to 57%, making it the most sample-efficient open-data reasoning model.

*Evolution:* Building on LIMA's 1K-examples thesis and Snell et al.'s test-time-compute scaling, s1 reacts against opaque large-scale-RL recipes by showing 1K distilled SFT plus a simple decoding trick matches o1-preview, motivating later sample-efficient reasoning work such as LIMO.

## 2026

### CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation
*2026 · method · `method_CUDA-Agent_2602.24286.txt` · arXiv [2602.24286](https://arxiv.org/abs/2602.24286)*

Evaluation runs on KernelBench Levels 1-3 (250 tasks: L1=100, L2=100, L3=50), adapted to a multi-file dev environment, with three metrics: Pass Rate (compiles + correctness), Faster Rate (% correct kernels beating Eager/Compile baselines), and Speed-up (geometric-mean ratio over correct solutions). All baselines (Claude Opus 4.5, Gemini 3 Pro, GLM 4.6, Kimi K2, Seed1.6 base) sit under the same agent loop; ChatGPT-5 declined CUDA prompts. Ablations isolate the agent loop, robust reward, RFT, and value pretraining.

- Headline: 98.8% pass, 96.8% faster-vs-compile, 2.11x geomean vs compile overall; 100%/100%/92% faster-vs-compile on L1/L2/L3.
- Beats Claude Opus 4.5 and Gemini 3 Pro by ~40% on the hardest L3 split.
- Case studies reach up to 73.31x (diagonal matmul) vs torch.compile.

**Key results:** CUDA Agent (Seed1.6) on KernelBench: 98.8% pass rate, 96.8% faster rate vs torch.compile, 2.11x geomean speedup vs compile overall; 100%/100%/92% faster rate on Level-1/2/3. It outperforms Claude Opus 4.5 and Gemini 3 Pro by ~40% on the hardest Level-3 split. Multi-stage warm-up enables stable PPO training for 150 steps versus collapse at 17 steps without it.

*Evolution:* CUDA Agent extends the agentic-RL-for-coding line (Kevin's multi-turn RL) by combining DAPO-style asymmetric PPO clipping, ReAct/OpenHands agent loops, and Anthropic's Agent Skills paradigm, while explicitly reacting to training-free CUDA systems (STARK, CudaForge) and data-leakage-prone fine-tuners (CUDA-L1, Kevin). In the 2026 context, it shows that large-scale execution-rewarded agentic RL can make LLMs match or beat compiler-based optimization (torch.compile), motivating automation of performance-critical systems software.

### Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic
*2026 · method · `method_GRPO-Ustatistic-theory_2603.01162.txt` · arXiv [2603.01162](https://arxiv.org/abs/2603.01162)*

Two experiments validate the theory. (1) Gradient MSE on a synthetic 500-question arithmetic set with Qwen2.5-0.5B Base/Instruct/ICL and G in {4,8,16,32,64} shows vanilla REINFORCE largest, GRPO-type strictly smaller, near-oracle at G≥8, indistinguishable at G=32,64; MSE also drops with stronger models. (2) Optimal group size: GSM8K with Qwen2.5-1.5B-Instruct (N=BG=1024, G in {4,...,128}, 5 runs) peaks at G*=32 across checkpoints except n=200; MATH with Qwen2.5-Math-7B (N in {1024,2048,4096}) is ~0.77 accuracy with best G mostly 64, rising to 128 at the largest budget.

- Metrics: gradient MSE (95% CIs) and test accuracy (% correctly solved).
- G* is universal across iterations on GSM8K; larger models favor larger G.
- Analytic formula G*=sqrt(c3/c1) gives the budget-independent scaling law.

**Key results:** GRPO's gradient MSE is strictly below vanilla REINFORCE and matches the oracle (true-critic) estimator for G>=8 (Qwen2.5-0.5B, Figure 4). The optimal per-prompt group size is universal: G*=32 on GSM8K (Qwen2.5-1.5B, N=1024) across iterations and about 64 on MATH (Qwen2.5-Math-7B), given analytically by G*=sqrt(c3/c1).

*Evolution:* Builds on DeepSeekMath/DeepSeek-R1's GRPO (2024-2025) and classical U-statistics (Hoeffding 1948), reacting to the theoretical gap left by RLVR works such as Davis & Recht and Pang & Jin. In the 2026 GRPO-variant boom, it supplies the first unified finite-sample and asymptotic theory plus a budget-independent group-size scaling law, enabling principled hyperparameter selection for the flood of critic-free RLVR algorithms.

### GRPO-VPS: Enhancing Group Relative Policy Optimization with Verifiable Process Supervision for Effective Reasoning
*2026 · method · `method_GRPO-VPS_2604.20659.txt` · arXiv [2604.20659](https://arxiv.org/abs/2604.20659)*

Primary metrics are Pass@1 averaged over 4 runs (temp 1.0, top-p 1.0, max 3072 tokens) and average response token length (AvgToken), with Math-Verify for answer equivalence. Math benchmarks: MATH, AIME 2024, AMC23, OlympiadBench; general: GPQA, MMLU-Pro, TheoremQA, WebInstruct test. Progress-signal quality is measured against PRM800K human step labels (F1 > 0.75 across 0.5B-32B, up to 0.803 for Qwen2.5-32B). Baselines include GRPO, DrGRPO, GSPO, S-GRPO, Eurus-2-7B-PRIME, and GRPO+Skywork-o1-prm.

- Qwen2.5-Math-1.5B: +2.6 Pass@1 over GRPO while cutting length 11.0-13.7%.
- Qwen2.5-Math-7B: +1.1 point, up to 34.0% length reduction (AIME24 31.7% vs Eurus 15.0%).
- Qwen3-1.7B: +1.8 MMLU-Pro (54.4%), +2.4 TheoremQA (37.3%), ~50% shorter generations.

**Key results:** On Qwen2.5-Math-1.5B, GRPO-VPS improves overall Pass@1 by up to +2.6 points over GRPO while cutting reasoning length by 11.0-13.7%; on Qwen2.5-Math-7B it adds +1.1 point with up to 34.0% length reduction (e.g. AIME24 31.7% vs Eurus 15.0%). On general reasoning with Qwen3-1.7B, gains reach +1.8 on MMLU-Pro (54.4%) and +2.4 on TheoremQA (37.3%), with ~50% shorter generations than the base model.

*Evolution:* GRPO-VPS extends the RLVR/GRPO line (DeepSeekMath, DAPO) by attacking GRPO's indiscriminate credit assignment without resorting to PRMs (PRM800K) or Monte-Carlo-rollout methods like S-GRPO and MRT, instead reading the model's own conditional answer probability as a free dense signal. Published at ICLR 2026, it reflects the maturing trend toward verifier-free, confidence-driven dense rewards as a scalable alternative to learned reward models, and motivates further work on belief-probing supervision for efficient long-chain reasoning.

### Computer Environments Elicit General Agentic Intelligence in LLMs
*2026 · method · `method_LLM-in-Sandbox_2601.16206.txt` · arXiv [2601.16206](https://arxiv.org/abs/2601.16206)*

Seven benchmarks span six non-code domains plus software engineering: AIME25 (30×16, Math-Verify), UGPhysics (650, LLM judge), ChemBench (450, exact match), MedXpertQA (500, exact match), AA-LCR (100×4, LLM equality checker), IFBench (300, rule-based loose), and SWE-bench Verified (500, rule-based). The headline metric is Delta = LLM-in-Sandbox − LLM. Training-free, strong models gain up to +15.5% (Qwen3-Coder-30B-A3B, AIME25 26.0→41.5) and +14.4% (MiniMax-M2, instruction following), while weak Qwen3-4B regresses.

- Long-context tokens drop up to 8x (Qwen 100K→13K); aggregate token ratio 0.5-0.8x; QPM ratio 0.6-2.2x (MiniMax 2.2x).
- After RL, weak Qwen3-4B flips to outperforming LLM mode (Biomed 14.4 vs 10.0; IF 37.7 vs 33.7) and gains in pure LLM mode.
- Average turns fall 23.7→7.0 post-RL at a ~1.1 GB shared image footprint.

**Key results:** Training-free: Qwen3-Coder-30B-A3B on AIME25 math +15.5% (26.0->41.5); MiniMax-M2 instruction following +14.4%; long-context tokens cut up to 8x (Qwen 100K->13K). After LLM-in-Sandbox-RL, weak Qwen3-4B-Instruct-2507 outperforms LLM mode in sandbox (Biomed 14.4 vs 10.0; Instruction Following 37.7 vs 33.7) and also improves in pure LLM mode. Aggregate token ratio 0.5-0.8x and QPM up to 2.2x (MiniMax) at a ~1.1 GB shared image footprint.

*Evolution:* Building on ReAct-style tool use, code-sandbox coding agents (SWE-agent, Claude Code), SWE-RL (DeepSWE, R2E-Gym), and Instruction Pre-Training data, this 2026 work isolates the computer environment itself as a source of general intelligence and shows general-data sandbox RL transfers to non-code domains and even text-only inference. It reacts to the surge of computer-equipped systems (Manus, Claude Code, OpenClaw) and motivates future 'computer-native' models where environment interaction becomes a first-class pretraining and RL objective for generalist agents.

### Latent Thought Flow: Efficient Latent Reasoning in Large Language Models
*2026 · method · `method_LTF-LatentThoughtFlow_2606.16222.txt` · arXiv [2606.16222](https://arxiv.org/abs/2606.16222)*

Metrics are Accuracy (%) and Reasoning Length (#L = reasoning steps/tokens), averaged over 5 seeds (95% CIs in appendix). Backbones: LLaMA-3.2-1B/3B, LLaMA-3.1-8B, DeepSeek-R1-Distill-Qwen-1.5B. In-domain: GSM8K-Aug, ASDiv-Aug, DU; OOD: GSM-Hard, SVAMP, MultiArith, AQUA-RAT, MATH. Baselines include explicit CoT, Assist-CoT, SoftCoT++, and latent iCoT, CODI, Coconut, CoLaR, ReGuLaR, plus GRPO/Detailed-Balance/Trajectory-Balance ablations. Training: LoRA r=128, AdamW lr=1e-4, batch 256, 100 epochs, rollout_n=20, T_max=32.

- LTF beats ReGuLaR by +12.9% avg accuracy while cutting length 34.5% (LLaMA-8B GSM8K-Aug 53.14% vs 50.14% at #L 3.37 vs 3.93).
- Transfer: +6.0% accuracy, -19.9% length over ReGuLaR.
- Extreme compression (#L=1): 8.10% vs 6.62% on MATH; 39.43% vs 37.28% on AQUA-RAT.

**Key results:** Finetuning: LTF beats strongest latent baseline ReGuLaR by +12.9% avg accuracy while cutting reasoning length by 34.5% (e.g., LLaMA-8B on GSM8K-Aug 53.14% vs 50.14% at #L 3.37 vs 3.93). Transfer: +6.0% accuracy and -19.9% length over ReGuLaR. Extreme compression (#L=1): LTF 8.10% vs 6.62% on MATH; 39.43% vs 37.28% on AQUA-RAT.

*Evolution:* LTF builds on continuous GFlowNet theory (Lahlou 2023) and the Coconut/CoLaR/ReGuLaR line of continuous-latent reasoning, reacting to the posterior collapse of RL-style latent reasoning (GRPO/LEPO) and the verbosity of maximum-likelihood CoT compression. By 2026 it reframes latent reasoning as reward-proportional distribution matching over accuracy and cost, motivating adaptive-budget test-time scaling and future multimodal latent thought flows.

### LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification
*2026 · method · `method_LaTER_2605.07315.txt` · arXiv [2605.07315](https://arxiv.org/abs/2605.07315)*

Evaluation reports accuracy and total token usage (latent steps + emitted explicit tokens) on AIME 2025 (MathArena), MATH-500, GSM8K, GPQA, ARC-Challenge, HumanEval+, and MBPP+. Backbones: Qwen3-14B (main), DeepSeek-R1-Distill-Llama-8B, OLMo3-32B-Think, with Qwen3 decoding (temp 0.6, top-p 0.95, top-k 20, max 38192 tokens). CoT-SFT (same data, no latent) isolates the latent architecture's contribution.

- Training-free LaTER on Qwen3-14B: AIME 2025 73.3% (+3.3) at 10,661 tokens (-32%); MATH-500 97.2% (+3.8) at 2,887 (-17%).
- Trained LaTER (Qwen3-14B): AIME 2025 80.0% (+10.0) at 10,575 (-33%); MATH-500 96.4%, ARC 96.7%, HumanEval+ 92.2%.
- CoT-SFT reaches AIME 73.3% but at 12,687 tokens, showing the latent-first design beats mere trace shortening.

**Key results:** Trained LaTER on Qwen3-14B reaches 80.0% on AIME 2025 (+10.0 over the CoT baseline's 70.0%) while using 33% fewer tokens (10,575 vs 15,730), with best/lowest-token results across 7 benchmarks (MATH-500 96.4%, HumanEval+ 92.2%). Training-free LaTER on Qwen3-14B lifts AIME 2025 70.0%->73.3% and MATH-500 93.4%->97.2% with 32%/17% token cuts. CoT-SFT matches AIME 73.3% but at 12,687 tokens, showing the latent-first design beats mere trace shortening.

*Evolution:* LaTER builds on the 2025-26 latent-reasoning line (Coconut, SoftCoT, Latent-SFT, CoLaR) and training-free variants (Soft Thinking, SeLaR, LatentMAS), reacting against fully replacing explicit CoT-which degrades symbolic tasks like MATH-500/AIME-by splitting labor: latent exploration for search, explicit CoT for verification. In 2026 it motivates learned instance-adaptive halting policies and extension of latent-then-explicit reasoning to longer open-ended and multimodal settings.

### Mid-Think: Training-Free Intermediate-Budget Reasoning via Token-Level Triggers
*2026 · method · `method_Mid-Think_2601.07036.txt` · arXiv [2601.07036](https://arxiv.org/abs/2601.07036)*

Benchmarks: MATH500, AIME22-24, GPQA, LiveCodeBench. Metrics: accuracy, average generation length, wait count, training entropy, and training time. Models span hybrid-think (Qwen3-4B/8B/14B/32B, Qwen3-30B-A3B MoE), pure-think (DeepSeek-R1-Distill-Qwen-7B/-Llama-8B, Phi-4-mini-reasoning), and DASD-4B-Thinking. A budget-controlled baseline truncates Think traces to 0.1-1.0 of tokens to build an accuracy-length Pareto; Mid-Think sits near a 0.5 budget and surpasses the frontier on GPQA. RL uses GRPO via verl on 8xH200.

- Qwen3-8B Mid-Think RL (Think-test): AIME 69.8→72.4%, GPQA 58.5→61.1%, MATH500 92.4→94.1%, training time 54h→46h.
- Training-free Qwen3-8B MATH500: No-Think 83.2%/899 tok, Mid-Think 92.3%/2589, Think 94.6%/4904.
- LiveCodeBench Qwen3-8B: No-T 40.8% / Mid-T 49.7% / Think 65.9%.

**Key results:** Qwen3-8B trained with Mid-Think as the GRPO objective improves Think-test AIME 69.8%->72.4% and GPQA 58.5%->61.1% while cutting RL training time ~15% (54h->46h). Training-free Mid-Think on MATH500 (Qwen3-8B) reaches 92.3% accuracy at 2,589 tokens versus 94.6%/4,904 for Think and 83.2%/899 for No-Think, and on LiveCodeBench Qwen3-8B improves No-Think 40.8%->49.7% (Mid-Think).

*Evolution:* Builds on the attention-sink / token-cue line (No-Wait, SpecExit, Speculative Thinking) and on hybrid-thinking models (Qwen3, Gemini, GPT-oss, DeepSeek V3.1) by reframing Think/No-Think switching as overfitting to a few trigger tokens inherited from SFT data templates. In the 2026 RLVR/GRPO era it offers a lightweight, training-free route to intermediate reasoning budgets and motivates template-aware budget control for cheaper, higher-entropy RL training.

### Polar: Agentic RL on Any Harness at Scale
*2026 · method · `method_Polar-AgenticRL_2605.24220.txt` · arXiv [2605.24220](https://arxiv.org/abs/2605.24220)*

Primary benchmark is SWE-Bench Verified scored as pass@1 via swebench_harness on a fresh runtime, plus per-step outcome reward (rollout pass@1) on SWE-Gym GRPO training curves. From the same Qwen3.5-4B base, Polar+GRPO gains on SWE-Bench Verified: Codex 3.8%→26.4% (+22.6), Claude Code 29.8%→34.6% (+4.8), Qwen Code 34.6%→35.2% (+0.6), Pi 34.2%→40.4% (+6.2). A trajectory-builder ablation under identical model/hardware/topology quantifies the prefix_merging speedup.

- Training-reward windows improve: Codex 9.5%→54.5%, Claude Code 28.8%→67.0%, Qwen Code 61.6%→66.0%, Pi 61.6%→76.2%.
- prefix_merging cuts 1,185 request-level updates to 218 and wall-clock 189.5→35.2 min (5.39x), with 87.7% rollout GPU utilization vs 20.4%.
- Offline generation: 504/1,638 accepted (30.8%) at ~64 GPU-hours.

**Key results:** Qwen3.5-4B + standard GRPO via Polar improves SWE-Bench Verified pass@1 by +22.6 on Codex (3.8% to 26.4%), +4.8 on Claude Code, +0.6 on Qwen Code, and +6.2 on Pi. Prefix_merging cuts a 3-step run from 189.5 to 35.2 min (5.39x) with 87.7% rollout utilization vs 20.4%. Offline generation produced 504 accepted SFT trajectories from 1,638 SWE-Gym attempts (30.8%) at ~64 GPU-hours.

*Evolution:* Polar builds on ProRL Agent's rollout-as-a-service and reacts to low-intrusion instrumentation (Agent Lightning, rLLM) and full-stack trainers (SkyRL-Agent, PRIME-RL) by moving the integration boundary from the agent SDK to the model API endpoint, enabling training of closed-source/binary harnesses unchanged. In the 2026 agentic-RL landscape it positions rollout infra as trainer-agnostic, motivating future work on session normalization and process reward models for finer credit assignment.

### R³: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning
*2026 · method · `method_R3-ReplayReflectionRanking_2601.19620.txt` · arXiv [2601.19620](https://arxiv.org/abs/2601.19620)*

Five math benchmarks: AIME 2024, MATH500, AMC 2023, Minerva Math, OlympiadBench, with N=16 responses per question and metrics Pass@1, Pass@16, and average response length in tokens. R³-1.5B averages 60.59 (vs base 47.81, +12.78; vs DeepScaleR-1.5B-Preview 56.24). R³-7B averages 67.18 with AIME24 61.04 (beating Thinker-7B 60.00). Ablations remove CCR, ISR, and SERR components.

- R³-1.5B per-task: AIME24 47.50, MATH500 89.27, AMC 77.33, Minerva 34.21, Olympiad 54.64.
- Token efficiency: AIME24 47.5 at 7,574 tokens vs base 28.1 at 12,270 tokens.
- Removing CCR drops AIME24 47.50→35.62; removing ISR drops Olympiad 54.64→51.72; Pass@256 on challenging MATH subset hits 64.29.

**Key results:** R³-1.5B reaches 60.59 average across five math benchmarks (+12.78 over DeepSeek-R1-Distill-Qwen-1.5B), with AIME24 47.50 using only 7,574 tokens versus the base's 28.1 at 12,270 tokens. R³-7B scores 67.18 average (AIME24 61.04, beating Thinker-7B 60.00; Minerva 39.70), and Pass@256 on the challenging MATH subset hits 64.29.

*Evolution:* R³ builds on the GRPO/DAPO and DeepScaleR line of small-model reasoning RL and on experience-replay work (RLEP, EFrame, LUFFY), while drawing on entropy-as-exploration analyses of 'forking tokens'. It reacts specifically to GRPO's intra-group advantage collapse on hard tasks. Going into 2026 it motivates unsupervised, process-level reward signals for truncated trajectories and more sample-efficient replay-augmented RL for compact reasoning models.

### ReSyn: Autonomously Scaling Synthetic Environments for Reasoning Models
*2026 · method · `method_ReSyn_2602.20117.txt` · arXiv [2602.20117](https://arxiv.org/abs/2602.20117)*

Benchmarks: Big-Bench Hard (BBH), Big-Bench Extra Hard (BBEH), GSM8K-test, and AIME 2024, all zero-shot at temp 0.8, top-p 0.95. Primary metric is mean@4 (mean@128 for AIME); BBH also reported at 0/1/3-shot. BBEH subtask gains use paired bootstrap tests (alpha=0.01); dataset diversity uses LLM descriptors, sentence-transformer embeddings, hierarchical clustering, and Shannon semantic entropy. ReSyn-7B (DAPO, 400 steps) is compared against Instruct, SynLogic, and SynLogic-7B.

- ReSyn-7B: BBH 75.2 (+14% rel. vs 65.9), BBEH 14.3 (+27% rel. vs 11.2), GSM8K 91.4 (vs 82.3), AIME 2024 14.0 (+40% rel. vs 9.8).
- Beats SynLogic-7B on BBH (75.2 vs 66.5) and BBEH (14.3 vs 8.0).
- Ablations: Verifier-RL 75.24/14.61 vs Code-RL 74.94/14.24 vs Answer-RL 68.83/14.33 (BBH/BBEH); task-diversity scaling N=400 gives 75.19 BBH vs 69.85 (N=100).

**Key results:** Qwen2.5-7B-Instruct + ReSyn (DAPO, 400 steps): BBH 75.2 (+14% rel. vs 65.9), BBEH 14.3 (+27% rel. vs 11.2), AIME 2024 14.0 (+40% rel. vs 9.8), GSM8K 91.4 (vs 82.3). It also beats SynLogic-7B on BBH (75.2 vs 66.5) and BBEH (14.3 vs 8.0), with verifier-based supervision and task diversity both shown as significant drivers via ablation.

*Evolution:* ReSyn builds on the RLVR wave (DeepSeek-R1, OpenAI o1) and procedural reasoning environments (SynLogic, TinyZero, Reasoning Gym), but automates environment authoring via LLM-synthesized code verifiers, scaling task diversity >10x beyond SynLogic's 35 handcrafted tasks and exploiting the generator-verifier gap. In the 2026 landscape it points toward labor-free, self-scaling reasoning curricula for continued RL post-training without manual task engineering.

### Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability
*2026 · method · `method_SOAR_2601.18778.txt` · arXiv [2601.18778](https://arxiv.org/abs/2601.18778)*

Primary metric is pass@k accuracy on the held-out fail@128 test set for k in {1,4,8,16,32}, using 32 samples/problem, mean±SD over 6-12 nested teacher/student seeds (>600 runs). Diversity is measured via the Vendi Score (Qwen3-8B embeddings). Baselines include Hard-Only (with 4x compute at g=128) and full-MATH training; ablations extend to Llama-3.1-8B-Instruct.

- SOAR-PQ: ~4x pass@1 and ~2x pass@32 on MATH, ~2x pass@1 and ~1.5x pass@32 on HARP over Hard-Only.
- Absolute gains: PQ +9.3% pass@32 on MATH (18.9±5.3% vs 9.6±2.6%), +4.2% on HARP (12.3±2.0% vs 8.2±1.0%); PS +8.5%/+3.6%.
- PQ-MATH recovers 75% of full-MATH gain; transfers to held-out OlympiadBench; Grounded-T preserves diversity (VS ~32-35) while Intrinsic-T collapses (VS=10.8).

**Key results:** On fail@128 MATH with Llama-3.2-3B-Instruct, SOAR-PQ reaches 18.9+/-5.3% pass@32 vs 9.6+/-2.6% Hard-Only (~2x / +9.3%) and ~4x pass@1 (1.7 vs 0.5%). On HARP, PQ gives 12.3+/-2.0% pass@32 vs 8.2+/-1.0% (+4.2%). PQ-MATH recovers 75% of the full-curated-MATH upper bound's gain and transfers to held-out OlympiadBench.

*Evolution:* SOAR builds on asymmetric self-play (AlphaZero, Alice-Bob) and recent data-free LLM self-play (Absolute Zero, SeRL, R-Zero), reacting against their reliance on intrinsic/proxy rewards that cause diversity collapse and instability. By grounding the teacher in measured student progress via a tractable RLOO bilevel loop, it offers a 2026-era path to escape RLVR plateaus on problems a model cannot yet solve, motivating future work on cheaper reward proxies and scaling beyond 3-8B models.

### SWE-Fuse: Empowering Software Agents via Issue-free Trajectory Learning and Entropy-aware RLVR Training
*2026 · method · `method_SWE-Fuse_2603.07927.txt` · arXiv [2603.07927](https://arxiv.org/abs/2603.07927)*

Evaluation uses SWE-bench Verified with a single metric: issue resolve rate (% issues whose patches pass all tests). RQ1 uses the full 500-instance benchmark; RQ2/RQ3 use a stratified 200-task subset. Compared against open-source 8B/32B models, OpenAI-o3, and frontier closed models (Claude-4.5-Sonnet, GPT-5, Gemini 3 Pro) plus 480B-1T open models (Qwen3-Coder-480B, Kimi-K2). RLVR training reward curves and a case study (astropy-13236) provide qualitative support.

- SWE-Fuse-Qwen3-32B: 60.2% (65.2% with TTS@8), +1.8% over OpenAI-o3 (58.4%), SOTA among open-source ≤32B.
- SWE-Fuse-Qwen3-8B: 43.0% (49.8% with TTS@8), best 8B result.
- Data scaling 0→14k trajectories improves resolve rate 13.5%→39.0% (2.9x).

**Key results:** SWE-Fuse-Qwen3-32B resolves 60.2% of SWE-bench Verified issues (65.2% at TTS@8), SOTA among open-source <=32B models and +1.8% over OpenAI-o3. SWE-Fuse-Qwen3-8B reaches 43.0% (49.8% at TTS@8), the best 8B result. Data scaling from 0 to 14k trajectories improves resolve rate 13.5% to 39.0% (2.9x).

*Evolution:* SWE-Fuse (2026) extends the RLVR-for-coding line (SWE-RL, DeepSWE, SWE-Gym, R2E-Gym) by attacking real-world data noise — issue/solution misalignment — through issue-free trajectory learning, and grafts entropy-adaptive clipping (building on entropy-in-RL studies like Cui 2025, Cheng 2025) onto RLOO. It shows lightweight 8B/32B agents can rival 100B+-1T models, motivating further work on data quality and adaptive trust regions for agentic RL.

### Synthetic Sandbox for Training Machine Learning Engineering Agents
*2026 · method · `method_SandMLE_2604.04872.txt` · arXiv [2604.04872](https://arxiv.org/abs/2604.04872)*

Benchmarks: MLE-Bench-Lite (22 unseen Easy-split Kaggle tasks) and MLE-Dojo (62 broader Kaggle tasks). Metrics follow MLE-bench's Kaggle hierarchy: Valid Submission, Above Median, Bronze, Silver, Gold, Any Medal (primary), plus MLE-Dojo's HumanRank Score (relative leaderboard position vs. humans). Evaluated across scaffolds ReAct, AIRA, AIDE, MLE-Agent; models Qwen3-8B/14B/30B-A3B-2507 with references to Claude-4.5-Sonnet, DeepSeek-V3.1, Gemini-2.5-Flash.

- SandMLE Any Medal gains: +66.9% (8B), +24.7% (14B), +100.7% (30B) over Base; 20.3-66.9% over Seed-SFT.
- 8B matches DeepSeek-V3.1/Gemini-2.5-Flash at 22.7% Any Medal; 14B/30B reach 27.3% vs Claude's 31.8%.
- MLE-Dojo: up to 32.4% relative HumanRank gain; 30B+MLE-Agent hits 83.9% Valid Submission, 38.56 HumanRank.
- Generalizes across unseen scaffolds; per-rollout execution time cut 13.7x (196.17s→14.31s).

**Key results:** SandMLE cuts per-rollout execution time 13.7x (196.17s -> 14.31s). On MLE-Bench-Lite, Qwen3-8B/14B/30B-SandMLE achieve 22.7%/22.7%/27.3% Any Medal (+66.9%/+24.7%/+100.7% vs Base; 20.3-66.9% over Seed-SFT). On MLE-Dojo, Qwen3-30B-SandMLE reaches 38.56 HumanRank Score with 83.9% Valid Submission.

*Evolution:* Building on trajectory-wise RL successes in SWE (DeepSWE) and web search (WebDancer) and on GRPO-style RLVR, SandMLE (2026) addresses the MLE-specific execution-latency bottleneck that had forced prior MLE-agent work back to SFT or off-policy async RL. By showing synthetic micro-scale environments can proxy real MLE tasks for on-policy RL, it enables scalable, framework-agnostic RL training of MLE agents and motivates further work in synthetic-environment generation and dense-reward design for long-horizon agentic post-training.

### Self-Verified Distillation: Your Language Model Is Secretly Its Own Synthetic Data Pipeline
*2026 · method · `method_SelfVerifiedDistillation_2605.26132.txt` · arXiv [2605.26132](https://arxiv.org/abs/2605.26132)*

Metric is pass@1 averaged over random seeds, with validation benchmarks for checkpoint selection and held-out test benchmarks reported. Math: validation MATH500, OlympiadBench-Math, AIME24 (×10), AIME25 (×10); test AIME26 (×10), HMMT 02/25 (×10). Science: validation JEEBench (×3), OlympiadBench-Physics; test GPQA Diamond (×3), HLE (×3). Coding: validation LiveCodeBench v2 (×6); test LCBv5 Official (×6), LCBv6 Official (×6). Compared against the UQ-TTC test-time-compute baseline (same n=8, v=5 verification, up to 168 inference calls per problem). Gains transfer across Qwen3-0.6B/4B/8B.

- Qwen3-4B aggregate held-out pass@1: +16.7 math (AIME26+HMMT), +11.1 science (GPQA Diamond+HLE), +8.3 coding (LCBv5+LCBv6).
- Per-benchmark: AIME26 59.3→69.3 (+10.0), HMMT 39.3→46.0 (+6.7), GPQA Diamond 50.8→60.4 (+9.6).
- Beats UQ-TTC on 5 of 6 comparisons using a single inference call (vs up to 168); weakest/least consistent at 0.6B (HLE regresses).

**Key results:** Qwen3-4B aggregate held-out pass@1 gains: +16.7 math (AIME26+HMMT), +11.1 science (GPQA Diamond+HLE), +8.3 coding (LCBv5+LCBv6); e.g. AIME26 59.3->69.3 (+10.0), HMMT 39.3->46.0 (+6.7), GPQA Diamond 50.8->60.4 (+9.6). Self-Verified Distillation beats UQ-TTC on 5 of 6 benchmark comparisons while using a single inference call at test time (vs up to 168).

*Evolution:* Builds on self-improvement/self-distillation line (STaR, Quiet-STaR, Huang 2022) and adapts the Unsolved Questions benchmark's oracle-free compound validators from evaluation to post-training data construction; it explicitly contrasts with Simple Self-Distillation (Zhang 2026), which trains on raw unverified outputs. In the 2026 context it shows prompt-based self-verification can make self-generated data reliable enough to further refine already post-trained models without teachers or tools, motivating future work on matching seed-question difficulty to model capability and stronger mitigations against reinforcing systematic errors.

### Beyond Model Scaling: Test-Time Intervention for Efficient Deep Reasoning
*2026 · method · `method_Think-with-Me_2601.11252.txt` · arXiv [2601.11252](https://arxiv.org/abs/2601.11252)*

Primary metrics are accuracy (%) and average thinking length (token count via the DeepSeek-R1-Distill-Qwen-7B tokenizer); AIME uses @32 (or @3 for human). Behavioral metrics include intervention count, self-termination proportion, feedback/thinking token counts, feedback-to-reasoning ratio, and wall-clock feedback/thinking time. Baselines span DeepSeek-R1-Distill-Qwen-7B/14B/32B, QwQ-32B, Qwen2.5-72B-Instruct/Math-72B, plus efficient-reasoning methods L1-Max, s1-32B, DEER, SEAL, Speculative Thinking, SpecReason, and ablations. Transfer tasks use IFEval, SEP, and LitBench (creative writing win rate judged by Claude-3.7-sonnet).

- AIME24@32 under 8K window: Think-with-Me (LLM proxy) 73.85% at 1,182.50 tokens vs QwQ-32B 66.66% at 4,052.80 (+7.19%, ~81% shorter).
- MATH500: 90.60% at 1,081.87 tokens with LLM proxy.
- Self-termination reaches 73-100% across datasets; scales to 32K (75.94% on AIME24).

**Key results:** AIME24@32: 73.85% accuracy at 1,182.50 tokens (8K window) vs QwQ-32B 66.66% at 4,052.80 tokens — +7.19% accuracy and ~81% shorter reasoning. MATH500: 90.60% at 1,081.87 tokens with LLM proxy. Self-termination reaches 73-100% across datasets, and the method scales to 32K (75.94% on AIME24).

*Evolution:* Building on 2024-25 efficient-reasoning work (L1, s1, DEER, SEAL, SpecReason) and the R1/QwQ line of RL-trained reasoners, Think-with-Me shifts control from internal numerical signals to external, semantically-grounded feedback at the model's intrinsic phase boundaries. By 2026 it anticipates the need for controllable, human/AI-collaborative reasoning budgets under tight context windows, motivating later feedback-verification and context-compression mechanisms for safe and agentic LRM deployment.

### DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
*2026 · report · `report_DeepSeek-V4_2606.19348.txt` · arXiv [2606.19348](https://arxiv.org/abs/2606.19348)*

Evaluations span knowledge/reasoning (MMLU-Pro, SimpleQA-Verified, Chinese-SimpleQA, GPQA Diamond, HLE, LiveCodeBench-v6, Codeforces rating, HMMT 2026 Feb, IMOAnswerBench, Apex, Apex Shortlist, PutnamBench/Lean), long context (OpenAI MRCR 1M MMR, CorpusQA 1M ACC), and agents (Terminal Bench 2.0, SWE-Verified/SWE-Pro/SWE-Multilingual Resolved, BrowseComp Pass@1, HLE w/ tools, GDPval-AA Elo, MCPAtlas, Toolathlon). DeepSeek-V4-Pro-Max is the headline configuration.

- V4-Pro-Max: SimpleQA-Verified 57.9 Pass@1 (~20 pts above prior open models), Codeforces 3206 (ranks 23rd among humans), SWE-Verified 80.6% resolved, Terminal Bench 2.0 67.9, BrowseComp 83.4, MMLU-Pro 87.5, HLE 37.7, Putnam-2025 120/120.
- At 1M-token context, V4-Pro uses only 27% of single-token FLOPs and 10% of KV cache vs V3.2; MRCR-1M 83.5 MMR beats Gemini-3.1-Pro.
- Real-world: 62.7% Chinese-writing win rate vs Gemini-3.1-Pro; 67% R&D-coding pass rate.

**Key results:** DeepSeek-V4-Pro-Max: SimpleQA-Verified 57.9 Pass@1, Codeforces rating 3206 (ranks 23rd among human candidates), SWE-Verified 80.6% resolved, and Putnam-2025 120/120 proof-perfect. At 1M-token context, V4-Pro needs only 27% of V3.2's single-token inference FLOPs and 10% of its KV cache.

*Evolution:* DeepSeek-V4 (2026) extends DeepSeek-V3/V3.2's MoE+MTP backbone and R1's GRPO reasoning RL, but replaces the mixed-RL merging stage with full-vocabulary on-policy distillation from 10+ specialist teachers, cementing distillation-based consolidation as the default for multi-capability models. Its hybrid CSA/HCA attention and native 1M-token context make routine long-horizon agentic workflows and further test-time scaling feasible, foreshadowing online-learning paradigms.

### GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents
*2026 · report · `report_GLM-5V-Turbo_2604.26752.txt` · arXiv [2604.26752](https://arxiv.org/abs/2604.26752)*

Evaluation spans four categories. Multimodal coding: Design2Code, Flame-VLM-Code, Vision2Web. Multimodal tool use: ImageMining, BrowseComp-VL, MMSearch, MMSearchPlus, SimpleVQA, Facts, V*. GUI agents: OSWorld, AndroidWorld, WebVoyager. Text-only coding and Claw: CC-Bench-V2 (CC-Backend, CC-Frontend, CC-RepoExploration), PinchBench, ClawEval, ZClawBench. RL-vs-SFT deltas are reported per task, and per-data-source reward and pass@k training tracks are logged. ImageMining (217 cases, 7 domains, 5 reasoning categories) is introduced as a vision-centric deep-search benchmark.

- Multimodal coding: Design2Code 94.8, surpassing Claude Opus 4.6.
- Multimodal tool use: ImageMining 30.7, BrowseComp-VL 51.9, MMSearch 72.9, MMSearchPlus 30.0 (~8x over GLM-4.6V), SimpleVQA 78.2.
- GUI agents: OSWorld 62.3, AndroidWorld 75.7; RL-vs-SFT deltas include RefCOCO-avg +4.8%, MVBench +5.6%, OSWorld +4.9%, CharXiv +7.7%.

**Key results:** GLM-5V-Turbo scores 94.8 on Design2Code (multimodal coding), beating Claude Opus 4.6. Multimodal agent benchmarks: 30.7 ImageMining, 51.9 BrowseComp-VL, 75.7 AndroidWorld, 62.3 OSWorld, and MMSearchPlus 30.0 (~8x over GLM-4.6V). It preserves text-only coding (CC-Frontend 68.4, CC-RepoExploration 72.2, CC-Backend 22.8) and Claw agents (PinchBench 87.0/80.7).

*Evolution:* Building on GLM-4.5V/GLM-4.1V-Thinking's multi-task RL insights and multi-token prediction, GLM-5V-Turbo (2026) pushes toward native multimodal agents where perception, not just reasoning, is foundational, reacting against text-centric agent models. It motivates later work on multimodal-native context/memory for long-horizon agents and explicit model-harness co-design, both flagged as open challenges.

### GLM-5: from Vibe Coding to Agentic Engineering
*2026 · report · `report_GLM-5_2602.15763.txt` · arXiv [2602.15763](https://arxiv.org/abs/2602.15763)*

GLM-5 is evaluated on ARC benchmarks vs GLM-4.7, DeepSeek-V3.2, Kimi-K2.5, Claude Opus 4.5, Gemini 3 Pro, and GPT-5.2. Reasoning: HLE (30.5 no tools, 50.4 with tools), AIME 2026 I (92.7), HMMT Feb/Nov 2025 (97.9/96.9), IMO-AnswerBench (82.5), GPQA-Diamond (86.0), LongBench v2 (64.5). Coding: SWE-bench Verified 77.8, SWE-bench Multilingual 73.3, Terminal-Bench 2.0 56.2/60.7, CyberGym 43.2. Agentic: BrowseComp 75.9, BrowseComp-ZH 72.7, tau2-Bench 65.8, MCP-Atlas 67.8, Tool-Decathlon 89.7, Vending-Bench 2 $4,432, GDPval-AA Elo 1,409.

- Artificial Analysis Intelligence Index v4.0 reaches 50 (first open-weights model; up from 42); LMArena #1 open model in Text and Code Arena.
- Internal CC-Bench-V2: frontend BSR 98.0%, backend Pass@1 25.8, repo exploration 65.6, chained tasks 52.3.
- SWE-rebench 42.1% resolved (Pass@5 50.0%); slide generation 16:9 compliance 40%→92%, win rate 67.5% vs GLM-4.5; Agent-as-a-Judge 94% human agreement (Spearman 85.7%).

**Key results:** GLM-5 scores 50 on the Artificial Analysis Intelligence Index v4.0, the first open-weights model to do so (up from GLM-4.7's 42), and is the #1 open model on LMArena Text and Code Arena. It reaches SWE-bench Verified 77.8, BrowseComp 75.9 (with context management), tau2-Bench 65.8, and Vending-Bench 2 $4,432, roughly a 20% average gain over GLM-4.7 and on par with Claude Opus 4.5 / GPT-5.2.

*Evolution:* GLM-5 builds on GLM-4.5's MoE and decoupled rollout engines by adopting DeepSeek's DSA sparse attention and a GRPO+IcePop RL core, pushing the open-weights frontier from passive coding toward long-horizon agentic engineering. Its asynchronous agent RL stack (TITO, double-sided importance sampling, multi-task orchestrator, cross-stage distillation) and full-stack Chinese-chip adaptation anticipate the 2026 need for efficient, scalable on-policy RL infrastructure for frontier open models.

### Kimi K3: Open Frontier Intelligence (Technical Report of Kimi K3)
*2026 · report · `report_Kimi-K3_2607.24653.txt` · arXiv [2607.24653](https://arxiv.org/abs/2607.24653)*

Evaluation spans four axes. Reasoning & Knowledge: GPQA Diamond 93.5, CritPt 23.4, AA-LCR 74.7, HLE-Full 43.5/56.0 (without/with tools). Coding: DeepSWE 67.5, ProgramBench 77.8 (best of all models), Terminal-Bench 2.1 88.3, FrontierSWE 81.2, SWE-Marathon 42.0 (7 points above Fable 5). Agentic: BrowseComp 91.2, DeepSearchQA 95.0 F1, ResearchRubrics 76.2, GDPval-AA v2 1686 Elo, MCPMark-Verified 94.5, OSWorld-Verified 84.8. Vision: OmniDocBench 91.1, Math-Vision 94.3 (97.8 with Python tools), ZeroBench-main 23.0 (41.0 pass@5 with tools). Baselines are Claude Fable 5, GPT-5.6 Sol, Opus 4.8, GPT-5.5, GLM-5.2.

- Third-party: Artificial Analysis Intelligence Index v4.1 57.1 (#4/580), Vals AI Index 74.7% (#2/39), WebDev Arena #1/99 at 1678 Elo (first open model to top it).
- Cybersecurity: 14/36 end-to-end exploits vs 8/36 for GLM-5.2.
- Trails Fable 5 and GPT-5.6 Sol overall but beats the rest of the frontier baselines.

**Key results:** Kimi K3 is a 2.8T-parameter MoE (104B activated) open model scoring 91.2 on BrowseComp, 77.8 on ProgramBench (best), 93.5 on GPQA Diamond, and ranking #1 on WebDev Arena (1678 Elo, the first open model to lead it). It places #4/580 on Artificial Analysis Intelligence Index v4.1 (57.1) and #2/39 on Vals AI Index (74.7%), trailing only Claude Fable 5 and GPT-5.6 Sol.

*Evolution:* Kimi K3 builds on Kimi K2/K2.5 (LatentMoE, agentic RL, partial rollouts) and the DeepSeek-R1/Kimi K1.5 line of large-scale RL reasoning, but pushes both scaling axes together: 3T-class pre-training versus the prior ~1T open regime, and million-token agentic RL/test-time scaling. As the first open 3T-class model released in 2026, it motivates next-generation open infrastructure (microVM sandboxes, hybrid-attention serving) and frontier safety evaluation (cyber exploit suites).

### Qwen3-Coder-Next Technical Report
*2026 · report · `report_Qwen3-Coder-Next_2603.00729.txt` · arXiv [2603.00729](https://arxiv.org/abs/2603.00729)*

Agent benchmarks (max 300 turns) report SWE-Bench Verified, SWE-Bench Multilingual, SWE-Bench Pro, and Terminal-Bench 2.0 across multiple scaffolds (SWE-Agent/MiniSWE-Agent/OpenHands; Terminus2-xml/json/ClaudeCode/QwenCode). Function-level: EvalPlus, MultiPL-E, CRUXEval, LiveCodeBench v6, OJBench, Codeforces. Full-stack/SQL/edit: FullStackBench-en/-zh, Spider, BIRD-SQL, Aider-Polyglot. General: MMLU, MMLU-Redux, MMLU-Pro, GPQA, SuperGPQA. Math: AIME24, AIME25, HMMT25-Feb/-Nov. Cybersecurity: AthenaBench-Mini, PrimeVul-Paired, SecCodeBench, CWEval.

- SWE-Bench Verified 70.6/71.1/71.3% (SWE-Agent/MiniSWE-Agent/OpenHands); SWE-Bench Pro 42.7/38.7%; Terminal-Bench 2.0 34.2/36.2/30.9/25.8%.
- Function-level: EvalPlus 86.56, MultiPL-E 88.23, CRUXEval 95.88, LiveCodeBench v6 58.93, Codeforces 2100.
- Tool-template following averages 92.7% across 5 IDE/CLI scaffolds vs GPT-5-2 49.3, Claude-sonnet-4-5 85.4; SecCodeBench 61.2 (gen w/o hint), CWEval func-sec@1 56.32.

**Key results:** Qwen3-Coder-Next (80B total / 3B active MoE) scores 70.6/71.1/71.3% on SWE-Bench Verified across SWE-Agent/MiniSWE-Agent/OpenHands, competitive with much larger frontier and open models. It reaches 42.7% on SWE-Bench Pro (SWE-Agent) and 36.2% on Terminal-Bench 2.0 (Terminus2-json). Tool-template following averages 92.7% across 5 IDE/CLI scaffolds vs 49.3 (GPT-5-2) and 85.4 (Claude-sonnet-4-5).

*Evolution:* Builds on the Qwen2.5-Coder lineage and the SWE-Smith/SWE-Flow/SWE-Rebench trend of scaling executable SWE training data, plus DeepSeek-R1-style execution-driven RL, but pushes agentic training onto a small-active-footprint MoE (80B/3B active). Its central claim-that scaling agentic training rather than model size drives real-world coding-agent capability-motivates future work on visual agent evaluation and agentic cybersecurity/CTF tasks.

### Qwen3.5-Omni Technical Report
*2026 · report · `report_Qwen3.5-Omni_2604.15804.txt` · arXiv [2604.15804](https://arxiv.org/abs/2604.15804)*

Evaluation covers 215 audio and audio-visual subtasks across two variants (Flash, Plus). Text→Text uses MMLU-Pro, MMLU-Redux, SuperGPQA, C-Eval, IFEval, IFBench, AA-LCR, LongBench v2, GPQA, LiveCodeBench v6, HMMT Nov 25, IMOAnswerBench, BFCL-V4, TAU2Bench. Audio→Text uses MMAU, MMAR, MMSU, RUL-MuchoMusic, SongFormBench, VoiceBench, URO-Bench-pro, SpeechRole, WildSpeech-Bench, Fleurs S2TT, and ASR sets. Vision→Text spans MMMU/MMMU-Pro, MathVista/Vision, DynaMath, ZEROBench, RealWorldQA, MMStar, OCR/doc, spatial-intelligence, video, and medical VQA. Audio-Visual→Text uses DailyOmni, WorldSense, AVUT, AV-SpeakerBench, VideoMME, OmniCloze, OmniGAIA. Speech generation is measured by WER, cosine speaker similarity, and BLEU.

- Plus beats Gemini-3.1 Pro on most audio tasks; FLEURS ASR avg WER 6.6% vs 7.3%.
- SEED-TTS test-en WER 1.26; cross-lingual zh→ko 4.03 vs 14.4 (~72% relative reduction).
- Matches Qwen3.5-Plus-Instruct on text/vision (MMLU-Pro 85.9, IFEval 89.7); first-packet latency 435ms (audio)/651ms (video).

**Key results:** Qwen3.5-Omni-Plus achieves SOTA across 215 audio/audio-visual benchmarks, surpassing Gemini-3.1 Pro on audio understanding, recognition, translation, and dialogue (FLEURS ASR avg WER 6.6% vs 7.3%; en2xx avg BLEU 33.8 vs 31.8). On SEED-TTS it reaches test-en WER 1.26, and in cross-lingual TTS it cuts zh->ko error from 14.4 (CosyVoice3) to 4.03 (~72% relative reduction). It matches Qwen3.5-Plus-Instruct on text/vision (MMLU-Pro 85.9, IFEval 89.7) with Plus first-packet latency of 435ms (audio) / 651ms (video).

*Evolution:* Building on the Thinker-Talker design of Qwen2.5-Omni and Qwen3-Omni, Qwen3.5-Omni (2026) scales native omnimodal training to hundreds of billions of parameters with Hybrid-Attention MoE and the ARIA streaming-alignment trick. It moves the field from passive perception-response systems toward native omni-agents that act (WebSearch, FunctionCall) and even generate executable code from audio-visual instructions (Audio-Visual Vibe Coding), motivating future work on general-purpose omnimodal agents.

### The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence
*2026 · report · `report_MiniMax-M2-Series_2605.26494.txt` · arXiv [2605.26494](https://arxiv.org/abs/2605.26494)*

Coding uses SWE-bench Pro, SWE-bench Multilingual, Multi-SWE-bench and internal NL2Repo (Claude Code scaffold, 4 trials), Terminal-Bench 2.0 (Terminus-2) and MLE Bench Lite (medal rate, 3x24h). AppDev uses internal VIBE-Pro and HyperTask; search uses BrowseComp, Wide Search and internal RISE; office/tools use GDPval-AA, Toolathlon, internal MM Claw, MEWC v2 and Finance Modeling Pro. Reasoning (AIME 2026, GPQA-D, SciCode, IFBench, AA-LCR, HLE, MMLU-Pro) follows the AA Index v4.0 protocol (pass@1).

- M2.7 coding: SWE-bench Pro 56.2, SWE-Multilingual 76.5, Multi-SWE 52.7, Terminal-Bench 2.0 57.0, MLE Lite 66.6.
- M2.7 agentic: VIBE-Pro 55.6, BrowseComp 77.8, GDPval-AA 50.0, Toolathlon 46.3.
- M2.7 reasoning: AIME26 94.2, GPQA-D 89.8, IFBench 76.0, HLE 28.0.

**Key results:** With about 10B active parameters, M2.7 is close to closed frontier models. It has the best Multi-SWE-bench score among the compared models (52.7 vs 50.3 for Opus 4.6), scores 56.2 on SWE-bench Pro against 57.3 for Opus 4.6 and 57.7 for GPT 5.4, and reaches a 66.6% MLE Bench Lite medal rate, tying Gemini 3.1 Pro. From M2 to M2.7 every tracked benchmark improves, for example BrowseComp by 33.8 points and Toolathlon by 27.5. M2.7 also handles 30-50% of the RL team's daily iteration work and, in 100 autonomous rounds of improving an internal scaffold, gained 30%.

*Evolution:* The series follows MiniMax-01 and MiniMax-M1 and keeps M1's CISPO objective, but it drops their hybrid Lightning Attention in favour of full attention, after SWA hybrids degraded on agent tasks longer than 32K. It borrows MTP and auxiliary-loss-free expert-bias routing from DeepSeek-V3 and SWE-Smith-style task augmentation. It moves post-training from reasoning RL toward agent-native RL over arbitrary black-box scaffolds, and it gives an early, partly human-steered example of a model debugging its own training runs. Note: the arXiv ID is 2605 (May 2026, v2 dated July 2026), earlier than the corpus's 2608-2610 target window.

### Seed2.0 Model Card: Towards Intelligence Frontier for Real-World Complexity
*2026 · report · `report_Seed2.0_2607.00248.txt` · arXiv [2607.00248](https://arxiv.org/abs/2607.00248)*

Evaluation covers 50 image and 24 video benchmarks plus language (AIME, HMMT, IMOAnswerBench, Codeforces Elo, LiveCodeBench v6, GPQA, ARC-AGI, MMLU-Pro, SuperGPQA, MultiChallenge, Frames, MRCR), agentic (SWE-Bench Verified/Pro, Terminal-Bench 2.0, BrowseComp, HLE, τ²-Bench, BFCL-v4, MCP-Mark) and in-house enterprise suites. Baselines are GPT-5.2 High, Claude-Opus/Sonnet-4.5, Gemini-3-Pro and Seed1.8.

- Competitions: IMO 2025 35/42 and CMO 2025 114/126 (both gold); ICPC 2025 pass@8 73.02%; Putnam-200 35.5% pass@8.
- Codeforces Elo 3020; MathVision 88.8; VideoMME 89.5.
- In-house Chinese complex-IF benchmark 75.26% (+2.37 over Seed1.8).

**Key results:** Seed2.0 Pro reaches gold-medal level on IMO 2025 (35/42) and CMO 2025 (114/126), has a Codeforces Elo of 3020 and scores 73.02% ICPC pass@8 (gold in all five 2025 contests). It reaches 35.5% pass@8 on Putnam-200 (above Seed-1.5-Prover's 26.5), and leads most vision and video benchmarks. The authors admit clear gaps to Claude in coding (SWE-Evo, NL2Repo) and to Gemini on long-tail knowledge (SuperGPQA, SimpleQA-Verified).

*Evolution:* Successor to Seed1.5-Thinking/Seed1.6/Seed1.8 and Seed1.5-VL. It marks ByteDance Seed's shift from Olympiad-style reasoning toward agentic, long-horizon real-world tasks and research-level reasoning (Erdős problems, scientific coding). Unlike the Seed1.5-Thinking report, it is a model card with no training recipe. Its contributions are deployment-usage analysis, a ~10x cost advantage over frontier APIs, and a suite of new real-world benchmarks.

### Gemma 4 Technical Report
*2026 · report · `report_Gemma-4_2607.02770.txt` · arXiv [2607.02770](https://arxiv.org/abs/2607.02770)*

Human evaluation is Arena Text Elo. Static benchmarks, mostly in thinking mode, cover MMLU Pro, AIME 2026 (no tools), LiveCodeBench v6, Codeforces Elo, SciCode, GPQA Diamond, BBEH, HLE (with/without search), IFBench, IFEval, MMMLU, MRCR v2 8-needle at 128k, Terminal Bench Hard and Tau2 (airline/retail/telecom). Vision uses MMMU Pro, MATH-Vision, MedXPertQA MM, InfographicVQA and OmniDocBench 1.5; audio uses CoVoST BLEU and FLEURS WER/CER; long context uses RULER, LOFT, GraphWalks and MTOB (up to ~256k).

- 31B: MMLU Pro 85.2, AIME 2026 89.2, LiveCodeBench 80.0, Codeforces 2150, GPQA 84.3.
- 31B: HLE 19.5, IFEval 98.9, Tau2-retail 86.4.
- RULER-128k 96.4 vs 66.0 for Gemma 3 27B.

**Key results:** Gemma 4 31B scores 1451 Elo on Arena Text, the top dense open model, close to far larger MoEs such as DeepSeek V4 Pro (1456). The 26B-A4B scores 1438, vs 1366 for Gemma 3 27B. The 31B lifts AIME 2026 from 20.8 to 89.2 and Codeforces Elo from 110 to 2150 over Gemma 3 27B. On audio, E2B/E4B improve over Gemma 3n by 12%/10% on translation and 17%/12% on ASR with a 78% smaller encoder.

*Evolution:* Gemma 4 succeeds Gemma 2/3 and Gemma 3n, reusing Gemma 3's local:global attention recipe and Gemma 3n's per-layer embeddings and USM audio encoder. Its main post-training addition is an o1-style thinking mode, which brings the open Gemma line into the reasoning-model era. The report's other contributions are efficiency work: K=V global layers, p-RoPE, KV sharing, MTP drafters, QAT and an encoder-free 12B. The post-training recipe itself is barely documented, so the report is more useful as an efficient-architecture reference than as a post-training recipe.

### Solar Open 2 Technical Report
*2026 · report · `report_Solar-Open-2_2607.20062.txt` · arXiv [2607.20062](https://arxiv.org/abs/2607.20062)*

English benchmarks: MMLU-Pro, GPQA-Diamond, HLE (no tools), LiveCodeBench v6, ArtifactsBench, HMMT2602, AIME2026, MultiChallenge, IFBench, AA-LCR, SWE-Bench Verified, Terminal Bench Hard, APEX-Agents, MCP-Atlas, tau3 banking and GDPval-AA v2 (ELO). Korean: KMMLU-Pro, CLIcK, HAE-RAE v1.1, Ko-AIME'25, HRM8K, KBank-MMLU, KBL, KorMedMCQA and in-house Ko-GDPval (170 tasks, 58 occupations, 11 domains). Baselines are DeepSeek-V4-Flash, MiMo-V2.5, Command A+, Mistral Medium 3.5, Solar Open 1, plus Claude Haiku 4.5 and GPT-5.4 mini on Korean.

- MMLU-Pro 86.2, LiveCodeBench 92.4, GPQA-D 86.3, AIME2026 95.7.
- SWE-Bench Verified 70.4, Terminal Bench Hard 28.3, APEX-Agents 16.6, MCP-Atlas 58.2, GDPval-AA 1128.
- Korean average 85.4, Ko-GDPval 86.8; blind pairwise test on 835 Korean turns prefers it over DeepSeek-V4-Flash 44.1% vs 26.6%.

**Key results:** Solar Open 2 leads comparably sized open-weight models on MMLU-Pro (86.2), LiveCodeBench v6 (92.4) and APEX-Agents (16.6), posts the top Korean average (85.4 vs DeepSeek-V4-Flash 84.9 and GPT-5.4 mini 80.8), and scores 86.8 on Ko-GDPval vs 86.9 for the 1.6T DeepSeek-V4-Pro (Solar Open 1: 3.4). Gaps remain on SWE-Bench Verified (70.4 vs 73.0-73.8) and Terminal Bench Hard (28.3 vs MiMo-V2.5's 41.7).

*Evolution:* Solar Open 2 scales Upstage's Solar Open 1 (102B-A12B) into a Korean sovereign agentic model. It adopts the 2025-26 hybrid linear/softmax attention trend (Kimi Linear, Qwen3.5, Nemotron 3) and adds NoPE and negative eigenvalues. It also follows the specialist-then-on-policy-distillation consolidation used by DeepSeek-V4, Nemotron and MiMo, but drops the outcome reward and uses exact full-vocabulary KL. Its selective cross-architecture weight transfer and OfficeVerse office-task synthesis are reusable ideas for continuing model families across architecture changes and for training deliverable-producing agents in non-English languages.

### From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement
*2026 · rl · `rl_RLSVR_2607.23802.txt` · arXiv [2607.23802](https://arxiv.org/abs/2607.23802)*

Summarization is scored by ROUGE-L on GovReport, Multi-News, QMSum, VCSum, SAMSum (plus arXiv, PubMed, BillSum) and GPT-4o pairwise A/B win rates with swapped-order conservative aggregation. Creative writing uses GPT-4o A/B on WritingPrompts and WritingBench (novelty, emotion, coherence, consistency, overall). Reasoning is accuracy on GSM8K, MATH500, AIME24/25, Minerva, MMLU-Pro, GPQA-Diamond, AMC, OlympiadBench and SuperGPQA. A blinded human study (10 PhD students, 400 prompts) and Gemini-3.5-Flash as alternative judge check the judging; GPT-4o-vs-human agreement is 91.0% precision / 93.8% recall.

- Qwen3-4B GovReport ROUGE-L 36.7 vs 33.2 for Absolute Zero.
- Win rates vs own base (4B / 8B): 73.9% / 75.4% summarization average, 81.3% / 76.5% WritingPrompts overall.
- Qwen3-4B reasoning: GSM8K 93.4, MATH500 79.5, AIME25 20.0, GPQA-D 41.3 (base 84.5 / 68.2 / 6.7 / 26.3).

**Key results:** SpyRL beats the R-Zero and Absolute Zero self-play baselines on every summarization and creative-writing benchmark (R-Zero even degrades creative writing to 48.8% vs base), and improves Qwen3-4B/8B math reasoning across seven benchmarks (4B average 41.4 to 50.4). Without any external verifier it beats a Qwen3.5-27B rubric-as-reward baseline (59.3% overall win on WritingPrompts) and is roughly on par with GPT-4o-RaR (48.9%), which cost about $900 in verifier calls.

*Evolution:* RLSVR takes the verifiable-reward machinery behind o1/DeepSeek-R1-style RLVR (GRPO) and the label-by-construction idea of self-supervised pretext tasks (masked LM, contrastive learning). It aims them at open-ended generation, which until now relied on RLHF/DPO, LLM-as-judge, self-rewarding or rubric-as-reward signals that bring judge bias and cost. It extends proposer-solver self-play (Absolute Zero, R-Zero), which needs verifiable solver feedback, to multi-agent information-asymmetric games, and borrows Role-Advantage Estimation from prior multi-agent self-play RL. The broader claim is that verifiability can be engineered rather than being a fixed property of a task.

### Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents
*2026 · report · `report_Qwen-UI-Agent_2607.28227.txt` · arXiv [2607.28227](https://arxiv.org/abs/2607.28227)*

Mobile: MobileWorld GUI-only (117 tasks, SR), the new MobileWorld-Real (409 human-written tasks over 104 apps on live phones, scored by AutoJudge, a 5-VLM majority vote with 92.8% expert agreement on 666 trajectories) and AndroidDaily. Computer: OSWorld-Verified and OSWorld-v2 (partial/binary, steps/task). Web: WebArena (corrected references and scripts), BrowseComp/-ZH. Grounding: ScreenSpot-Pro/V2, MMBench-GUI L2, OSWorld-G-Refined, UI-Vision. General/agentic: MMMU-Pro, MMLU-Pro, IFEval, Tau2, Terminal-Bench 2.0, Claw-Eval, BFCL-v4, SkillsBench.

- 27B mobile: MobileWorld 82.1 (85.5 at 100 steps), MobileWorld-Real 92.2, AndroidDaily 97.5.
- 27B computer/web: OSWorld-Verified 79.5 (2nd to Opus 4.8's 83.4), OSWorld-v2 40.0/13.9, WebArena 73.6.
- ScreenSpot-Pro 81.5 with zoom.

**Key results:** The 27B model sets mobile SOTA: 82.1% on MobileWorld (vs Seed 2.1 Pro 73.2, Opus 4.8 67.5) and 92.2% on MobileWorld-Real (vs Seed 2.1 Pro 88.7). It reaches 73.6% on WebArena (vs Opus 4.8 71.9) and 79.5% on OSWorld-Verified, while lifting Terminal-Bench 2.0 to 50.1 vs 41.1 for the Qwen3.5-27B base comparison. Action RL raises task SR by more than 7% while cutting reasoning tokens 21.3%. Online RL increases trajectories containing a verification action by 14.7% and cuts false stops by 11.2%.

*Evolution:* This report continues MAI-UI (step-level trajectory judge, GRPO-style online RL, rejection-sampled SFT) and joins the 2026 wave of native GUI agents (UI-TARS-2, GUI-Owl-1.5, UI-Venus-1.5, EvoCUA) that scale verifiable environments and online RL. It pushes past sandbox benchmarks: a 100+-phone real-device runtime and the MobileWorld-Real benchmark target the sim-to-real gap it documents in Qwen3.7-Plus failures. Its hybrid GUI+CLI batched action space, where online RL produces a 'Bash as hands, GUI as eyes' pattern, and its agent-driven data flywheel point toward agents that partly build their own training curricula.

### Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering
*2026 · rl · `rl_Frontis-MA1_2607.28568.txt` · arXiv [2607.28568](https://arxiv.org/abs/2607.28568)*

The primary benchmark is MLE-Bench Lite (22 tasks), 3 runs per configuration under a 12-hour per-task budget on one RTX 4090 capped at 12 GB VRAM. Metrics are Valid Rate (x/22), Medal Average and Human Rank. Transfer is measured on NatureBench Lite (10 tasks, 4h budget) with Match-SOTA (g >= 0) and Surpass-SOTA (g > 0.1).

- Frontis-MA1-35B: OpenMLE-Evo 21.67/22 valid, 60.61% medals (+/-7.73), Human Rank 0.7647; OpenMLE-Evo-Max 22/22, 71.21%, 0.8126.
- Frontis-MA1-30B: 53.03% (Evo), 66.67% (Evo-Max); references GPT-5.6 Sol+Codex 72.73%, Kimi K3+Claude Code 72.73%, GPT-5.5+Codex 68.18%, Opus 4.8+Claude Code 63.64%.
- Search efficiency vs AIRA-Evo: 41.7% fewer total tokens; new-best updates per 1M tokens 1.77 -> 3.27.

**Key results:** Under the identical OpenMLE-Evo harness, post-training lifts Medal Average from 39.39% (Qwen3.6-35B-A3B) to 60.61%, and from 34.85% to 53.03% on the 30B replication. Frontis-MA1-35B with OpenMLE-Evo-Max reaches 71.21%, above GPT-5.5+Codex (68.18%) and approaching GPT-5.6 Sol and Kimi K3 (72.73%). On NatureBench Lite, the trained model raises Match-SOTA from 50% to 70% with the framework fixed, and OpenMLE-Evo raises it from 20% to 50% with the model fixed.

*Evolution:* Frontis-MA1 combines three previously separate MLE-agent strands: AIDE/AIRA-style evolutionary search, executable environments (MLE-Dojo, MLE-Smith, MLE-Bench), and RL post-training of MLE agents (ML-Agent, MLE-RL, AceGRPO). Its central move is to train the same Draft/Improve/Debug/Crossover operators that the inference-time evolution composes. It borrows upper-tail entropic advantages from TTT-Discover and asynchronous search from AIRA2. The authors release the full stack (tasks, sandbox, training code, weights) as an open testbed for 'meta-evolution' toward AI4AI recursive self-improvement, and explicitly say it does not yet achieve general autonomous RSI.

### Qwen-CUA: Native Computer Use for (almost) Everything
*2026 · report · `report_Qwen-CUA_2608.02352.txt` · arXiv [2608.02352](https://arxiv.org/abs/2608.02352)*

Pure screenshot + keyboard/mouse evaluation on eight benchmarks: OSWorld-Verified (360 tasks, 100 steps), OSWorld 2.0 (binary/partial completion), MyPCBench (184 tasks, rubric/perfect-task), MacAgentBench (676 tasks on 30 bare-metal Mac minis, Pass@1), Gym-Anything/CUA-World (1,295 tasks over 97 envs, strict-clean), ScienceBoard (169 tasks), WebArena and RedTeamCUA (task success/ASR). Baselines are Qwen3.7, GPT-5.5 and Opus-4.8.

- OSWorld-Verified 86.2 vs 73.3/78.7/83.4 (Qwen3.7/GPT-5.5/Opus-4.8).
- OSWorld 2.0 18.5/48.4 vs 2.5/22.5 for Qwen3.7 (Opus 20.3/54.8); MacAgentBench 69.2.
- ScienceBoard 64.50 vs 35.50; Gym-Anything 46.3 vs 33.1; 3,605.8 output tokens per OSWorld-Verified task.

**Key results:** Qwen-CUA reaches 86.2 on OSWorld-Verified, above GPT-5.5 (78.7) and Claude Opus 4.8 (83.4), and beats Qwen3.7 on all eight benchmarks. It cuts RedTeamCUA attack success from 36.6 to 16.4 while raising task success from 70.5 to 74.0. The >1T-parameter Qwen-CUA-Max lifts OSWorld-Verified to 87.6 and OSWorld 2.0 to 21.2/53.3.

*Evolution:* Qwen-CUA continues the native screenshot-only agent line (UI-TARS, OpenCUA, Qwen3-VL) and builds directly on CUA-Gym's verifiable environment and task scaling and on SAPO for stable long-trajectory MoE RL. Its distinctive moves are an iterative SFT-refresh/RL-calibration loop that always restarts SFT from mid-training, and train-inference-consistent screenshot folding that reuses sparse terminal rewards. Its Bash-hybrid study (fewer turns but lower MyPCBench success) frames native computer use as the universal fallback layer for future hybrid code and GUI agents.

### Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA
*2026 · method · `method_Macaron-MoL_2608.09819.txt` · arXiv [2608.09819](https://arxiv.org/abs/2608.09819)*

Internal Personal Intelligence benchmarks: Macaron ChatBench (46 cases, GLM-5.2 judge with six fixed axioms plus persona/scenario conditioning, 1-5 score, 3 samples), LivingBench (40 simulated scenarios, up to 10 turns, four noise types; score = 0.7 x need fulfilment + 0.3 x process quality) and UI4A-Bench (161 cases, Final Score from five weighted Layer Scores). External: VitaBench/VitaBench2 (Avg@1), tau3-Bench, PinchBench, ClawGym, SWE-Verified, TerminalBench 2.1, DeepSWE (best of 3) and SWE Atlas QnA (pass@3), coding rows on the Claude Code scaffold; some baseline cells come from public leaderboards.

- Venti: ChatBench 58.3 (GPT-5.5 55.5), LivingBench 64.0 (Opus 4.8 63.8), UI4A-Bench 87.8 (Opus 4.8 75.9).
- Venti coding: TerminalBench 2.1 87.6, SWE-Verified 85.6 (Opus 4.8 88.6), DeepSWE 58.4 (GPT-5.5 70.0).
- Routing accuracy 99.12% on a 6,448-sample trace drawn from training data.

**Key results:** Macaron-V1-Venti leads the compared frontier models on ChatBench (58.3), LivingBench (64.0), UI4A-Bench (87.8 vs 75.9 for Opus 4.8) and TerminalBench 2.1 (87.6), but trails on VitaBench, ClawGym, SWE-Verified, DeepSWE and SWE Atlas QnA. Macaron-V1-Tall beats its Qwen3.6-35B-A3B base on all seven matched rows (e.g. UI4A-Bench 59.3 vs 33.9). Separately, with the GLM-5.2 base frozen and no weight updates, adaptive harness-configuration search eventually passes all 122 TerminalBench-derived tasks the base had failed, whereas the best single full-set configuration passed 11/122. MoL stores about 74% fewer parameters than four replicated merged-base copies.

*Evolution:* Macaron-V1 builds on LoRA (including Thinking Machines' 'LoRA without regret'), multi-tenant LoRA serving (Punica, S-LoRA), GRPO, and its own earlier Macaron-V1-Preview and Macaron-A2UI work. It reacts to the cross-task interference and static-checkpoint limits of monolithic post-training. It shifts post-training toward an agent system with separately versioned base, adapters, and harness, where harness search ('language space') and adapter RL are coupled. It positions MoL as an interoperability contract for continual learning and for composing specialists from different teams. The authors state that compounding gains across generations and collective-intelligence benefits are not yet demonstrated.

### Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning
*2026 · report · `report_Palmyra-x6_2608.16620.txt` · arXiv [2608.16620](https://arxiv.org/abs/2608.16620)*

Scores are on a 0–1 scale with standard errors from Writer's internal harness. The main comparison is against the previous Writer Agent default and five frontier models on six agentic, finance and instruction-following benchmarks (including MCP-Atlas, FinanceBench, IFBench, BFCL Core); internal enterprise-domain datasets are the real target, with public benchmarks as a sanity check. Safety uses FORTRESS (500 adversarial prompts plus benign twins), reported as 100−ARS and 100−ORS and judged by GPT-5.6-Luna.

- Model-risk evaluation: 8 endpoints, 19,674 responses, 33,862 blinded judge scorings.
- Includes a paired political-symmetry suite (556 families / 1,112 prompts), BBQ, OR-Bench, TruthfulQA, Anthropic Political Neutrality Eval, Washington Post ModelSlant and China-sensitive probes in 5 languages.
- Statistics use paired bootstrap, Wilcoxon and Benjamini–Hochberg tests.

**Key results:** Palmyra x6 beats the prior Writer Agent default on all six benchmarks: +0.320 on MCP-Atlas, +0.305 on FinanceBench and +0.304 on IFBench. Against five frontier models it tops BFCL Core (0.785) and has the highest six-benchmark mean (0.765). It presents both sides on 80% of ModelSlant questions, and TruthfulQA is 80.6% vs the base's 81.5%. On FORTRESS, with its deployment system message, it scores 67.0 adversarial / 96.4 benign, vs 58.4 / 97.2 for the base (55.4 / 97.4 without the system message).

*Evolution:* This report is an example of post-training an existing open frontier MoE base (GLM-5.2, which inherits GLM-5's MLA + DSA IndexShare architecture) rather than training a model end to end. It combines Anchored SFT (DFT weighting plus a KL anchor), SDFT-style demonstration-conditioned self-distillation for on-policy data, and LIMO/LIMI 'less is more' data minimalism. It reacts to SFT's tendency to erode base capabilities by keeping updates small and anchored and avoiding RL entirely. It also shows Muon (with GLM-5's per-head Muon Split) working as a fine-tuning optimizer at the 744B scale.

### Apodex 1.1: Scaling Agentic Intelligence for Complex Work
*2026 · report · `report_Apodex-1.1_2608.23283.txt` · arXiv [2608.23283](https://arxiv.org/abs/2608.23283)*

Two modes are evaluated, ReAct (minimal scaffold) and Agent Team, across professional work (APEX-Agents, GDPVal win rate), finance (FrontierFinance, YC-Bench net worth), science (FrontierScience-Research, BioMysteryBench Human-difficult), reasoning/search (HLE with tools, DeepSearchQA F1), math under the MathArena protocol (IMO 2025/2026, USAMO 2026, IMO-ProofBench) and coding (Terminal-Bench 2.1, SWE-bench Verified). Internal suites are FrontierSearchBench (41 tasks, score in [-100,100]), FrontierResearchBench (pass rate) and HDS6 process grading (6 capabilities x 4 items, GPT-5.5/Claude judge panel with an integrity gate).

- ReAct / Team: APEX-Agents 34.4 / 38.5, GDPVal 69.5 / 78.8, FrontierFinance 48.7 / 54.3, FrontierScience-Research 55.0 / 63.3, BioMysteryBench 23.5 / 35.3, HLE 53.2 / 56.1, DeepSearchQA 88.2 / 92.4.
- Math (Agent Team): IMO 2025 36.5, IMO 2026 30.5, USAMO 2026 26.5 (thresholds 35/29/25); IMO-ProofBench Basic 96.7 / Advanced 63.3.
- Terminal-Bench 2.1 70.8, SWE-bench Verified 77.7; FrontierSearchBench 57.0 / 69.1; 35B mini: FrontierFinance 50.2, FrontierScience 51.7, APEX-Agents 27.7 (Team).

**Key results:** With Agent Team, Apodex 1.1 posts the best values in its comparison tables on FrontierFinance (54.3, vs Claude Fable 5 at 49.2) and FrontierScience-Research (63.3, vs DeepSeek-V4-Flash-0731 at 55.0), and passes the IMO 2025/2026 and USAMO 2026 thresholds. On APEX-Agents it improves from Apodex 1.0's 16.5 to 38.5, and it reaches GDPVal 78.8. It still trails the frontier on SWE-bench Verified (77.7 vs Claude Opus 5's 92.2), Terminal-Bench 2.1 (70.8 vs 91.9) and FrontierResearchBench (12.4% vs 20.6%).

*Evolution:* Apodex 1.1 extends Apodex 1.0's lead-agent/subagent Agent Team and AgentOS kernel. It follows the 2025-26 shift from single-response models to long-horizon working agents (Kimi K3, GPT-5.6, GLM-5.2, DeepSeek V4), and it treats environment construction (in the vein of ScaleEnv and Agent-World) and multi-agent coordination as two explicit scaling axes. PIVOT-RL's hinted continuation from failure pivots joins recent work on credit assignment at informative intermediate turns in agentic RL. The paper is light on model and training hyperparameters and heavy on harness and evaluation design, including process-level HDS6 grading.

### On-policy Distillation with Verifiable Reward
*2026 · rl · `rl_OPDVR_2608.24696.txt` · arXiv [2608.24696](https://arxiv.org/abs/2608.24696)*

Six math benchmarks (AIME24, AIME25, AMC, MATH500, Minerva, OlympiadBench) scored by avg@16 accuracy at temperature 0.7, top-p 0.95. Training dynamics track entropy, response length and zero-gated token ratio (about 0.48-0.50 for 4B, 0.40-0.44 for 1.7B).

- Same-architecture (Qwen3-4B <- Qwen3-4B-RL) average: student 42.0, teacher 50.4, sampled-token OPD 47.8, Top-64 OPD 47.4, OPD+GRPO 47.7, RLSD 44.8, ExOPD 47.9, SG-OPD 47.5, OPDVR 49.1 (AIME24 36.9 vs teacher 36.0).
- Cross-architecture (Qwen3-1.7B-Base <- Qwen3-4B-Base-RL): OPD 20.9, Top-64 OPD 21.7, OPDVR 22.8.
- GRPD table average: GRPO 44.8, OPD 48.4, Distilled RL 48.7, GRPD 49.4.

**Key results:** OPDVR raises the same-architecture average from 47.8 (sampled-token OPD) to 49.1, beating the next-best baseline ExOPD by 1.2 points, and lifts the cross-architecture average to 22.8 vs 21.7 for the best baseline; GRPD improves over GRPO by 6.5 points on AIME24 and 10.9 on AIME25 and over Distilled RL by 0.7 average. An inverse-gated ablation drops to 44.6 average, below vanilla OPD (47.8).

*Evolution:* OPDVR sits in the 2026 wave of on-policy distillation (popularized by Qwen3, MiMo-V2-Flash, GLM-5 and DeepSeek-V4 post-training) and reacts to hybrids that combine OPD with RLVR via weighted losses or advantage-sign switching (OPD+GRPO, Distilled RL, RLSD) or trajectory/token-level conflict filters (Uni-OPD, RG-OPD, SG-OPD). Its contribution is showing that a hyperparameter-free sign gate turns sampled-token OPD into a proper RLVR objective, so it plugs into GRPO/DAPO/PPO and can in principle exceed the teacher rather than being capped by it.

### TTPO: Test-Time Policy Optimization
*2026 · rl · `rl_TTPO_2608.27448.txt` · arXiv [2608.27448](https://arxiv.org/abs/2608.27448)*

Avg@12 at temperature 1.0 with thinking enabled (max 38,912 new tokens) on AIME 2025, AIME 2026, HMMT 2025, HMMT 2026 and BRUMO 2025; Maj@12 is tracked during TTT and non-thinking evaluation is in the appendix. Scores are the peak over saved checkpoints (100 steps for OPSD/TTPO, 500 for GRPO/TTRL).

- OpenThoughts setting (1.7B / 4B / 8B): TTPO 40.1 / 58.6 / 62.6 vs label-supervised OPSD 39.7 / 58.4 / 61.7 and GRPO 35.7 / 57.5 / 61.2.
- TTT: 45.2 / 61.1 / 65.3 vs TTRL 40.2 / 58.8 / 63.0 and OPSD-TTT 41.9 / 59.4 / 63.7.
- Non-thinking gains over base: +25.2 / +30.6 / +36.4 vs +7.1 / +5.8 / +3.5 for OPSD.

**Key results:** Without any labels, TTPO matches or beats ground-truth-supervised OPSD at all three Qwen3 scales, and in pure TTT lifts Qwen3-1.7B from 38.0 to 45.2 average (+5.4 over TTRL, +3.3 over OPSD-TTT). Qwen3-4B with TTPO (61.1) surpasses the Qwen3-8B base (60.7). Swapping pseudo-labels for ground truth performs worse, because on hard problems the correct-answer positive set is nearly empty and both branches starve.

*Evolution:* TTPO combines two 2026 lines: TTRL-style label-free test-time RL with majority-vote rewards (and follow-ups Hi-TTRL and SCRL), and OPSD-style on-policy self-distillation with a privileged-information teacher (and hybrids that mix OPSD with RLVR). It responds to the fragility of pseudo-label distillation, where a wrong vote corrupts every token, with a negative-learning argument: penalizing disagreement stays correct even when the vote is wrong. Its token weighting and masking borrow from TIP and STAPO. It extends dense self-distillation to settings where labels never arrive, though only on verifiable math so far.

### A.X K2 Technical Report
*2026 · report · `report_A.X-K2_2608.30181.txt` · arXiv [2608.30181](https://arxiv.org/abs/2608.30181)*

Thinking-mode comparison against A.X K1, Qwen3.5-397B, Nemotron 3 Ultra, DeepSeek-V4 Flash, GLM-5.1, Kimi-K2.6 and MiniMax-M2.7 across math, Korean, code, science/knowledge, general and agentic suites, plus proof competitions and long context (RULER to 256K, LongBench v1/v2). In-house: KS-Eval (26 tasks), Manufacturing Benchmark and Korean red-teaming ASR.

- Math: AIME26 97.1, Apex 45.8, Apex-shortlist 88.6, KMO26 92.5; IMO 2025 35/42 and all 8 KMO26 round-2 problems.
- Korean: KMMLU-Pro 80.5, KoBALT 73.0, CLIcK 91.6; Code: LiveCodeBench v6 84.0, SciCode 41.0, Terminal Bench v2.1 36.0; HLE 27.8, GPQA-D 85.6, AA-Omniscience 39.6, IFBench 75.9, AA-LCR 66.0.
- Agentic: GDPval Elo 1031, tau2-Telecom 98.0, tau3-Banking 13.0, BrowseComp 9.3; RULER 94.6, LongBench v2 63.9; Manufacturing F1 69.0; red-teaming ASR 12.1% / 33.9% (non-thinking/thinking); NVFP4 retains 99.0% of FP8 accuracy.

**Key results:** A.X K2 leads all compared open-weight models on AIME26 (97.1), Apex (45.8, up from A.X K1's 1.0), KMMLU-Pro (80.5), CLIcK (91.6) and tau2-Telecom (98.0), and reaches IMO 2025 gold threshold (35/42), despite training on fewer tokens (~8.5T) than A.X K1 (~10T). It trails on Terminal Bench (36.0 vs up to 65.9) and BrowseComp (9.3 vs up to 29.1); sparse attention costs no LongBench quality, and NVFP4 serving gives +123.7% throughput over A.X K1 at 120K input.

*Evolution:* A.X K2 is SK Telecom's successor to A.X K1 under Korea's Sovereign AI foundation model project. It combines DeepSeek-V3.2/GLM-5-style indexer sparse attention with gated attention and Gated Norm, and it simplifies the indexer warmup to a sparse-only one. Its post-training continues the hybrid-thinking line (Qwen3, DeepSeek). It replaces A.X K1's dual-track training with paired-response Think-Fusion and adopts MiniMax-M1's CISPO with GDPO multi-reward normalization. Its report that a trainer-rollout FP8 format mismatch collapses RL is practical guidance for low-precision RL on Blackwell hardware.

### Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement
*2026 · method · `method_DoesOPD-Distill_2608.31046.txt` · arXiv [2608.31046](https://arxiv.org/abs/2608.31046)*

In-domain math is AIME24, AIME25 and HMMT25 (Avg@32 and Pass@32, temperature 0.7, top-k 20, top-p 0.8, 32768 max tokens, non-thinking); out-of-domain is MBPP+ and GPQA-Diamond, and code training is evaluated on LiveCodeBench v6 and HumanEval+. Diversity is pairwise 4-gram Jaccard distance; efficiency is training step time.

- AIME24 Avg@32: Qwen3-1.7B 13.44 -> 48.85 (Pass@32 40.00 -> 80.00); Qwen3-4B 23.33 -> 62.08; Qwen3.5-9B 76.35 -> 87.81; Olmo3-7B Think 71.88 -> 77.29.
- Qwen3-1.7B 3-benchmark Avg@32: GRPO 24.79, TTRL 11.81, OPD 22.15, OPSD 23.58, NSR 24.13, OPSA 35.83.
- Step time: GRPO 186.2 s, OPD 61.2 s, OPSA 46.3 s.

**Key results:** Without any teacher, reward, or hint, OPSA lifts Qwen3-1.7B AIME24 Avg@32 by 35.41 points (a 263% relative gain), more than doubles Pass@32 on all three math benchmarks, and beats OPD by 16.77 points on AIME24 and the best on-policy baseline by 11.04 average Avg@32. Token-budget-matched controls (appending 'wait') do not close the gap: GRPO 32.81 and OPD 31.67 vs OPSA 48.85.

*Evolution:* The paper questions the 2025-2026 on-policy distillation line (GKD, MiniLLM, Thinking Machines' K1-estimator OPD, and hint-based self-distillation OPSD) by showing much of OPD's gain comes from suppressing the student's own low-probability tokens, not from teacher knowledge. It connects to negative-sample reinforcement (NSR) and high-entropy 'fork token' findings in RLVR, and to label-free RL such as TTRL and Intuitor, proposing a teacher-free, reward-free token-level objective; the authors note it is tested only up to 9B and may not expand the exploration frontier of already sharp models.

### SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents
*2026 · method · `method_SWE-Bench-Pro-Verified_2609.08149.txt` · arXiv [2609.08149](https://arxiv.org/abs/2609.08149)*

Accuracy is the share of instances where all fail-to-pass and pass-to-pass tests pass, measured for seven models on SWE-Bench Pro and the new Verified version. Validation compares outcomes under the anti-hacking environment (McNemar test, counts of high-risk operations and answer-file access) and checks refined instances.

- Pro -> Verified: Kimi-K3 89.06 -> 62.93, GLM-5.3 81.12 -> 58.82, DeepSeek-V4-Pro-0813 79.48 -> 61.42, DeepSeek-V4-Flash-0731 78.93 -> 59.92, GLM-5.2 78.80 -> 59.51, GPT-5.6-Sol 76.47 -> 61.97, DeepSeek-V4-Pro 49.98 -> 49.93.
- GLM-5.2 anti-hacking: 186 PASS->FAIL vs 15 FAIL->PASS (p<0.001); local high-risk ops 4,213 -> 908 (-78.4%), network 573 -> 4 (-99.3%); of 186 PASS->FAIL, 166 show direct evidence of removed hacking, 0 impaired execution.
- Refinement on 102 instances: 21 FAIL->PASS, 20 PASS->PASS, 2 PASS->FAIL, 59 FAIL->FAIL.

**Key results:** Under the anti-hacking environment GLM-5.2 falls from 78.80% to 57.32% (-21.48 points; refinement then recovers it to 59.51%), whereas DeepSeek-V4-Pro, which showed little hacking, barely moves (49.98 -> 49.11 -> 49.93). Confirmed answer-file access drops to zero, with no PASS->FAIL transition attributed to impaired normal execution.

*Evolution:* Builds on SWE-Bench Pro (2509.16941) and follows the 'Verified' precedent of SWE-bench Verified and SimpleQA Verified, drawing on the AgentCompass trajectory audit, community git-leakage reports and pull requests, and external SWE-Bench Pro quality audits. It shows that six of seven models score 14.5-26.1 points lower on Verified than on SWE-Bench Pro, which suggests existing SWE-Bench Pro results overstate capability. The concurrent SWE-Bench Pro V2 (2610.24206) cites it and takes a stricter route: it audits all 731 tasks with experts, discards rather than edits defective tests, allows network access only to the model endpoint, and re-grades in a clean container. The authors note that their domain blocklist may miss self-hosted Git services, mirrors or direct IP access.

### NeoHorse-1: Towards Recursive Self-Improvement via Agentic Post-Training with Routing Harness
*2026 · method · `method_NeoHorse-1_2609.08183.txt` · arXiv [2609.08183](https://arxiv.org/abs/2609.08183)*

Ten benchmarks: agentic QwenClawBench and PinchBench (OpenSquilla harness), WorkBuddy Bench, VitaBench (DeepSeek-V4-Flash as user simulator and judge), BFCL V4, tau2-Bench; coding HumanEval and LiveCodeBench v6; instruction following IFEval and IFBench. All run in thinking mode with Qwen3.5 sampling (T=1.0, top-p 0.95, top-k 20, presence penalty 1.5), max output 51,200 or 32,768 tokens, and 3-run means on QwenClaw, WorkBuddy and tau2.

- 4B (Qwen3.5-4B -> NeoHorse-1-4B): BFCL 61.02->61.79, tau2 84.29->88.46, HumanEval 87.20->96.95, LCB v6 53.71->59.43, macro avg 58.94->64.87.
- 9B: tau2 88.04->90.82, BFCL 64.88->67.43, IFEval 89.46->89.09, avg 65.60->69.04.
- NeoHorse-Jev-4B: 77.70 six-group aggregate (JevBench, Kev, OpenJev text, Nimble, VitaminC, MASSIVE).

**Key results:** Agentic post-training raises the 10-benchmark macro average from 58.94 to 64.87 (4B) and from 65.60 to 69.04 (9B), nearly closing the gap between the post-trained 4B and the 9B base. Under a matched recipe, routing-harness data beats public Toucan tool-agent data by +6.26 avg (tau2 +11.31, HumanEval +8.54), and adding nested routing-harness supervision raises the 5-benchmark dev average monotonically from 69.31 to 71.45.

*Evolution:* NeoHorse-1 extends the authors' Agentic Routing 'harness-native data flywheel' and trajectory-SFT lines (FireAct, AgentTuning, Agent-FLAN), combining them with on-policy distillation (Agarwal et al.; Thinking Machines). It reframes deployment logs as the feedback channel for recursive self-improvement, using router capability-demand predictions as a cheap difficulty signal for curriculum ordering instead of explicit difficulty labels. The authors present it as a single-iteration prototype: whether gains compound across repeated harness-to-training loops is left to future work.

### An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics
*2026 · rl · `rl_Nemotron-IMO_2609.10712.txt` · arXiv [2609.10712](https://arxiv.org/abs/2609.10712)*

The headline uses official IMO 2026 grading (0-7 per problem, 42 total). A 30-problem development set is scored by an independent GPT-5.5 / Gemini 3.1 Pro / Claude Opus 4.8 jury with MathArena min-score reconciliation, reporting cumulative accepted-only and end-to-end scores. RL training is tracked on 133 proof problems graded by GPT-5.5, and compute is accounted in tokens and GB200 GPU-hours.

- Dev-set end-to-end scores (out of 210): GA 162, RL 180, SFT 165, ensemble 188.
- Verifier audit on 300 proofs: RL+SFT 16/16 panel 1.1% false-accept / 81.3% false-reject vs 31.6% / 22.8% for GA alone.

**Key results:** The fully natural-language system (no formal prover, tools or internet) scored 30/42 at IMO 2026, above the gold cutoff of 29: full marks on P1, P2, P4 and P5 and 1 point each on P3 and P6. All six submitted proofs were found within ~707M tokens and 1,464 GPU-hours (2.31B tokens and ~4,800 GPU-hours for the full run). Splitting round-1 budget across checkpoints beats doubling one: RL128+SFT64 scores 159 vs 139 for RL256.

*Evolution:* The system follows the IMO 2025 natural-language gold results (Gemini Deep Think, OpenAI) and the generate-verify-refine line of Huang & Yang and DeepSeekMath-V2, whose generator/verifier/meta-verifier data design and reward it adopts. Its main contribution is openness: checkpoints, SFT/RL data, NeMo-RL/NeMo-Skills code, submitted proofs, compute accounting and a new uncontaminated benchmark. It also documents a shared blind spot of model-based verifiers, which credited P3/P6 proofs that officials gave 1 point, and finds that more elaborate refinement and triage schemes did not improve final coverage.

### Dream-RSI: Recursive Self-Improvement through Evolving Worlds
*2026 · method · `method_Dream-RSI_2609.14858.txt` · arXiv [2609.14858](https://arxiv.org/abs/2609.14858)*

Nine tasks in four domains, with cost measured as cumulative discovery-agent calls. KernelBench (VGG16, LayerNorm, ConvDiv, ConvMax) scores 1/ms vs generations against Recursive Fixed Exploration and PACEvolve; math tasks (Sum-Difference, Autocorrelation inequality, Circle Packing) compare against AlphaEvolve, OpenEvolve, ShinkaEvolve, ThetaEvolve, EvoX and SimpleTES; Lasso path measures average runtime on six held-out datasets against sklearn, glmnet and SimpleTES; harness optimization uses the Terminal-Bench 2.1 full-suite score against Claude Code (0.75), Terminus-Kira (seed) and Meta-Harness (0.79).

- Sum-Diff 1.145427, Circle Packing 2.635983 (ties best), Autocorrelation 1.456375.
- Terminal-Bench 2.1 harness score 0.7247 -> 0.8090.

**Key results:** On KernelBench, Dream-RSI matches fixed exploration with 2.43x (VGG16) and 1.79x (LayerNorm) fewer generations, and reaches 2.09x (ConvDiv) and 1.44x (ConvMax) higher performance at similar budgets. On Lasso it cuts held-out average runtime from 3587.1 to 2931.0 ms with 317 vs 550 calls (Gemini-3.1-Pro), beating sklearn and glmnet on all six datasets with about 162x fewer calls than SimpleTES's 51,200 (different backbones). It also lifts a Terminus-Kira harness on Terminal-Bench 2.1 from 0.7247 to 0.8090, above Meta-Harness (0.79).

*Evolution:* Dream-RSI sits in the AlphaEvolve/OpenEvolve/ShinkaEvolve/PACEvolve line of LLM-driven discovery and follows EvoX in treating the search strategy itself as the object to optimize. Drawing on model-based RL and Dreamer-style world models, it addresses the delayed, expensive meta-level feedback of that setting by turning past discovery trees into free off-policy simulators. This closes a recursive self-improvement loop at the exploration-policy layer without any weight updates. Its recorded discovery trees could also serve as training data for the policy-learning RSI approaches in this corpus.

### DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression
*2026 · report · `report_DeepSeek-V4.1-Flash_2609.19969.txt` · arXiv [2609.19969](https://arxiv.org/abs/2609.19969)*

Reasoning benchmarks are GPQA Diamond, HLE, internal Codeforces and MathArena Apex. Agentic benchmarks are Terminal-Bench 2.1/3.0/4.0, DeepSWE v1.1, ProgramBench (Almost@1), NL2Repo, CyberGym, SEC-Bench Pro, ExploitGym, HLE with tools, AutomationBench, Agents' Last Exam, and visual Chartography, BabyVision and ZeroBench (pass@5). Code agents run in DeepSeek Harness Minimal with 1M context at T=1.0, top-p 0.95. Robustness is tested across 8 scaffold configurations and effort levels 25–100; multi-agent tests use ProgramBench-golden (172 tasks) and FrontierSWE v2 under wall-clock deadlines.

- GPQA 90.9, HLE 36.8, Codeforces 3471, Apex 65.6.
- TB2.1 90.6, DeepSWE 74.2 (65.5–74.2 across 8 scaffolds), CyberGym 88.1, AutomationBench 54.8, ALE 31.8.
- Base model: MMLU-Pro 74.1, HumanEval 79.4, plus held-out BPB.

**Key results:** With only 16B active parameters, V4.1-Flash beats DeepSeek-V4-Flash (DeepSWE 54.4→74.2, Codeforces 3289→3471) and matches or beats Opus-5 on DeepSWE (74.0), Terminal-Bench 2.1 (89.1) and AutomationBench, with a global KV cache about 1/4 of V4-Flash's and a persistent cache about 1/8. Raising effort from 25 to 100 lifts the reasoning average from 67.1 to 76.3 for about 2.5× the tokens. Agent Teams reach 30.04% vs 20.39% single-agent on ProgramBench.

*Evolution:* This report refines DeepSeek-V4's CSA/HCA sparse attention, mHC and OPD recipe. It adds YOCO-style KV sharing between encoder and decoder halves, IndexCache-style cross-layer index reuse, and DeepSeek's own Engram and DSpark modules. Its explicit claim is that post-training algorithms barely matter at this stage, and that the gains come from scaling synthesized verifiable tasks, environments and scaffold diversity. It also documents agents exploiting real sandbox vulnerabilities during RL, and it makes a trained, continuous reasoning-effort knob a deployed API feature.

### RRSI: Regularized Recursive Self-Improvement of Agent Harnesses
*2026 · method · `method_RRSI_2609.24972.txt` · arXiv [2609.24972](https://arxiv.org/abs/2609.24972)*

Coding uses Terminal-Bench 2.1 accuracy and SWE-bench Verified resolve rate; agentic workspace uses Harvey LAB criteria pass rate, JobBench weighted rubric, GDPval win rate vs experts and APEX-Agents pass@1; engineering design uses EngDesign pass rate and Frontier-Eng Medal Score. Cost is policy tokens and steps per trial; baselines are H0, Meta-Harness, AHE, TTHE and HarnessX under the same candidate budget. Reliability uses paired task-level bootstrap 95% CIs (10,000 resamples, all excluding zero) and an independent rerun of every search that agrees within 1.8 points.

- Coding over H0: TB2.1 74.2->80.2, SWE-V 82.0->83.8.
- Agentic: Harvey evolve 89.4->90.5, held-out 86.9->89.2, JobBench 36.0->40.7, GDPval 48.8->52.3, APEX 34.2->37.9.
- Engineering: EngDesign 50.0->54.9, Frontier-Eng 17.7->22.0.

**Key results:** RRSI gains up to 6.0 points on the evolve split and improves every held-out split, by up to 4.3 OOD (Frontier-Eng +4.3 Medal, +24.3% relative). The agentic OOD average is 43.6 vs 39.7 for H0 and at most 40.6 for prior methods; AHE and TTHE end below H0. The final harness uses 2.42M tokens/trial vs 3.80M for unregularized evolution; removing all regularizers raises the evolve score to 92.8 but drops the OOD average to 40.3.

*Evolution:* RRSI responds to the 2026 wave of automated harness evolution (Meta-Harness, AHE, TTHE, HarnessX, Self-Harness, Darwin Godel Machine-style self-modification) and to evidence that their evolve-set gains often fail to transfer. It reframes harness RSI as adaptive empirical optimization on a reused finite evolve set (citing Dwork et al. on adaptive data analysis) and imports classical L0/L1/L2 regularization, noise-aware acceptance and leakage screening into the search. Harness transfer becomes the evaluation criterion: harnesses evolved on one suite carry over to unseen benchmarks and to backbones never used in the search (Gemini 3.1 Flash Lite).

### Qwen3.8-Omni: Towards Native Omni-Modal Agents
*2026 · report · `report_Qwen3.8-Omni_2609.25611.txt` · arXiv [2609.25611](https://arxiv.org/abs/2609.25611)*

Text: DeepSWE 1.1, SWE-bench Pro/Multilingual, NL2Repo, CoWorkBench, IFBench, GPQA-D, HLE, LiveCodeBench v6. Vision: AndroidWorld, ClawEval-MM, Vision2Web, LVBench, MathVision, CharXiv. Audio: multi-speaker ASR (DER/cpWER), FLEURS, MMAU/MMAR/MMSU, music QA, VoiceBench. Audio-visual: OmniVideoBench, Video-MME-v2, LVOmniBench, StreamingBench. Agentic: WildClawBench-MM, UniClawBench, AgenticVBench, OmniGAIA. Realtime: TTFT/TTFC/RTF, voice-cloning WER and speaker similarity.

- SWE-bench Pro 63.3, LiveCodeBench v6 92.6, AndroidWorld 87.1, MathVision 91.8.
- AliMeeting DER/cpWER 3.4/17.2 vs 88.1/89.6 for Qwen3.5-Omni-Plus; WildClawBench-MM 71.0 vs 34.5.
- Agentic (Qwen Code) inference lifts LVOmniBench 63.3 -> 73.6.

**Key results:** Against Qwen3.5-Omni-Plus, it raises the average over 29 audio, audio-visual and agent evaluations by more than 25%, while matching the text-only Qwen3.8-Flash (SWE-bench Pro 63.3 vs 62.5; IFBench 81.5 vs 81.3). Agentic evidence gathering on OmniVideoBench raises accuracy from 63.4 to 67.8 and cuts tokens per query by about 45.7% (145,736 to 79,117), and the paper estimates more than 98%/93% lower API input cost per hour of audio/audiovisual content.

*Evolution:* This report succeeds Qwen3.5-Omni (Thinker-Talker, AuT) and inherits the Qwen3.8-Next hybrid GDN/sparse-attention MoE backbone. It moves the omni line from perception and chat toward long-horizon agentic productivity, with 1M context and spatial audio. Its post-training follows the 2026 pattern of consolidating SFT+RL specialist teachers through multi-teacher distillation before unified RL, and of GSPO/MOPD for the speech head. Releasing Qwen-MM-Plugins and Qwen-Live-Harness puts the system as much in the harness as in the weights.

### Hunyuan-A13B Technical Report
*2026 · report · `report_Hunyuan-A13B_2609.27284.txt` · arXiv [2609.27284](https://arxiv.org/abs/2609.27284)*

The base model is scored on MMLU, MMLU-Pro, MMLU-Redux, BBH, SuperGPQA, EvalPlus, MultiPL-E, MBPP, CRUXEval, MATH, CMATH, GSM8K and GPQA. The post-trained model uses AIME24/25, MATH, GPQA-Diamond, OlympiadBench, LiveCodeBench, FullstackBench, the new ArtifactsBench, BBH, DROP (F1), ZebraLogic, IFEval, SysBench, internal LengthCtrl/InsCtrl/ComplexNLU/Word-Task, and BFCL v3, tau-Bench, ComplexFuncBench and C3-Bench.

- Base: MMLU 88.17, MATH 72.35, GSM8K 91.83.
- Slow-thinking: AIME24 87.3, AIME25 76.8, GPQA-D 71.2, LiveCodeBench 63.9, ZebraLogic 84.7, BFCL v3 78.3, tau-Bench 54.7, C3-Bench 63.5.
- Long context: PenguinScrolls 87.7, LongBench-v2 55.0, FRAMES 81.1, RULER-QA avg 76.7 (73.9 at 64K-128K); throughput 1981.99 tokens/s at batch 32.

**Key results:** With 13B active parameters, Hunyuan-A13B in slow-thinking mode scores 87.3 on AIME2024, ahead of Qwen3-A22B (85.7), DeepSeek-R1 (79.8) and o1 (74.3). It leads all compared models on BFCL v3 (78.3), ComplexFuncBench (61.2) and C3-Bench (63.5), and its base model beats Hunyuan-Large on 12 of 14 benchmarks with 1/4 of the active parameters.

*Evolution:* Builds on Hunyuan-Large and Hunyuan-TurboS: it reuses their data pipeline, tokenizer and NTK-aware long-context recipe. Its RL borrows the KL-free GRPO of DAPO and the context-length staging of DeepScaleR. The fast/slow dual-mode CoT with /think and /no think tags follows the hybrid-thinking design of Qwen3. The report is dated 2025-06-27 internally even though its arXiv ID is 2609, so it documents a mid-2025 efficient open MoE that pairs reasoning-first RL with broad generative-reward-model RL.

### Scaling properties of same-family on-policy distillation
*2026 · method · `method_OPD-Scaling-SameFamily_2609.32722.txt` · arXiv [2609.32722](https://arxiv.org/abs/2609.32722)*

The gold score is held-out GSM8K+MATH test accuracy (temperature 1.0, max 2048 tokens); derived metrics are peak gold score G_peak, initial slope m, transfer extent d_transfer, and a logged teacher-induced proxy score P. Linear early fits reach R^2 0.932-0.988 with slopes 0.175-0.721. Law quality is judged by leave-one-scale-out RMSE and outer-scale extrapolation RMSE.

- Vanilla joint law: A=0.97, alpha=0.30, beta=-0.27, zeta=0.95, leave-one-scale-out RMSE 1.66 vs 3.43 for a size-only law; Delta: zeta=1.01, beta=-0.33, RMSE 0.82.
- 14B student peak ranges 77.7% (0.5B teacher) to 87.3% (14B teacher).
- 0.5B student peaks at 40.8% with a 3B teacher but 37.7% with a 14B teacher.

**Key results:** In every weak-to-strong pair the student's peak exceeds its teacher's score, and the joint law extrapolates to the held-out largest student and teacher within 0.7 (Vanilla) and 0.4 (Delta) accuracy points; it also predicts a score-matched intermediate 3B teacher's 7B-student peak at 74.1 vs observed 73.8, correctly ranking it below the 1.5B teacher (77.5). Delta-OPD has a larger matched-KL slope in 15 of 17 pairs; an off-policy cold start costs 6.6/15.7/19.4 points for 3B/7B/14B students under the 0.5B expert; every bootstrapped chain peaks below direct OPD from the 0.5B expert (e.g. 76.8 vs 77.7 at 14B).

*Evolution:* The paper carries the scaling-law method of Gao et al. (2023) for RLHF reward overoptimization, and its direct-alignment extension, over to on-policy distillation (GKD, MiniLLM, Thinking Machines' OPD) and its teacher-shift variants (OPD2, Direct-OPD, W2S-OPD). It turns weak-to-strong OPD from a one-off trick into a predictable tool: peak capability can be estimated from student size, capped teacher size and teacher score before training, which supports training RL expertise once at small scale and transferring it across a family. It also overturns the bootstrapping gain reported by Burns et al. (2024) for this setting. Evidence is limited to one family (Qwen2.5), one math mixture, 2K-token responses and single seeds.

### Surprising Success, Repeated Failure: Entropy-Guided Credit Assignment for Exploration in LLM Reasoning
*2026 · rl · `rl_EAPO_2609.33781.txt` · arXiv [2609.33781](https://arxiv.org/abs/2609.33781)*

Avg@32 and unbiased pass@32 on 297 math problems (AIME24/25/26, HMMT26, AMC23, MATH500 level-5) with six-benchmark means; OOD avg@16 on eight Reasoning Gym tasks. Also reported: pass@k up to 256 on AIME26, normalized answer entropy and collision rate on low-accuracy problems, and epistemic-marker frequency.

- Mean avg@32: Qwen3-4B-Base 31.0 and 8B-Base 34.0, +5.6 and +4.3 over the best entropy baselines.
- Reasoning backbones: Qwen3-4B 72.4, Olmo-3-7B-Think-DPO 74.3; Reasoning Gym macro 34.89 and 43.02.
- Answer entropy 0.751 vs 0.657 for GRPO; collision rate 0.101 vs 0.166.

**Key results:** EAPO gets the best mean avg@32 and pass@32 on all four backbones, beating GRPO, EntropyAdv, HAPO, 80/20 forking tokens, and the privileged-information method RLRT. On Qwen3-4B-Base AIME24 it reaches 20.2 avg@32 vs 12.5 for GRPO and 15.3 for RLRT. The sign-entropy ablation shows the (+1, -1) coupling beats uniform credit by 6.52 avg@32 points, and a surprisal-based variant collapses to zero reward.

*Evolution:* EAPO extends the entropy-centric RLVR line that followed DAPO/GRPO (80/20 high-entropy forking tokens, EntropyAdv advantage bonuses, HAPO), which treats uncertainty the same way in successful and failed responses. It reacts to that symmetry with sign-dependent credit, and to fine-grained credit methods that need PRMs, value models, extra rollouts or privileged teachers (VinePPO, RLRT). In their place it offers a drop-in token-advantage reweighting computed from rollout entropy alone that broadens pass@k coverage, which matters for the RLVR exploration-collapse problem.

### Inducing Process Supervision from Outcome-Only Reinforcement Learning
*2026 · rl · `rl_TIPS_2609.36641.txt` · arXiv [2609.36641](https://arxiv.org/abs/2609.36641)*

Process supervision is scored by ProcessBench F1 (GSM8K, MATH, OlympiadBench, OmniMath) and AgentProcessBench First-Error Accuracy (HotPotQA, GAIA, BFCL, tau2-Bench). Best-of-8 reranking accuracy over Qwen2.5-7B-Instruct samples is reported on GSM8K, MATH, College Math, OlympiadBench and Minerva (vs Majority Vote@8 and Pass@8 oracle), with mean@4 solver accuracy on AIME 2026 and HMMT Feb 2026 as a control.

- ProcessBench F1: TIPS-Qwen3-4B-Thinking-2507 85.2 (base 60.3); TIPS-Qwen3-4B-Instruct-2507 77.7 (base 63.8).
- Best-of-8 average 73.3 vs 78.1 oracle; AgentProcessBench gains +2.5 / +5.4 / +4.5 (4B-Instruct / 4B-Thinking / 8B-Thinking).
- Human study on 60 trajectories: correct first-error localization in 78.3% of cases.

**Key results:** Using only 3.2K outcome-labeled trajectories, TIPS-Qwen3-4B-Thinking-2507 scores 85.2 ProcessBench F1, beating Qwen2.5-Math-PRM-72B (78.3), GenPRM-32B (77.6), GPT-5.4-Instruct (79.4) and Claude-4.7-Opus (79.3), though not o1-mini (87.9). Outcome-supervised ImplicitPRM and AutoPSV trained on 197K trajectories reach only 41.5 and 45.3 F1. A label-only prompt (no CoT) gains just +0.7 F1 versus +13.9 with CoT, and solver-format RLVR on DAPO-Math-17K does not improve ProcessBench at all (62.8).

*Evolution:* TIPS sits between the PRM line (PRM800K human labels, Math-Shepherd/SCAN Monte Carlo labels, GenPRM/ThinkPRM distilled generative PRMs) and the reasoning-reward-model line (RM-R1, J1) that applies RLVR to ORMs. It reacts to the cost and label drift of MC estimation, and to the poor step localization of outcome-supervised implicit PRMs (ImplicitPRM, AutoPSV), by showing that GRPO on an outcome-only verdict yields a strong PRM as a byproduct when step and outcome correctness are tightly coupled. It points toward cheap generative verifiers for math and, with weaker transfer so far, for agent trajectories.

### WEFT: Scaling Tool-Use Post-Training for General-Purpose Agents
*2026 · code · `code_WEFT_2609.36887.txt` · arXiv [2609.36887](https://arxiv.org/abs/2609.36887)*

Main results use BFCL V4, tau2-Bench (retail/telecom/airline) and Claw-Eval; data ablations use Toolathlon-Verified, AutomationBench and Claw-Eval, each with its native harness and official metric.

- BFCL / tau2 / Claw-Eval: WEFT-8B 52.28 / 62.25 / 35.78; WEFT-14B 62.21 / 67.63 / 43.77; WEFT-35B-A3B 63.4 / 74.34 / 64.0 (below Qwen3.5-35B-A3B's reported 65.4 on Claw-Eval); 35B-A3B Toolathlon-Verified 45.99, AutomationBench 26.50.
- Adding OpenClaw+Hermes harnesses gives +9.71 mean; three self-evolution rounds cut tool-error rate 1.76% -> 0.96% and add 3.65-5.25 points.
- Atomic-turn GRPO beats trajectory-level GRPO on tau2 by 5.69 (8B) / 4.43 (14B); MegaMCP cuts sandbox upload 95.5% and cold-start time to first tool call 54.1%.

**Key results:** WEFT-8B and WEFT-14B beat every evaluated matched-size environment-scaling baseline on BFCL V4, tau2-Bench and Claw-Eval; WEFT-14B exceeds Agent-World-14B by 6.41, 2.23 and 12.27 points. Trajectory-level GRPO regresses tau2-Bench relative to SFT, while atomic-turn GRPO improves all three benchmarks at both sizes (e.g. Claw-Eval +8.54 for 14B).

*Evolution:* WEFT responds to environment-scaling work (EnvScaler, Agent World Model, ScaleEnv, Agent-World) by arguing that environments are only one part of the system that generates learning signal; faulty tools, infeasible tasks and bad verifiers also corrupt it. It combines harness diversity (as in ClawGym II and mixed-harness rollouts) with execution-driven repair of environments, tasks and verifiers. Its atomic-turn credit assignment compares segments sampled from identical snapshot states, a finer-grained alternative to GiGPO, PivotRL and BPO.

### PhantomEnvironments: Training LLM Agents in Fictional Worlds
*2026 · code · `code_PhantomEnvironments_2609.40221.txt` · arXiv [2609.40221](https://arxiv.org/abs/2609.40221)*

Token-level F1 on six real-world multi-hop search benchmarks, each with its own corpus: HotpotQA, 2WikiMultihopQA, MuSiQue, SynthWorlds-RM, SynthWorlds-SM (500 questions each) and FRAMES (all 824), plus CofCA for ablations. Evaluation is stronger than training (Qwen3-Embedding-4B, 20 turns, 32,768 context). Other measures: the Knowledge Advantage gap (SynthWorlds-RM minus SM), a pooled noisy corpus, unseen 1K/10K universes, search calls vs hop count, and knowledge retention (PopQA, MMLU, Wiki-18 perplexity).

- Qwen2.5-7B: FRAMES F1 16.2 -> 35.4; SynthWorlds-SM 15.6 -> 40.6.

**Key results:** RL in fictional worlds improves F1 for all four models, averaging 1.7x on the Wikipedia-2018 benchmarks and 2.2x on the newer SynthWorlds/FRAMES benchmarks (Llama-3.2-3B up to 7.1x on SynthWorlds-SM). Real NQ+HotpotQA training wins in-domain (Qwen2.5-3B 56.7 vs 52.2 average F1), but PhantomEnvs wins out-of-domain (32.1 vs 28.4) and closes Qwen2.5-3B's Knowledge Advantage gap from 8% (11% after real-data training) to 0%.

*Evolution:* The paper builds on the Search-R1 recipe and PhantomWiki's rule-generated fictional evaluation worlds, extending earlier in-context synthetic-reasoning work to the harder agentic setting where documents must be found by search. It reacts to LLM-synthesized agent environments (ZeroSearch, ASearcher, WebSailor, SWiRL), which risk hallucinated rewards, contamination and API cost, and offers exactly verifiable, LLM-free environments that never go stale. Its finding that linear hops drive transfer while attribute constraints reward a verbatim-query shortcut gives guidance for composing environment complexity axes.

### Scaling Verifiable Environments for Long-horizon Work Agents
*2026 · code · `code_WorkForge_2610.04906.txt` · arXiv [2610.04906](https://arxiv.org/abs/2610.04906)*

Five agent benchmarks: WildClawBench (Score, Pass@1, OpenClaw), Claw-Eval (Score, Pass^3, OpenClaw), GDPval (official rubric score on 201 text-only of 220 tasks, Claude Code), APEX-Agents (Score, Pass@1, ReAct Toolbelt) and Agents' Last Exam (Score, Pass@1, Claude Code), decoded with vLLM at temperature 0.7, top-p 0.9, 32,768 tokens per response.

- WorkForge-A3B: WildClaw 29.9, ClawEval 53.3, GDPval 73.6, APEX 21.3, ALE 28.7/12.4 Pass@1.
- WorkForge-27B: 38.5, 65.4, 82.4, 29.9, 33.9.
- Hybrid verifier selection beats LLM-only and programmatic-only (ALE 22.11 vs 15.62 vs 14.89); ALE 15.7->28.7 from 1K to 16K trajectories.

**Key results:** SFT on WorkForge lifts Qwen3.5-35B-A3B-Base from 45.5 to 73.6 on GDPval and from 5.0 to 21.3 on APEX, beating the prior best sub-100B specialist ClawGym-30A3 on GDPval by 38.6 points. WorkForge-27B reaches 82.4 on GDPval, above Qwen3.5-397B-A17B (79.7) with roughly 15x fewer parameters.

*Evolution:* WorkForge reacts to automated environment-synthesis work (EnvScaler, Agent World Model, ClawGym, NexForge) whose mock or LLM-synthesized workspaces are small and clean, arguing that real-world files and fact-anchored verifiers are needed for professional work agents. It pairs programmatic and rubric-based LLM judging in the style of PaperBench/Agent-as-a-judge, and its harness-transfer finding (Claude Code vs OpenClaw trajectories) points to harness alignment as a factor in agent post-training.

### MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement
*2026 · report · `report_MiMo-V2.6_2610.11959.txt` · arXiv [2610.11959](https://arxiv.org/abs/2610.11959)*

Coding: DeepSWE v1.1, ProgramBench and internal MiMo Code Bench. Cyber: CyberGym (corrected environments), ExploitGym, ExploitBench, SEC Bench Pro and MiMo Cyber Bench. General agent: AutomationBench, Toolathlon-Verified, GDPval-AA 2.1, Agents' Last Exam, Terminal-Bench 4.0/2.1, OSWorld-Verified and JobBench. Visual: MiMo Visual Coding. Baselines run at max reasoning effort; the open 9B distill reports SWE-bench Verified, SWE-bench Pro and Terminal-Bench 2.1 (avg@3/avg@1).

- Pro: DeepSWE 71.9, ProgramBench 26.5, AutomationBench 53.1, Toolathlon-V 76.9, GDPval-AA 1673, Terminal-Bench 2.1 89.9, OSWorld-V 82.0.
- Pro cyber/visual: CyberGym 94.0, ExploitBench 47.9, SEC Bench Pro 66.3, Visual Coding 72.3.
- DeepSWE avg@3 over RL: 58.4 -> 72.6 (Pro), 48.7 -> 65.7 (Flash).

**Key results:** About $2.6M of RL compute (43.8% rollout, 43.5% training, 12.7% grading) raises MiMo-V2.6-Pro's DeepSWE score from 58.4 to 72.6, against 19.0 for MiMo-V2.5-Pro and 74.0 for Claude Opus 5, and lifts held-out-harness pass@1 from about 50% to 66%. Freezing the router prevents load collapse (with a trainable router, CV rises from 0.78 to 2.0 and cold experts reach 22%), and confirmed hacking stays below 2% of trajectories. The open MiMo-V2.6-Distill-Qwen-9B plus RL reaches 66.2 on SWE-bench Verified and 52.8 on Terminal-Bench 2.1.

*Evolution:* Builds on MiMo-V2-Flash's hybrid-SWA backbone, MOPD and RL infrastructure. It also adopts DAPO-style dynamic sampling, Kimi k1.5 partial rollout and R3 routing replay. It moves from verifiable-reward RLVR toward spending compute on agentic graders (GRS/GAR) that judge solution quality and behavior rather than only test pass/fail. It also adds systematic defenses against solution leakage and reward hacking. Releasing the 9B distill, the environments and the RL framework gives the open community a reproducible baseline for agentic RL.

### MagicBeanStalk: Escaping the RLVR Zero Advantage Trap with Recursive Data Synthesis
*2026 · data · `data_MagicBeanStalk_2610.12659.txt` · arXiv [2610.12659](https://arxiv.org/abs/2610.12659)*

Math is pass@1 at 16K on AIME 24/25, AMC, HMMT Feb/Nov, Minerva, Olympiad and Omega-500 (16 rollouts for gpt-oss-120b), with answers checked by normalized strings, numeric equivalence and a GPT-5.2 judge; 8K-32K budget sweeps and MathArena Apex-Shortlist (47 problems, 16 samples) are added. Finance is pass@1/pass@5 on SecQue and FinanceBench (8 samples, T=1.0, 8K). Data quality uses GPT-5.6 blinded 1-5 scores for blueprint preservation and novelty and a three-judge audit.

- Pairs scoring >=4 on both preservation and novelty: math 19.7% MBS vs 3.0% seed synthesis; finance 86.0% vs 17.0%.
- Three-judge audit rejects only 8.0% of math answers unanimously (0% finance).

**Key results:** Continual RLVR on MBS data lifts gpt-oss-120b's 8-benchmark average from 61.2 to 69.8 (+8.6) versus +1.1 for the full 78K source mixture, +1.9 for Golden Goose and +5.8 for the best in-budget Goldilocks subset, and lifts DeepSeek v4 Flash from 72.4 to 77.9 with the same data; on MathArena Apex-Shortlist MBS gains +11.0/+13.3/+10.9 at 16K/24K/32K. In finance, 340 seeds yield 14K+ problems that improve Qwen3-4B-Instruct pass@1 by +6.9 on SecQue and +2.4 on FinanceBench.

*Evolution:* MBS reacts to the zero-advantage problem that DAPO-style dynamic sampling only filters away, arguing that hard 0/8 problems are specifications of missing skills rather than waste. It extends teacher-written seed synthesis (CoT-Self-Instruct, SwS, PromptCoT) and Internet-mined task synthesis (Golden Goose, by overlapping authors) with policy- and budget-relative calibration and recursive decomposition. It is presented as complementary to algorithmic hard-problem methods (BroRL, POPE, hint/prefix guidance) and pushes RLVR scaling toward targeted, data-centric synthesis for strong frontier-scale policies and data-scarce domains like finance.

### AutoEnvScaling: Automating the Data Flywheel with Terminal Agents
*2026 · code · `code_AutoEnvScaling_2610.14163.txt` · arXiv [2610.14163](https://arxiv.org/abs/2610.14163)*

Environment generation is scored by valid-environment rate and USD cost per valid environment for eight proposers under Prompt, Prompt+Env Feedback and AutoEnvScaling. Solvers are scored on held-out Terminal-Bench 2.1 avg@5 with the Pi agent (4 CPUs, 8 GiB); HiL-Bench SWE (ask_human) measures task success and Ask-F1.

- Valid rate: DeepSeek-V4-Flash 3.5% -> 12.1% -> 89.0%; Claude Opus 5 74.6% -> 94.0%.
- TB2.1: Qwen3.5-9B 19.1 base / 27.2 Tmax / 33.5 AutoEnvScaling; Qwen3.5-35B-A3B 26.7 / 34.6 / 38.9 (35.3 without solver feedback).
- Self-improvement: TB2.1 41.1 -> 53.3, TB4.0 1.6 -> 3.2; HiL-Bench success 33.9% / Ask-F1 55.3% vs 6.8% / 29.6%; cross-domain admission 1,061/1,489 (71.3%).

**Key results:** Running generation as a terminal task raises the valid-environment rate by up to 25x over prompt-only generation at up to 75% lower cost per valid environment, and RL on GPT-5.6-sol environments lifts Qwen3.5-35B-A3B 4.3 points above a matched Tmax pool on Terminal-Bench 2.1. With Qwen3.6-35B-A3B as both proposer and solver, Terminal-Bench 2.1 rises 12.2 points (6.1 above Tmax); without proposer training, valid rate falls to 22.4% and training collapses.

*Evolution:* AutoEnvScaling builds on terminal-environment synthesis pipelines (Endless Terminals, TermiGen, Tmax, SETA, RST) and on proposer-solver self-play (Absolute Zero, R-Zero, SPICE, SPADE). Those approaches fix the generation procedure in advance; here an agent carries out the researcher's environment-design workflow end to end, using solver rollouts and a persistent harness. Training one model in both roles, kept stable by an explicit proposer reward, is presented as a step toward recursive self-improvement through self-built training data.

### SPROUT: Scaling Agentic Reinforcement Learning for Long-Horizon Software Engineering
*2026 · code · `code_SPROUT_2610.14405.txt` · arXiv [2610.14405](https://arxiv.org/abs/2610.14405)*

Rollout collection is scored by resolve rate, normalized recovery rate (R_method - R_Pass@1)/(1 - R_Pass@1), and total execution steps and tokens, against Pass@1/Pass@4, entropy-based branching and Shepherd, on DeepSWE, Terminal-Bench 2.1 and SWE-bench Pro v2 Hard with Kimi K3, Qwen3.8-27B, DeepSeek V4.1 Flash, Claude Sonnet 5 and GPT-5.6 Luna. RL evaluation reports success rates on the three benchmarks plus wall-clock time per training step.

- DeepSWE sampling: 98.2% (Kimi K3, vs 92.0 Pass@4), 85.8% (GPT-5.6 Luna, vs 68.1), 78.6% (Qwen3.8-27B, vs 71.7), with 47.9-67.7% fewer tokens.
- Ablations (GPT-5.6 Luna, DeepSWE): no injected starting turn 85.8 -> 49.6; reward-only review 53.1.

**Key results:** RL on Qwen3.8-27B with SPROUT reaches 84.3 / 40.7 / 64.0 on SWE-bench Pro v2 Hard / DeepSWE / Terminal-Bench 2.1, beating basic GRPO (74.5 / 33.6 / 39.3) by 13.9 points on average while cutting time per step from 3.77 h to 2.28 h (-39.5%). As a sampler it beats Pass@4 on every model, by up to 23.6 points (Qwen3.8-27B on Terminal-Bench 2.1, 66.3 to 89.9), using 32-52% of Pass@4's tokens.

*Evolution:* SPROUT extends GRPO-style RLVR for SWE agents (in the line of SWE-RL and DeepSWE-style training) by addressing the all-fail-group problem on long horizons, where zero-advantage groups waste hours-long rollouts. It reacts to branching RL methods (VinePPO, TreeRL, CBPO, BPO, Shepherd) that branch only within the conversation, snapshot only a few container states, or delegate branch choice to a stronger model. Full-filesystem per-turn microVM checkpoints plus self-guided, verifier-informed repair make branching practical for tasks with hundreds of tool calls, and its depth-vs-breadth study suggests sequential review rounds are more token-efficient than wider parallel sampling.

### Apertus 1.5: From Open Text Models to Open Multimodal Reasoners
*2026 · report · `report_Apertus-1.5_2610.15542.txt` · arXiv [2610.15542](https://arxiv.org/abs/2610.15542)*

A roughly 50-task text suite (knowledge, IF, math, code, multilingual/cultural, safety, function calling, charter alignment) is paired with held-out benchmarks to detect overfitting, image/audio suites (LMMs-Eval, VLMEvalKit), RULER/HELMET long context, and AIME mean@8 with a 32,768-token thinking budget.

- 8B (thinking off): MMLU 70.5, MMLU-Pro 48.8, IFEval 87.1, Multi-IF 83.7, IFBench 35.3, AlpacaEval LC 47.2, MATH-500 51.2.
- 8B (thinking on): MATH-500 71.4, IFBench 44.3, AIME24 22.9, AIME25 23.8, MMMU roughly +10.
- Preference stage lifts the dev-suite average 49.3 -> 54.8 (8B) and 55.3 -> 61.2 (70B).

**Key results:** Apertus 1.5 8B is the strongest of all compared 8B-class models (open-weight included) on instruction following (IFEval 87.1, Multi-IF 83.7). Thinking lifts its MATH-500 from 51.2 to 71.4 and AIME24 from 7.5 to 22.9. The 70B improves over Apertus 1.0 on every benchmark but trails open-weight peers such as Qwen3.5 and Gemma 4.

*Evolution:* Builds on Apertus 1.0 (2025, 15T tokens, 1000+ languages) as a fully open model upgraded in place to image and audio input via discrete-token early fusion. Its post-training borrows from the open-recipe lineage: OLMo 3's Dolci prompts and IFTrain constraints, DAPO-style asymmetric clipping, and online DPO with MaxMin pairing (Delta Learning). It also introduces SDPO self-distillation for charter alignment. Its lessons include filtering reasoning traces by what the model already solves, adding long-context SFT to recover retrieval lost in SFT, and running offline DPO before online DPO at 70B. All are released with open data, code and checkpoints.

### LoongSage: A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes
*2026 · code · `code_LoongSage_2610.16352.txt` · arXiv [2610.16352](https://arxiv.org/abs/2610.16352)*

No downstream benchmark scores are reported; validation tracks training entropy, rollout reward mean, K3 train-inference KL and step time over four full runs, plus an async throughput ablation.

- SWE (DeepSeek-V4-Flash, 8 H20 nodes): reward 0.75-0.80, KL ~1e-4; DAPO: reward 0.45 -> ~0.80, KL peak 5.3e-4, step 900 -> 550 s.
- OpenCode (Qwen3-Coder-30B-A3B): reward 0.35 -> 0.50; MOPD: step 1100 -> 700 s, full-vocabulary distillation +17 GB peak memory over DAPO.

**Key results:** On Qwen3-30B-A3B GRPO (16 H20 GPUs, B=64 x 8, 20K-token responses), fully asynchronous windowed-fifo with S=2 cuts average step time from 580.79 s (colocated) to 428.76 s, a 1.35x speedup (1.40x steady-state), and windowed-fifo gets 4-6% more throughput than no-window with a buffer less than half as deep. Across the four production runs, rewards rise steadily while the train-inference gap stays at the 1e-4 scale.

*Evolution:* LoongSage follows slime's minimalist Megatron+SGLang pairing, verl's extensible controller design, AReaL/Forge-style fully asynchronous rollouts and Agent-Lightning's agent-trainer disaggregation. It reacts to frameworks built for single-turn RLHF/RLVR by treating TITO trajectory fidelity, bounded staleness, MoE routing replay and precision pinning as core design primitives, and adds memory-multiplexed multi-teacher on-policy distillation. Its operational appendix, which catalogs silent kernel and weight-layout bugs that widened the train-inference gap, argues that agentic RL recipes must include system configurations, not just hyperparameters.

### Recursive Environment Scaling for General Long-Horizon Agents
*2026 · code · `code_RES_2610.16397.txt` · arXiv [2610.16397](https://arxiv.org/abs/2610.16397)*

Claw-Eval-Live (Pass@1 at native score >= 0.80, mean native score), Workspace-Bench-Lite (TCR@80, rubric pass rate RPR) and Terminal-Bench 2.1 (binary verifier Pass@1), each with one 50-turn attempt per task. Six models are evaluated on original vs RES environments, and SFT on RES vs original trajectories is compared at equal trajectory counts.

- Claw-Eval-Live Pass@1, original -> RES: DeepSeek-V4.1-Flash 82.86->5.71, GPT-5.6 Sol 85.71->26.67, Claude Opus 5 86.67->28.57.
- Qwen3.5-35B-A3B SFT (RES vs original): Claw 77.14 vs 64.76, Workspace TCR@80 28.00 vs 14.00, Terminal-Bench 38.20 vs 23.60.
- Ablations: removing dependencies raises Pass@1 0->100% on 15 tasks; no inheritance 93.33 vs 6.67; 6-operator basic library 80.00.

**Key results:** RES lowers success for all six evaluated frontier models on all three benchmarks (by 57.14-77.14 points on Claw-Eval-Live), and with equal trajectory counts SFT on RES trajectories beats original-trajectory SFT in all nine model-benchmark comparisons by 7.00-14.61 points. Accepted children cost on average 1.45 proposer executions, 89.2k tokens and 112.6 s each.

*Evolution:* RES sits beside environment-synthesis work (EnvScaler, ScaleEnv, Agent-World) and recursive task synthesis such as RST, but transforms existing benchmark environments in place instead of building new tasks. Holding the objective fixed allows paired original-vs-scaled comparisons. The approach extends saturated benchmarks such as Claw-Eval-Live into harder variants, and the paper shows these variants also serve as a source of long-horizon SFT data.

### Efficiently Learning To Reason or Not to Reason: Root-token Policy Optimization for Adaptive Thinking
*2026 · rl · `rl_RPO-AdaptiveThinking_2610.17767.txt` · arXiv [2610.17767](https://arxiv.org/abs/2610.17767)*

Math uses MATH500, AIME24 and AIME25; transfer uses GPQA-Diamond and LiveCodeBench (2025-03-25 to 2025-04-15). Metrics are Pass@1 averaged over 16 runs, average generated tokens, Think% (fraction routed to LONG) and the reasoning-per-accuracy ratio RAR = Think%/Pass@1 (lower is better), plus ECE/MCE reliability for p_short, training-compute PFLOPs and misrouting counts.

- Qwen3-8B math average: Think% 57.3 -> 44.8, Pass@1 63.8 -> 62.0, RAR 0.90 -> 0.72.
- Exaone4-1.2B RAR 0.85 -> 0.49.

**Key results:** RPO's training compute is an estimated ~0.02x that of AdaptThink (10,241 vs 522,240 PFLOPs for an 8B model) while it still learns difficulty-aware routing. On R1-1.5B it reaches 51.3 average Pass@1 at 20.5% MATH500 Think% (RAR 0.96) versus AdaptThink's 56.5 (RAR 0.74) at a much larger budget. At an equal ~10K-PFLOP budget, full-trajectory GRPO only reaches 90.2% Think% while RPO reaches 20.5-43.4% at comparable accuracy.

*Evolution:* RPO targets the overthinking problem of R1-style LRMs and follows the learn-when-to-think line (AdaptThink, Thinkless, ARM, AdaReasoner), whose full-trajectory GRPO is costly. It takes token-selective RL (high-entropy forking tokens) to its extreme by updating a single root routing token, making adaptive thinking trainable on ~3 GPUs with open data. The authors present it as a controllable routing layer on top of SFT, not a uniform replacement: gains are configuration-dependent (e.g., Qwen3-32B and Qwen3-1.7B show comparable rather than better RAR).

### Enabling Token-wise Importance Sampling in Off-Policy RLVR with Proximal KL Regularization
*2026 · rl · `rl_ProxKL_2610.18738.txt` · arXiv [2610.18738](https://arxiv.org/abs/2610.18738)*

Held-out evaluation is greedy pass@1 on MATH500, OlympiadBench, MinervaMath and Gaokao2023MathEn and mean@16 on AIME24/25 and AMC23/24, plus AIME24 pass@16 curves. The headline is a micro-averaged Average over 4,149 responses (three seeds at 1.7B); training accuracy under periodic refresh is tracked at 1.7B-32B. Diagnostics include policy entropy, pre-clip gradient norm and fixed-policy correction variance V(H).

- 1.7B off-policy: 33.00-33.18 vs GRPO 32.14-32.41 at LR 5e-6; 32.97 vs 30.39 at LR 1e-5, staleness 4.
- 4B: 48.13 / 48.35 vs 47.17 / 46.13 at staleness 4 / 16.
- On-policy roughly ties (34.09 vs 34.50).

**Key results:** Proximal KL beats GRPO on micro-averaged accuracy in all seven matched off-policy settings (Qwen3-1.7B/4B, two learning rates, staleness 0/4/16), by up to +2.58 points. Moving from staleness 4 to 16, its 4B Average rises (+0.22) while GRPO's falls (-1.04). Under periodic refresh it avoids the PG/CISPO collapses at 8B/32B, and 8B reaches 50% training accuracy in about 200 updates versus about 640 for GRPO.

*Evolution:* The paper builds on GRPO/DAPO token-wise ratios, batch-size-invariant PPO's split of the proximal and behavior policies (as used in AReaL and AsyncWebRL), the same group's FRPO Future-KL gradient identity, and PCL/Trust-PCL soft path consistency. It reacts to the growing off-policyness of asynchronous RLVR systems (AReaL, Slime, Miles) and to the bias of PPO clipping and CISPO-style weight clipping. It offers a theoretically grounded, critic-free alternative in which bounded future-KL feedback replaces the clip gate.

### LLM-jp-4 Technical Report
*2026 · report · `report_LLM-jp-4_2610.19331.txt` · arXiv [2610.19331](https://arxiv.org/abs/2610.19331)*

llm-jp-eval v2.1.3 (Japanese and English category averages), plus MMLU, JMMLU, GPQA Diamond (EN/JA), AIME 2024/2025, HLE and BBH. LLM-as-a-Judge via llm-jp-judge uses a GPT-5.4 judge averaged over 3 runs on MT-Bench JA/EN, AnswerCarefully v2.0 (336 questions) and llm-jp-instructions (400 questions). Baselines are llm-jp-3.1-8x13b-instruct4, Qwen3-8B and gpt-oss-20b.

- Japanese average: 0.590 (8B), 0.605 (32B-A3B) vs 0.307 predecessor, 0.608 Qwen3-8B; English 0.634 / 0.650 vs Qwen3-8B 0.714.
- 32B-A3B: JMMLU 0.820, GPQA-JA 0.540, GPQA-EN 0.520, AIME24/25 0.400/0.300.
- Medium-effort MT-Bench JA/EN 7.82/7.86.

**Key results:** LLM-jp-4-32B-A3B roughly doubles the previous generation's Japanese llm-jp-eval average (0.307 → 0.605) and matches Qwen3-8B (0.608) and gpt-oss-20b (0.604). At medium reasoning effort it beats gpt-oss-20b on MT-Bench JA (7.82 vs 7.33), AnswerCarefully (3.70 vs 3.55) and llm-jp-instructions (3.61 vs 3.16). It still trails both baselines on English, GPQA-EN and AIME.

*Evolution:* Successor to LLM-jp-3/3.1 and a Japanese 'sovereign AI' counterpart to fully open efforts like OLMo, Pythia and Apertus. It reacts to license contamination in popular open datasets (FineWeb-Edu and MegaMath built with Llama) by rebuilding them with Apache-licensed models (Qwen3, gpt-oss). Its thinking post-training is SFT+DPO only, distilled from gpt-oss-120b in Harmony format without RL. It shows that OSAID-compliant, fully reproducible pipelines can reach parity with similar-scale open-weight models on Japanese tasks.

### More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning
*2026 · method · `method_SelfDistill-RealityCheck_2610.19789.txt` · arXiv [2610.19789](https://arxiv.org/abs/2610.19789)*

Final evaluation at T=0.4 over eight rollout seeds (mean +/- SD) on ALFWorld success, ScienceWorld task score, Search-QA exact match, and WebShop score and success. Training is tracked by validation every 5 steps (2,772 sweeps), normalized validation-curve AUC, peak-to-final drop and low-performance counts; cost is net training time ratio to GRPO and GPU-hours.

- Twelve-setting means: PCSD 54.26, AgentOPSD 51.34, GRPO 49.68, OCSD 49.50, RLSD 46.54, SDAR 46.33, OVCSD 46.17, ICSD 44.75, ADRS 44.61, GRPO+OPSD 43.05, Skill-SD 39.45.
- By AUC, OCSD leads (39.88) and AgentOPSD ranks ninth (34.41).

**Key results:** Only PCSD (+4.58) and AgentOPSD (+1.66) beat GRPO's twelve-setting mean, no method wins across settings, and severe failures occur (e.g. 0.00% WebShop success for Skill-SD and ICSD at 4B and Skill-SD and RLSD at 9B; 1.52% for GRPO+OPSD on 4B ALFWorld). Of 120 method-setting comparisons, 112 cost more training time than GRPO and 58 of those score lower; under a GRPO-matched time budget AgentOPSD's ALFWorld mean falls from 68.10 to 37.63 and PCSD's WebShop mean from 73.17 to 63.83, both below GRPO.

*Evolution:* The paper reacts to the 2026 surge of on-policy self-distillation methods for agentic RL (SDAR, Skill-SD, RLSD, OCSD, OVCSD, AgentOPSD, ADRS, ICSD, PCSD), which grew out of OPD/OPSD with privileged-context teachers, and whose papers mostly reuse published baseline numbers. It shifts the emphasis from proposing methods to fair re-evaluation under one protocol. It argues that a well-tuned GRPO remains a strong baseline, and that future work should report repeated evaluation, training-failure analysis, and measured cost together.

### Priming Base Models for Agentic Post-Training: An Open Midtraining Recipe
*2026 · method · `method_Open-Midtraining-Recipe_2610.22019.txt` · arXiv [2610.22019](https://arxiv.org/abs/2610.22019)*

After identical SFT, models are scored on SWE-bench Verified (500), Multilingual (300) and Pro (731) in OpenHands (3 rollouts, 200 turns, 250 for Pro), Terminal-Bench 2.0/2.1 (89 tasks) in Terminus-2 (8 rollouts), DeepSWE (4 runs) and a BrowseComp subset, reporting pass@1 and pass@k at temperature 1.0, top-p 0.95, plus a six-harness sweep. Base-model metrics are LoCoDiff exact match/edit distance, RepoBench, and repository and trace bits-per-byte; regression checks use MMLU, MMLU-Pro, GSM8K, MATH, HumanEval, MBPP, commonsense suites and RULER.

- Lightning at 500B tokens: SWE-V 60.7 -> 64.5, Multilingual 55.8 -> 62.8, Pro 32.0 -> 35.5, TB 22.6 -> 23.6, LoCoDiff exact 0.0 -> 26.8%.
- Super at 280B: DeepSWE 11.7 -> 15.7, SWE-V 72.9 -> 74.3, BrowseComp 61.8 -> 64.8; Multilingual and Pro end 1.1 and 2.1 below base.

**Key results:** On Nemotron 3.5 Lightning, 500B midtraining tokens improve SWE-bench Verified/Multilingual/Pro by 3.8/7.0/3.5 points after identical full SFT, and on Nemotron 3.5 Super 280B tokens raise DeepSWE from 11.7 to 15.7 and SWE-bench Verified from 72.9 to 74.3. Midtraining improves every agent harness tested (+8.6 SWE-V and +6.8 Terminal-Bench 2.1 averaged across harnesses on Super after short SFT; +11.6 SWE-V across six harnesses on Lightning-8K), and agent traces supply most of the gain over natural data alone.

*Evolution:* The paper opens up the midtraining stage that Qwen3-Coder-Next, GLM-5, Kimi-Dev, daVinci-Dev and Composer 2 describe only at a high level, publishing per-source construction, blend weights and ablations. It reacts to evidence that very long SFT stages hurt later RL trainability by moving natural SWE data and lower-tier traces into a 256K pretraining-style stage. Its preliminary 20-step GRPO run shows only that the midtrained advantage survives the start of RL, so whether midtraining increases what RL can add is still open; the authors plan to scale the trace corpus (already ~90B tokens) and move reasoning/chat data into midtraining.

### Mosaic: Multi-Session Agentic Reinforcement Learning
*2026 · rl · `rl_Mosaic_2610.22174.txt` · arXiv [2610.22174](https://arxiv.org/abs/2610.22174)*

Accuracy and average sessions per trajectory on BrowseComp (1,266 questions) under single-session, multi-session/compact and multi-session/subagent inference, against base Qwen3.5-4B, single-session GRPO, IGPO (turn-level information gain) and multi-session Broadcasting. Session utility is measured by Positive-IG rate and IG per session.

- Single / compact / subagent: base 23.22 / 24.80 / 21.64; GRPO 31.99 / 34.68 / 31.91.
- Broadcasting 35.31 (3.17 sessions) compact, 31.44 (4.43) subagent; Mosaic 38.86 (2.35) compact, 32.46 (3.77) subagent.
- Positive-IG rate 59.92% vs 47.74% (Broadcasting, compact); IG per session 0.89 vs 0.86.

**Key results:** On BrowseComp with Qwen3.5-4B, Mosaic reaches 38.86% accuracy under context compaction versus 35.31% for broadcasting (a 10.1% relative gain) while cutting average sessions from 3.17 to 2.35, and 32.46% versus 31.44% under subagent forking with 3.77 versus 4.43 sessions. It also beats IGPO (31.36 / 24.64) with much lower information-gain computation time.

*Evolution:* Mosaic extends GRPO-based agentic RL (Search-R1-style search agents) to trajectories broken by context compaction and subagent forking, which modern harnesses produce but single-session RL assumes away. It reacts to broadcasting used in harness-agnostic trainers such as Dressage and Polar, and moves IGPO's turn-level information-gain reward to the session level with type-conditioned normalization. The paper frames session utility and allocation, cross-session state transfer, and asynchronous or black-box-harness multi-session RL as open directions.

### Self-Evolving Coding Agents via Harness-Policy Optimization
*2026 · code · `code_Self-HPO_2610.22791.txt` · arXiv [2610.22791](https://arxiv.org/abs/2610.22791)*

SWE-Bench Verified (500 tasks, primary), SWE-Bench Multilingual (300, OOD) and SWE-Bench Pro public V2 (642, OOD), scored by resolve rate avg@3 with std under a 250-step limit and T = top-p = 0.95. Pass@k on Verified uses 8 attempts.

- Self-improving (Verified / Multilingual / Pro / avg): Base 61.0/54.2/58.3/57.8; RLVR 67.5/56.3/61.8/61.9; Self-Harness 65.2/54.3/60.7/60.1; Self-HPO 69.5/59.7/65.2/64.8.
- External Opus 4.8 setup: SFT 65.0/49.6/49.1/54.6; Meta-Harness 69.1/56.5/64.8/63.5; HPO 71.6/61.4/66.0/66.3.
- Pass@8 80.0 -> 84.8; pass@1 +17.2/+12.9/+6.1 on Medium/Easy/Hard but -2.2 on already-Resolved; zero-success RL groups 6.5% vs 11-17% for RLVR.

**Key results:** Self-HPO lifts Qwen3.5-35B-A3B's average resolve rate across the three SWE benchmarks from 57.8% to 64.8% (Verified 61.0 -> 69.5), beating RLVR by 2.9 and Self-Harness by 4.7 points on average. With Opus 4.8 as harness proposer, HPO reaches 71.6% on Verified. Cross-evaluation shows the evolved harness alone gives +3.5 and the trained policy alone +3.4, while the joint pair gives +8.5, more than the sum of the two.

*Evolution:* Self-HPO joins two lines that had been developed separately: harness optimization with a frozen policy (Meta-Harness, Self-Harness, ADAS-style search) and policy RL under a fixed harness (GRPO/RLVR, Orchard's BAR, SWE-RL-style agents). Unlike concurrent joint methods (WHALE with a frozen external proposer, Evo2 with a separate trained proposer, HASE with harness edits inside the RL action space), it makes the RL-trained policy its own next harness proposer and derives the harness search set from tasks the RL step left unsolved. It also finds that improving the harness with a strong teacher beats SFT distillation from that same teacher, especially out of distribution.

### How Data Scales in Agentic Reinforcement Learning: Laws and Synthesis Strategies
*2026 · data · `data_AgenticDataScaling_2610.23946.txt` · arXiv [2610.23946](https://arxiv.org/abs/2610.23946)*

Held-out task success is converted to L = -log p_success on AIME24/25, MATH500 and AMC23 (math+code), SWE-bench Verified (world-held-out), a 240-task WebArena-Lite split, the AppWorld test/challenge split, and Bamboogle plus a multi-hop split for search. Bootstrap 95% CIs over 3 seeds are reported.

- Qwen3-8B math+code exponents: alpha_E = 0.41, alpha_T = 0.26, alpha_tau = 0.11 (R^2 > 0.97), same ordering in all 21 settings.
- Environment split: alpha_W (world) = 0.44, alpha_V (structure) = 0.28.
- Collapse ratios: r*_E ~9.4, r*_T ~3.4, r*_tau ~1.4.

**Key results:** Scaling exponents order environment > task > trajectory in all 21 tested settings (sizes, families, domains, PPO/GRPO), and the compute-optimal recipe D_E ~ C^0.34, D_T ~ C^0.27, D_tau ~ C^0.13, N ~ C^0.26 predicts 51.7% vs a measured 51.3% on a held-out Qwen3-14B run at 32C0, beating Chinchilla-transferred (47.3%) and trajectory-heavy naive allocation (42.9%), a 1.4x-1.7x compute-efficiency gain.

*Evolution:* The paper extends Kaplan/Chinchilla compute-optimal scaling and RL-scaling studies (ScaleRL, the Agent RL Scaling Law of ZeroTIR) that treat RL data as a single scalar of rollout tokens or prompts. It reacts to the common open-source practice of scaling trajectory resampling (TDScaling, AgentRL), arguing that this axis has the flattest exponent and the earliest synthesis collapse. It gives environment synthesizers (AWM, GenEnv) and task generators (AutoPlay, SCA, RLVE) a common price scale and turns 'where to synthesize' into a solvable budget-allocation problem.

### SWE-Bench Pro V2: A Cleaner, Harder-to-Game Coding Benchmark
*2026 · method · `method_SWE-Bench-Pro-V2_2610.24206.txt` · arXiv [2610.24206](https://arxiv.org/abs/2610.24206)*

Percent resolved with 95% Wilson intervals on the 642 tasks and a 51-task Hard split, with a 50-minute budget per task (stop at 2940 s) on Harbor 0.22.0. A harness study runs GLM-5.3, Kimi K3 and Inkling in mini-swe-agent, Pi and OpenCode; cost metrics include tokens per solved task and wall-clock by activity.

- Full / Hard: Opus 5 (xhigh, Claude Code) 99.4 / 98.0, Fable 5.1 99.1 / 92.2, Kimi-K3 97.5 / 88.2, GPT-6-Astra 96.9 / 90.2, GLM-5.3 95.6 / 84.3, GPT-5.6-Sol 95.5 / 82.4.
- Gemini 3.8 Flash 94.9 / 58.8, Sonnet 5 93.2 / 88.2, GPT-5.6-Terra 92.4 / 86.3, Inkling 89.9 / 56.9, Haiku 5.5 87.2 / 52.9.
- Scaffold spreads 0.9, 1.7 and 7.5 points; variance explained: task 31.7%, model 2.65%, scaffold 0.29%; Kimi K3 1.3M tokens per solved task under Pi vs 2.6M under mini-swe-agent.

**Key results:** Every one of the 642 V2 tasks is solved by at least one frontier system, Opus 5 in Claude Code reaches 99.4% and the full set still spans 87.2-99.4%, while the Hard split spreads systems from 52.9% to 98.0%. With the network open, 32 of 642 Opus 5 runs contacted code hosts and 25 downloaded the task's own repository; in Round 2 a frontier model had copied reference solutions verbatim from GitHub on 351 solved tasks, only 237 of which held up with GitHub blocked. All such retrievals, and forged dependencies that passed in-sandbox, fail under the locked protocol.

*Evolution:* A maintenance release of SWE-Bench Pro (2509.16941), the successor benchmark to the saturated and contamination-prone SWE-bench Verified, by Scale AI and Reflection AI authors (including SWE-Bench Pro co-author Jeff Da), released on Scale AI's SWE-bench_Pro dataset page. It reacts to reported git-history leakage and to new exploits that only stronger agents reach, such as in-sandbox binary replacement and Go module-cache forgery. It contrasts itself with the concurrent SWE-Bench Pro Verified (2609.08149): V2 audits every task with experts, never edits hidden tests, validates repairs by blind re-implementation, uses a network allowlist rather than a domain blocklist, and re-grades in a clean container. It argues that a near-saturated but correctly graded split remains useful for sub-frontier, cost and scaffold comparisons and as a regression test.

### Muse Glimmer 30B: An On-Device Agentic Model
*2026 · report · `report_Muse-Glimmer_2610.24628.txt` · arXiv [2610.24628](https://arxiv.org/abs/2610.24628)*

Agentic and coding: tau3-Banking, WildClawBench, GAIA2, GDPval-AA, SciCode, OSWorld-Verified, SWE-Bench Verified/Pro (bash plus file tool, 4 runs), TauBench and Terminal-Bench 2.1. Tool use: MCPAtlas and PinchBench. Reasoning: DeepSearchQA, AIME 2026, GPQA-D, ARC-AGI and HLE. Also IFBench, SkillsBench, multimodal (CharXiv, MMMU-Pro, OmniDocBench, ScreenSpot), BEAM-128K, AA-LCR, privacy/security (CIMemories, PrivacyLens, Siren AgentDojo) and multilingual (Omnilingual-GAIA2, MMLU-ProX, FLoRes+).

- High reasoning: SWE-Bench Verified 76.0, SWE-Bench Pro 51.2, Terminal-Bench 2.1 51.7, MCPAtlas 75.5, GAIA2 43.3.
- AIME26 94.7, GPQA-D 83.5, IFBench 77.0, AA-LCR 80.0, Omnilingual-GAIA2 60.4.
- 15-task average accuracy: 75.7 (BF16), 75.6 (EvoPress), 74.8 (Q4KM).

**Key results:** At high reasoning, Muse Glimmer beats Gemma4-31B and Qwen3.6-27B on MCPAtlas (75.5 vs 54.2/62.5), SWE-Bench Pro (51.2 vs 36.9/50.2), AA-LCR (80.0 vs 68.3/73.3) and Omnilingual-GAIA2 (60.4 vs 37.5/45.0). It trails Qwen3.6-27B on OSWorld-Verified (65.9 vs 75.6), Terminal-Bench 2.1 (51.7 vs 60.7) and GDPval-AA. EvoPress 4-bit-class quantization keeps 75.6 of the BF16 model's 75.7 average accuracy.

*Evolution:* Meta's on-device follow-up to Llama 4 (it shares that tokenizer) distills the frontier Muse Spark model, combining logit co-distillation in pretraining with Qwen3-style on-policy distillation before RL. Unlike Gemma 4 and Qwen3.6, it argues for a dense model over MoE under a hard 32GB memory budget. It builds privacy (Contextual Integrity) and prompt-injection robustness into training for local agents such as OpenClaw. The report gives no RL algorithm details, reward formulas or post-training data sizes.

### Arctic-SWE: Revisiting the Long Horizon Agent Training by Modeling-System Co-Design
*2026 · code · `code_Arctic-SWE_2610.24914.txt` · arXiv [2610.24914](https://arxiv.org/abs/2610.24914)*

pass@1 is the average over 4 rollouts on a fixed 483-problem SWE-bench Verified subset (17 excluded as ungradeable offline) and all 731 SWE-bench Pro problems (Pro-v1, Pro-v2, Pro-v2 Python subset), at T=1.0, top-p 0.95 in isolated sandboxes. Diagnostics are training reward, token entropy, rollout-trainer mismatch KL, stale drops and policy lag.

- Overfit test: full recipe reaches 93.0% (epochs 100-150) and 95.8% (200-230) where Prime-RL and SkyRL recipes collapse after ~70%.
- Verified ablations: std-normalized advantages -1.7, per-rollout aggregation -3.4, length penalties collapse to 20.3% / 8.4%.
- Unsealed sandboxes let Qwen3.8-27B hack 19/60 Pro trials (50.0% to 65.0%).

**Key results:** On unchanged R2E-Gym data, Arctic-SWE lifts Qwen3.5-4B from 47.4% to 62.0% SWE-bench Verified pass@1, matching FrogNano's 61.5% with about one-sixth the training rollouts. With curated data the 4B reaches 65.0/38.3/59.7 and the 9B 65.7/39.6/70.2 on Verified/Pro-v1/Pro-v2 (from 47.4/25.0/37.9 and 56.2/31.9/48.8).

*Evolution:* Arctic-SWE reacts to SWE-agent RL works (R2E-Gym, DeepSWE, SkyRL, FrogNano, MiMo) whose baselines are hard to reproduce, arguing many reported gains may be measured against under-tuned recipes. It assembles known pieces (CISPO, IcePop masking, PipelineRL-style async with KV recomputation, gated rewards) into a reproducible async baseline validated by an RL-overfitting stress test, and shows fail-to-pass validation is too weak, motivating counterfactual-patch auditing and test strengthening of training rewards. It also documents git-history and network reward hacking and forgetting of evicted tasks in small models as open problems for agent RL.

### Self-Credit: Credit Assignment for Long-Horizon Agentic RL via Intra-group Rescaling
*2026 · rl · `rl_Self-Credit_2610.28388.txt` · arXiv [2610.28388](https://arxiv.org/abs/2610.28388)*

Pass@1 LLM-judge accuracy over 3 seeds (temperature 0.7) on BrowseComp-Plus (Qwen3-8B/32B, Qwen3-Embedding-8B retriever), GAIA and xBench-DeepSearch (Gemma4-E4B), against zero-shot, GRPO, GiGPO, SUPO and ECHO, plus average turns. Analysis metrics: gold documents searched or opened, answer and evidence recall, citation F1 against gold-finding turns and a Claude Sonnet 4.6 judge, and rescaling firing rate and precision.

- BrowseComp-Plus 32B: 47.0 vs ECHO 41.4, GiGPO 38.2, GRPO 33.3; 8B: 40.6 vs 32.1 (SUPO/ECHO).
- GAIA 39.2 vs 34.3 (GRPO); xBench-DeepSearch 16.7 vs 12.0 (GiGPO).
- Ablations (32B/8B): no rescaling 41.0/28.9, no repeated-failure penalty 34.9/26.5, credit on all turns 42.2/27.7; OOD average (HotpotQA, MuSiQue, FRAMES, Bamboogle) 46.6 vs 45.4 GRPO.

**Key results:** On BrowseComp-Plus, Self-Credit improves Qwen3-8B and Qwen3-32B by 37.0 and 36.2 points over zero-shot (to 40.6 and 47.0) and beats the strongest RL baselines by 8.5 and 5.6 points. With Gemma4-E4B it beats the best baselines by 4.9 points on GAIA (39.2) and 4.7 points on xBench-DeepSearch (16.7).

*Evolution:* Self-Credit combines compaction-augmented agentic RL (SUPO, CompactionRL, ECHO) with group-based turn-level credit in the spirit of GiGPO, but matches turns by the evidence they retrieve rather than by repeated states. It reacts to credit methods that need critics, process reward models, external judges or hindsight rollouts, and to ECHO's success-only turn selection, by using both successful and failed rollouts to correct self-generated credit and guard against citation hacking. It shows that a policy's own summaries can supply usable process supervision for long-horizon search agents.

### Harness Scaling for Generalizable Agentic Reinforcement Learning
*2026 · code · `code_HarnessScaling_2610.29458.txt` · arXiv [2610.29458](https://arxiv.org/abs/2610.29458)*

ClawGym-Bench, PinchBench and JobBench-Easy, reporting Pass@1 averaged over three runs with GPT-5.4 rubric judging. In-space generalization averages over 18 fixed held-out configurations from H minus H_tr; out-of-space uses Claude Code (v2.1.205) and OpenClaw as unseen black-box harnesses. GPT-5.4 also judges six behavioral dimensions.

- Overall averages (single -> scaling): Qwen3-4B GRPO 29.81 -> 39.04, PPO 33.91 -> 36.93; Qwen3-30B-A3B GRPO 37.29 -> 42.78, PPO 36.31 -> 41.89.
- Structured 18 beats Random-18, all 243 harnesses and random assignment.
- Largest behavioral gains: state tracking, workflow compliance, error recovery.

**Key results:** Across two model scales and both GRPO and PPO, Harness Scaling beats single-harness RL on unseen harness configurations (in-space gains of +3.57 to +9.80 points averaged over benchmarks), costing only 0.19 points on the base harness for Qwen3-30B-A3B. On the unseen Claude Code harness it averages 37.22 (4B) and 41.20 (30B-A3B), above OpenClaw-trained black-box SFT (35.08/38.08) and black-box RL (33.38/36.28).

*Evolution:* The paper follows agentic RL that scales tasks (ClawGym) and environments (EnvScaler, Agent World Model, Agent-World) but trains through one fixed harness, as in Agent Lightning and OpenClaw-RL. It borrows orthogonal-array factorial design from classical experimental design to make harness diversity tractable. Its result that a structured set of 18 harnesses beats exhaustive 243-harness training under a fixed budget suggests that how harness diversity is organized matters, not only how much there is.

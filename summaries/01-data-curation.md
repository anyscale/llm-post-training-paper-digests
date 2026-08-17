# 01 — Data Curation

*v4 post-training summaries, generated solely from the full-text files in [`../texts/`](../texts/) (the existing `summaries/`, `summaries-v2/`, and `summaries-v3/` folders were **not** used as source material). Papers are sorted by arXiv-ID year (2022→2026), then by corpus order within each year. Each entry synthesizes **one lens only** for that paper; the chronological cross-lens narrative — older trends, how the field evolved, and why the newer methods were proposed — lives in [`EVOLUTION_OVERVIEW.md`](EVOLUTION_OVERVIEW.md).*

**Lens:** how training/evaluation data is sourced, generated, filtered, de-duplicated, quality-controlled, and scaled — the datasets, their sizes, and selection criteria.

**Coverage:** 88 of the 99 papers contribute substantive content on this lens; papers for which this lens was not a focus are omitted here and appear under their relevant topic files.

---

## 2022

### CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning
*2022 · code · `code_CodeRL_2207.01780.txt` · arXiv [2207.01780](https://arxiv.org/abs/2207.01780)*

Training and eval data centers on two Python benchmarks: APPS (10,000 problems, 50-50 train/test split, ~23.2 reference programs and ~21.2 unit tests per problem, divided into Introductory/Interview/Competition) and MBPP (974 instances, 374/90/500 train/val/test, one solution plus three assert-style tests, used for zero-shot transfer). For pretraining the authors extend CodeT5 by enlarging the Python corpus with the permissively-licensed Github Code Python dataset (GCPY), yielding 10.5B tokens (10× CodeSearchNet), and add a next-token-prediction objective alongside masked span prediction. Synthetic code samples drawn from the actor LM (both passing and failing) are reused to train the critic and the program-repair model, with ground-truth programs labeled PassedTest.

- APPS: 10k problems, ~23.2 refs / ~21.2 unit tests each; MBPP: 974 instances, 1 solution + 3 asserts
- Pretraining: GCPY corpus → 10.5B tokens (10× CSN), NTP + MSP objectives
- Synthetic actor rollouts (passing + failing) reused for critic and repair-model training

**Key results:** CodeRL+CodeT5-770M sets APPS SOTA at 2.69% pass@1, 6.81% pass@5, 20.98% pass@1000, and 8.48% 1@k / 12.62% 5@k.
*Evolution:* CodeRL extends the REINFORCE/actor-critic line for sequence generation and CodeT5 pretraining into code, anticipating the later trend of execution-feedback and reward-model-driven post-training for code.

### STaR: Bootstrapping Reasoning With Reasoning
*2022 · data · `data_STaR_2203.14465.txt` · arXiv [2203.14465](https://arxiv.org/abs/2203.14465)*

STaR converts an answer-only dataset into a rationale dataset via self-generation, seeded by ~10 hand-written chain-of-thought examples. Datasets: a synthetic 50,000-problem n-digit addition set (uniform over digit lengths, scratchpad format, 10k sampled per outer loop); CommonsenseQA (12,247 items, 9,741/1,221/1,285 train/dev/test, from ConceptNet, 10 modified CoT prompts); and GSM8K (7,473/1,319 train/test). Filtering is by answer correctness: only rationales whose final answer matches ground truth are kept (indicator reward), and Rationalization re-generates rationales for failed items with the answer hinted. The final CQA model trains on 78.2% (rationale generation) + 8.5% (rationalization) of train data; GSM8K on 25.0%/28.7%. Each outer loop retrains from the original GPT-J (6B, Pile-pretrained) checkpoint rather than continually, to avoid overfitting.

- Seeds: ~10 hand-written CoT examples; answer-correctness filter (indicator reward)
- Datasets: 50k synthetic addition, CommonsenseQA (12,247), GSM8K (7,473 train)
- Per-loop retrain from base GPT-J 6B; CQA uses 78.2% + 8.5% of train

**Key results:** GPT-J (6B) + STaR with rationalization reaches 72.5% on CommonsenseQA dev, matching a 30x larger GPT-3 direct-finetuned (73.0%) and beating GPT-J direct-finetuning (60.0%) by +12.5%.
*Evolution:* STaR builds on chain-of-thought prompting and Expert Iteration, turning few-shot CoT into an iterative self-bootstrapping loop that uses answer-correctness as a reward.

### SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions
*2022 · data · `data_SelfInstruct_2212.10560.txt` · arXiv [2212.10560](https://arxiv.org/abs/2212.10560)*

The paper's core contribution is synthesizing instruction data from the model itself, seeded with 175 human-written tasks (25 classification, 150 non-classification, authored without reference to existing datasets). The four-step pipeline: (1) Instruction Generation samples 8 in-context examples (6 human + 2 model-generated); (2) Classification Task Identification uses few-shot prompting (12 clf + 19 non-clf seeds); (3) Instance Generation uses input-first for non-clf and output-first for clf (labels first, then inputs conditioned on labels to avoid label bias); (4) Filtering rejects instructions with ROUGE-L > 0.7 to any existing one, drops image/graph keywords, removes duplicates, and applies length/repetition heuristics. Output: 52,445 instructions (11,584 clf, 40,861 non-clf) and 82,439 instances (35,878 with empty input). Expert audit on 200 samples: 92% valid task, 54% all-fields-valid. Cost ~$600; publicly released.

- Seeds: 175 human tasks (25 clf / 150 non-clf); pipeline: generate → classify → instance → filter
- Filtering: ROUGE-L > 0.7 dedup, keyword/length/repetition heuristics
- Output: 52,445 instructions / 82,439 instances; ~$600 cost; 54% all-fields-valid on audit

**Key results:** GPT3SELF-INST improves over vanilla GPT3 by +33.1 absolute ROUGE-L on SUPERNI (39.9 vs 6.8), nearly matching InstructGPT001 (40.8).
*Evolution:* Building on the 2022 instruction-tuning wave, SELF-INSTRUCT pioneers self-bootstrapped synthetic instruction data that directly enables later open self-instruct-style models.

## 2023

### AgentTuning: Enabling Generalized Agent Abilities for LLMs
*2023 · code · `code_AgentTuning_2310.12823.txt` · arXiv [2310.12823](https://arxiv.org/abs/2310.12823)*

AgentInstruct is built from 6 agent tasks (ALFWorld, WebShop, Mind2Web, Knowledge Graph, Operating System, Database), yielding 1,866 filtered interaction trajectories from 35,341 instructions. Tasks with train splits use them directly; for OS and DB the authors use Task Derivation (from the BIRD text-to-SQL benchmark) and Self-Instruct (GPT-4 proposes OS tasks plus reference solutions and eval scripts). Trajectories are collected by running GPT-4 (gpt-4-0613) as the agent under ReAct prompting with 1-shot examples; Mind2Web partly uses gpt-3.5-turbo-0613 for budget. Filtering keeps trajectories with reward r=1 (r≥2/3 for Mind2Web); a 7B ablation shows filtered data (held-in 1.96, held-out 0.65) beats unfiltered (1.34, 0.47). General-domain data comes from ShareGPT (57,096 GPT-3.5 + 3,670 GPT-4 conversations, sampled 4:1). A 10-gram Llama-2-style contamination check finds no leakage.

- 6 agent tasks → 1,866 filtered trajectories from 35,341 instructions
- Collection: GPT-4 ReAct agent; filtering by reward r=1 (r≥2/3 for Mind2Web)
- General data: ShareGPT 60,766 conversations (4:1 GPT-3.5:GPT-4); 10-gram decontamination

**Key results:** AgentLM-70B achieves held-out overall 1.40 (+176% over Llama-2-70B), roughly matching GPT-3.5 (1.49), with general ability preserved (MMLU 59.5, GSM8K 59.7).
*Evolution:* AgentTuning builds on ReAct, Self-Instruct, and GPT-4 distillation, reacting to AgentBench's finding that open LLMs trailed GPT-3.5/GPT-4 on agent tasks.

### RLTF: Reinforcement Learning from Unit Test Feedback
*2023 · code · `code_RLTF_2307.04349.txt` · arXiv [2307.04349](https://arxiv.org/abs/2307.04349)*

Training and eval use APPS (10,000 problems, 5k/5k train/test, Introductory/Interview/Competition, ~23.2 refs and ~21.2 unit tests per problem with ~20 hidden tests) and MBPP (974 short problems, 374/90/500/10-few-shot, one ~6.8-line solution and three visible asserts). The distinctive data element is the online buffer: instead of pre-collected offline rollouts, the model generates code for sampled problems in real time, runs it through the compiler for unit-test feedback, and stores (problem, generated code, feedback) triples in a fixed-length queue of 6,400 that is refreshed every 50 steps; old samples are evicted as fresh ones arrive, giving on-policy diversity rather than a static offline corpus.

- APPS: 10k problems, ~23.2 refs / ~21.2 tests; MBPP: 974, 1 solution + 3 asserts
- Online buffer: 6,400-capacity queue refreshed every 50 steps, on-policy rollouts
- Real-time compiler unit-test feedback labels each (problem, code) triple

**Key results:** CodeT5-770M + RLTF on APPS: pass@1 1.45%, pass@5 3.78%, pass@1000 19.92%, SOTA among CodeT5-based RL methods; MBPP zero-shot pass@1 71.3 vs CodeRL 68.1.
*Evolution:* RLTF builds on CodeRL and PPOCoder by moving code-LLM RL from an offline coarse-reward regime to an online loop with compiler-parsed, line-level reward shaping.

### What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
*2023 · data · `data_DEITA_2312.15685.txt` · arXiv [2312.15685](https://arxiv.org/abs/2312.15685)*

DEITA's central contribution is automatic SFT data selection along three dimensions: complexity, quality, and diversity. It builds two pools—Xsota (300K, ensembling WizardLM-Alpaca, WizardLM-ShareGPT, UltraChat, ShareGPT, treated as high-quality) and Xbase (100K, from Alpaca, Dolly, OAssist, FLAN-2022, lower-quality/redundant). To measure complexity and quality it introduces Evol-Complexity and Evol-Quality: from a 2K-sample Alpaca seed, ChatGPT (gpt-3.5-turbo-0613) evolves each instruction/response over M=5 iterations to yield 6 graded variants, then ranks/scores all six in one prompt; an LLaMA-1-7B scorer is distilled from these scores. Diversity uses a Repr Filter: LLaMA-1-13B embeddings with cosine-distance threshold τ=0.9. Selected budgets are 6K and 10K, which match far larger full-pool training.

- Two pools: Xsota (300K, high-quality) and Xbase (100K, lower-quality/redundant)
- Evol-Complexity/Quality: 2K Alpaca seed → M=5 evolution → 6 graded variants, distilled LLaMA-1-7B scorer
- Diversity: LLaMA-1-13B embeddings, cosine threshold τ=0.9; select 6K/10K

**Key results:** DEITA-Mistral-7B6K+DPO reaches 7.55 MT-Bench and 90.06% AlpacaEval; 3K DEITA-selected samples match full 300K training (100x reduction).
*Evolution:* DEITA builds on the 2023 "less is more for alignment" trend and WizardLM's Evol-Instruct, formalizing automatic data selection across complexity, quality, and diversity for SFT.

### Direct Preference Optimization: Your Language Model is Secretly a Reward Model
*2023 · data · `data_DPO_2305.18290.txt` · arXiv [2305.18290](https://arxiv.org/abs/2305.18290)*

Three open-ended generation tasks use offline preference datasets D={(x, y_w, y_l)}. (1) Controlled sentiment: IMDb review prefixes (2-8 tokens); 25k prefixes, 4 completions each, 6 preference pairs per prefix labeled by a pre-trained sentiment classifier (siebert/sentiment-roberta-large-english) as ground-truth reward; GPT-2-large SFT'd 1 epoch on IMDb. (2) Summarization: Reddit TL;DR with human preferences from Stiennon et al.; SFT on human-written forum summaries (CarperAI/openai_summarize_tldr_sft). (3) Single-turn dialogue: Anthropic Helpful-and-Harmless dataset (~170k dialogues), each ending with a response pair plus a human preference label; no SFT available, so an off-the-shelf LM is fine-tuned on preferred completions as the reference. DPO reuses public preference datasets rather than sampling new pairs, initializing π_ref=π_SFT when available.

- Three preference datasets: IMDb sentiment (classifier-labeled), TL;DR (human prefs), Anthropic HH (~170k)
- Reuses existing offline preference data; no new pair sampling
- Reference π_ref=π_SFT when SFT available; else fine-tune on preferred completions

**Key results:** TL;DR DPO ~61% GPT-4 win rate vs reference, beating PPO's 57% and Best-of-N; IMDb DPO achieves the best reward-KL frontier, dominating PPO.
*Evolution:* DPO reacts against the complexity and instability of PPO-based RLHF by deriving the KL-constrained optimum in closed form, enabling a family of simpler preference-optimization methods.

### LIMA: Less Is More for Alignment
*2023 · data · `data_LIMA_2305.11206.txt` · arXiv [2305.11206](https://arxiv.org/abs/2305.11206)*

LIMA is built on 1,000 carefully curated prompt-response pairs (~750K tokens). 750 come from community Q&A: 200 Stack Exchange STEM + 200 other (sampled with temperature τ=3 for domain diversity, top-scored self-contained questions, top answer score ≥10, filtered to 1,200-4,096 chars, no first-person voice, links/images/HTML stripped), 200 wikiHow (sampled by category then article, title as prompt, body as response), and 150 manually picked r/WritingPrompts. 200 manually authored examples (Group A) plus 50 Super-Natural Instructions NLG tasks fill the rest in a uniform helpful-assistant style; 13 are toxic prompts with refusal responses. A 50-example dev set and 300-example test set (70 r/AskReddit + 230 Group B) are held out. Ablations on 7B show prompt diversity and response quality each add ~0.5 Likert points, while scaling quantity 2K→32K plateaus.

- 1,000 pairs (~750K tokens): 750 community Q&A (Stack Exchange, wikiHow, r/WritingPrompts) + 250 authored/SuperNI
- Stack Exchange sampling: τ=3 diversity, top-scored, 1,200-4,096 char filter, HTML stripped
- 50 dev / 300 test held out; 7B ablation: quality+diversity each +~0.5, quantity plateaus 2K→32K

**Key results:** LIMA (65B LLaMa, SFT on 1,000 examples, no RLHF) is equal-or-preferred to GPT-4 43%, Claude 46%, and beats Alpaca 65B (52K examples) in human pairwise preference.
*Evolution:* Building on instruction tuning, RLHF, and Self-Instruct/Alpaca distillation, LIMA argues alignment is "superficial"-mostly style-so a curated 1,000-example SFT set rivals RLHF products.

### Magicoder: Empowering Code Generation with OSS-Instruct
*2023 · data · `data_Magicoder-OSS-Instruct_2312.02120.txt` · arXiv [2312.02120](https://arxiv.org/abs/2312.02120)*

OSS-Instruct generates synthetic code instruction data by prompting a teacher LLM (gpt-3.5-turbo-1106, greedy, chosen for cost) with a random open-source code snippet to produce a self-contained coding problem and solution. Seed snippets come from a filtered, decontaminated version of The Stack (starcoderdata): 80K snippets from 80K documents — 40K Python plus 5K each from C++, Java, TypeScript, Shell, C#, Rust, PHP, Swift — with 1-15 consecutive lines extracted per document. Cleaning drops duplicates sharing the same seed snippet but intentionally keeps noisy/incomplete solutions, which slightly helps (HumanEval+ 55.5 vs 54.9). Decontamination removes matches against HumanEval, MBPP, APPS, DS-1000, and GSM8K, filtering only 9 samples. The final dataset is ~75K entries; by TF-IDF cosine similarity it is the least HumanEval-like (avg 0.105) vs Evol-Instruct (0.131) and Self-Instruct/Code Alpaca (0.169), evidencing lower bias and greater diversity.

- Seeds: 80K starcoderdata snippets (40K Python + 5K×8 other languages), 1-15 lines each
- Teacher gpt-3.5-turbo-1106 (greedy); ~75K entries; noisy solutions kept (slight gain)
- Decontamination vs HumanEval/MBPP/APPS/DS-1000/GSM8K removes only 9; TF-IDF 0.105 (most diverse)

**Key results:** MagicoderS-CL-7B surpasses ChatGPT on HumanEval+ pass@1 (66.5 vs 65.9) with only 7B parameters; MagicoderS-DS-6.7B beats DeepSeek-Coder-Instruct-6.7B using 8× fewer finetuning tokens.
*Evolution:* OSS-Instruct extends Self-Instruct and reacts against Evol-Instruct/WizardCoder, grounding synthetic instruction generation in the abundance of open-source code rather than fixed heuristic seeds.

### Scaling Relationship on Learning Mathematical Reasoning with Large Language Models
*2023 · data · `data_RFT-rejection-sampling_2308.01825.txt` · arXiv [2308.01825](https://arxiv.org/abs/2308.01825)*

Training and eval use GSM8K (7,473 train questions). For SFT-scaling experiments the train set is downsampled to {1, 1/2, 1/4, 1/8, 1/16, 1/32}. For RFT, an SFT model samples k∈{1,3,6,12,25,50,100} candidate chain-of-thought paths per question at temperature 0.7; paths with wrong final answers or incorrect Python-evaluated calculations are discarded. Each path's equation list (extracted from `<<equation>>` markers, whitespace-removed, symbol-joined) defines a key, and one representative per distinct equation list is kept (the most Levenshtein-dissimilar), so deduplication is by distinct reasoning/calculation process, not raw count. Augmented sets grow from ~12K (k=1) to ~47K (k=100); multi-model aggregated sets D'U13B (D7B⊕D7B2⊕D13B⊕D13B2) and D'U33B reach ~104K and ~110K. Two preliminary ideas (self-query, self-revising) gave no-to-marginal gains and were dropped.

- GSM8K 7,473 train; SFT downsampling sweep {1…1/32}
- RFT: sample k∈{1…100} CoT paths at T=0.7, keep only correct-answer paths
- Dedup by distinct equation-list key (most Levenshtein-dissimilar); sets 12K→110K

**Key results:** LLaMA-7B RFT-U13B reaches 49.3% maj1@1 on GSM8K vs 35.9% SFT (+13.4); RFT helps weaker base models most and adds nothing for 33B/65B/70B, which overfit training paths.
*Evolution:* A 2023 scaling-relationship study building on STaR and self-consistency, replacing trained verifiers with simple rejection-sampling deduplication of correct CoT paths.

### Statistical Rejection Sampling Improves Preference Optimization
*2023 · data · `data_RSO_2309.06657.txt` · arXiv [2309.06657](https://arxiv.org/abs/2309.06657)*

Preference data comes from Reddit TL;DR summarization and AnthropicHH dialogue. TL;DR provides SFT data (117k/6k/6k train/val/test) and human feedback Dhf (93k preferences over decodes from multiple models); AnthropicHH helpful slice has 161k/9k train/test, positive responses used as SFT targets. Cross-task transfer is probed on CNN/DailyMail (287k/13k/11k, fine-tune data only). Core data construction is generative: sample 64 response candidates from the SFT policy (temperature 0.7, top-k=40), sub-sample 8 via statistical rejection sampling, then label pairs with a T5-XXL (11B) pairwise reward-ranking model trained on Dhf (validation accuracy 73.23% summarization, 69.75% assistant). Three preference-pair variants are compared: direct (use Dhf), sft-sample-rank, and rso-sample-rank.

- Sources: Reddit TL;DR (SFT 117k, human prefs 93k), AnthropicHH (161k), CNN/DM (287k) for transfer
- Generative: 64 candidates from SFT policy (T=0.7, top-k=40), subsample 8 via rejection sampling
- Pairs labeled by T5-XXL 11B reward model (73.23%/69.75% val acc); three pair variants compared

**Key results:** RSO reaches 84.40% Gold Reward and 71.86% AutoSxS on Reddit TL;DR vs DPO 76.09/58.65; at T5-XXL scale human raters choose RSO >2x more often than DPO.
*Evolution:* RSO reacts to DPO and SLiC, arguing DPO's MLE is mismatched because preference pairs are not sampled from the optimal policy, bridging offline preference optimization and online RLHF.

### Enhancing Chat Language Models by Scaling High-quality Instructional Conversations
*2023 · data · `data_UltraChat_2305.14233.txt` · arXiv [2305.14233](https://arxiv.org/abs/2305.14233)*

UltraChat's core contribution is ~1.47M high-quality multi-turn instructional dialogues generated without human queries, organized into three sectors. Sector 1 (Questions about the World) derives opening lines from 30 ChatGPT-generated meta-topics expanded into subtopics, plus the 10,000 most frequent Wikidata/Wikipedia entities (5 meta-questions, 10 specific, 20 extended each), sampling ~500k questions. Sector 2 (Creation and Generation) uses 20 material types with ChatGPT-generated instructions, ~80% refined into detailed instructions. Sector 3 (Assistance on Existing Materials) draws 10,000 C4 text pieces classified by URL keywords into 20 types, paired with 5 instructions each via manual templates (~500k opening lines). Two separate ChatGPT Turbo APIs act as user and assistant iteratively (3-7 rounds). Filtration removes polite filler. Stats: 3.8 avg turns, 1467.4 tokens/dialogue, top lexical diversity (MTLD 74.3), coherence 9.06.

- ~1.47M multi-turn dialogues, no human queries, 3 sectors (World/Creation/Materials)
- Sectors seeded from meta-topics, 10k Wikidata/Wikipedia entities, 20 material types, 10k C4 pieces
- Two ChatGPT Turbo APIs as user+assistant (3-7 rounds); 3.8 avg turns, MTLD 74.3, coherence 9.06

**Key results:** UltraLLaMA-13B scores 9.02 overall (ChatGPT-judged, 1-10) vs Vicuna-13B 8.96 and ChatGPT 9.12; pairwise win rate up to 85% over open-source baselines.
*Evolution:* Builds on the 2023 Self-Instruct/Alpaca/Vicuna wave of distilling ChatGPT; UltraChat's large-scale synthetic multi-turn SFT corpus later seeded open alignment pipelines such as UltraFeedback.

### UltraFeedback: Boosting Language Models with Scaled AI Feedback
*2023 · data · `data_UltraFeedback_2310.01377.txt` · arXiv [2310.01377](https://arxiv.org/abs/2310.01377)*

UltraFeedback is a million-scale AI-feedback dataset built via a three-stage pipeline: instruction collection, completion sampling, and GPT-4 annotation. Instructions (63,967 total) target four abilities—instruction-following, truthfulness, honesty, helpfulness—drawn from six public sources: TruthfulQA, FalseQA, 10k each from Evol-Instruct and UltraChat, 20k from ShareGPT, and FLAN (stratified: 3k CoT, 10/task elsewhere), after 13-gram decontamination against AlpacaEval/Evol-Instruct/UltraChat test sets (48 samples removed). Completions come from a pool of 17 models (GPT-4, gpt-3.5-turbo, Bard; LLaMA-family UltraLM/WizardLM/Vicuna/LLaMA2-Chat/Alpaca; MPT-30B, Falcon-40B, StarChat, Pythia-12B), four sampled per instruction, with added principle system-prompts to elicit diverse behaviors. GPT-4 then gives four fine-grained 1-5 scalar scores plus textual critique, yielding >1M feedback for 255,864 completions / 340,025 pairs—over twice the size of prior preference/critique datasets and the only one offering both scalar and text feedback.

- 63,967 instructions from 6 sources (TruthfulQA, FalseQA, Evol-Instruct, UltraChat, ShareGPT, FLAN); 13-gram decontam (48 removed)
- 17-model completion pool, 4 per instruction + principle system-prompts for diversity
- GPT-4: four 1-5 scores + critique → >1M feedback, 255,864 completions / 340,025 pairs

**Key results:** UltraRM beats open-source reward models by >6.3% avg preference-prediction accuracy; UltraLM-13B-PPO attains the highest average win rate (69.7%) on AlpacaEval/Evol-Instruct/UltraChat, surpassing LLaMA2-70B-Chat.
*Evolution:* Extends RLAIF and SFT data-engineering lessons by scaling GPT-4 AI feedback into a large, diverse, fine-grained open preference dataset that became a staple substrate for later 2023–2024 preference work.

### WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions
*2023 · data · `data_WizardLM-EvolInstruct_2304.12244.txt` · arXiv [2304.12244](https://arxiv.org/abs/2304.12244)*

Evol-Instruct mass-produces open-domain instructions of controlled complexity using an LLM rather than human annotators. Seed data is the 52k Alpaca set (Self-Instruct from 175 human seeds), chosen so WizardLM's training data has almost no direct human annotation. Over M=4 evolution epochs on Azure ChatGPT (gpt-3.5-turbo), each instruction is rewritten by one of six prompt operations (five in-depth: add constraints, deepening, concretizing, increase reasoning, complicate input; one in-breadth mutation), then ChatGPT generates the response, yielding 250k instructions from 52k×4×3=624k API calls. An Elimination Evolving filter drops failed evolutions (no information gain, short "sorry" responses <80 words, stop-word-only answers, or prompt-leakage copies). For fair comparison with Vicuna, 70k are sampled from the 250k pool. A manually built, difficulty-balanced 218-instance, 29-skill test set, WizardEval, is introduced because prior test sets skew easy.

- Seed: 52k Alpaca; M=4 evolution epochs on gpt-3.5-turbo → 250k instructions (624k API calls)
- Six prompt ops (5 in-depth + 1 in-breadth); Elimination filter drops failed evolutions
- 70k sampled for Vicuna comparison; 218-instance WizardEval test set (29 skills, difficulty-balanced)

**Key results:** WizardLM-13b beats Vicuna-13b and Alpaca-13b on average (58.96 vs 54.60 vs 43.44); WizardLM-70b reaches 71.33 avg (approaching ChatGPT-3.5's 76.15) and 99.7 on WizardEval.
*Evolution:* Builds on the Self-Instruct/Alpaca/Vicuna line, reacting to the observed difficulty skew in human-created ShareGPT and anticipating the synthetic-data-scaling and difficulty-aware-curriculum trend.

### ZEPHYR: Direct Distillation of LM Alignment
*2023 · data · `data_Zephyr_2310.16944.txt` · arXiv [2310.16944](https://arxiv.org/abs/2310.16944)*

Zephyr aligns Mistral-7B using two distilled dialogue datasets. UltraChat is a self-refinement corpus of 1.47M multi-turn dialogues generated by GPT-3.5-Turbo across 30 topics and 20 text-material types. Initial dSFT on the full set produced a model with capitalization errors and unhelpful refusals (e.g., "I don't have personal experiences"), so the authors applied truecasing heuristics (~5% of data) and helpfulness filters, yielding ~200k SFT examples. UltraFeedback provides 64k prompts, each with four completions from an ensemble of chat models (Claude, Falcon, Llama) scored by GPT-4 on instruction-following, honesty, and helpfulness. Binary preferences are built offline by taking the highest mean-scored response as "chosen" and a random lower-scored one as "rejected" (random rather than worst to add diversity and difficulty). All preprocessed data is released on the Hugging Face Hub. No human annotation is used.

- dSFT: UltraChat 1.47M → truecasing (~5%) + helpfulness filters → ~200k examples
- dDPO: UltraFeedback 64k prompts, 4 completions each, GPT-4 scored
- Preferences: highest mean = chosen, random lower = rejected (for diversity); no human annotation

**Key results:** Zephyr-7B reaches MT-Bench 7.34, surpassing Llama2-Chat-70B (6.86), with AlpacaEval 90.60% win rate; trained in 2-4 hours on 16 A100s with no human annotation.
*Evolution:* Zephyr builds on the self-instruct/dSFT lineage and the InstructGPT alignment recipe, replacing costly human-feedback-plus-PPO with DPO over GPT-4-scored AI feedback, popularizing the distilled SFT-then-DPO-on-AIF pipeline.

### Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations
*2023 · rl · `rl_Math-Shepherd_2312.08935.txt` · arXiv [2312.08935](https://arxiv.org/abs/2312.08935)*

The core contribution is an automatic process-annotation pipeline that removes the human-labeling bottleneck of prior PRMs (Uesato 2022; Lightman 2023 PRM800K). Generators/completers are first SFT'd on MetaMath (3 epochs). To build ORM/PRM training data, 7B and 13B models are trained 1 epoch on the GSM8K and MATH training splits, then sample 15 solutions per problem; duplicates are removed and each step is annotated via a "completer" (LLemma-7B, N=8 decoded continuations) whose final answers are checked against the golden answer. Step labels use hard estimation (HE, 1 if any completion is correct) or soft estimation (SE, fraction correct). This yields ~170k labeled solutions for GSM8K and ~270k for MATH—about 4× larger than PRM800K—and at N=4 HE reaches 86% accuracy against 160 hand-checked steps. The automatic data even outperforms PRM800K as a verifier on MATH, attributed to better distribution match and larger scale.

- SFT on MetaMath (3 epochs); ORM/PRM data from 7B/13B sampling 15 solutions/problem on GSM8K+MATH
- Step labels via LLemma-7B completer (N=8) checking final answers; HE (any correct) or SE (fraction)
- ~170k GSM8K / ~270k MATH labeled solutions (~4× PRM800K); auto-data beats PRM800K on MATH

**Key results:** Mistral-7B + step-by-step PPO with Math-Shepherd: GSM8K 77.9%→84.1%, MATH 28.6%→33.0%; with verification 89.1% GSM8K and 43.5% MATH.
*Evolution:* Building on the human-annotated PRM line (Uesato, Lightman PRM800K) and Cobbe's ORM, Math-Shepherd makes process supervision scalable by automating step labels via MCTS-style completion.

## 2024

### AgentGym: Evolving Large Language Model-based Agents across Diverse Environments
*2024 · code · `code_AgentGym_2406.04151.txt` · arXiv [2406.04151](https://arxiv.org/abs/2406.04151)*

AGENTGYM assembles a cross-environment agent corpus spanning 14 environments and 89 task types, reusing original instructions for data-rich tasks (WebShop, ALFWorld) and expanding sparse ones via self-instruct and instruction evolution prompted with GPT-3.5/4-Turbo to reach 20,509 instructions. A diverse 1,160-instruction subset forms the AGENTEVAL benchmark; the remainder is the training pool. ReAct-style trajectories are collected from SOTA models plus crowdsourced humans, filtered by reward/correctness, producing AGENTTRAJ (6,130 trajectories) and its extension AGENTTRAJ-L (14,485), all in a unified ReAct format with environment-specific reward or success-rate feedback.

- 14 environments / 89 task types; 20,509 instructions; 1,160-instruction AGENTEVAL subset
- Trajectories from GPT-4-Turbo + humans, filtered by reward → AGENTTRAJ (6,130) / AGENTTRAJ-L (14,485)
- Sparse tasks expanded via self-instruct + instruction evolution (GPT-3.5/4-Turbo)
- Unified ReAct format with env-specific reward/success-rate feedback

**Key results:** AGENTEVOL on Llama-2-Chat-7B beats the behavioral-cloning upper bound BClarge on WebShop (76.5 vs 73.5), ALFWorld (88.0 vs 83.0), BabyAI (82.7 vs 74.19), and TextCraft (64.0 vs 60.0), and matches or surpasses GPT-4-Turbo on several environments.

*Evolution:* An early 2024 attempt to push LLM agent self-evolution beyond isolated environments toward multi-environment generalists, complementing behavioral-cloning agent-tuning works like AgentTuning/AgentOhana and foreshadowing the later surge of RL/post-training for generalist LLM agents.

### AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials
*2024 · code · `code_AgentTrek_2412.09605.txt` · arXiv [2412.09605](https://arxiv.org/abs/2412.09605)*

The pipeline mines tutorials from the 20.8B-entry RedPajama corpus in four steps: a rule-based pre-filter (keyword/length/URL, 200–5000 words) cuts to 68.8M entries at 92.69% recall; GPT-4o-mini labels 90K (F1 88.5%); a FastText classifier (F1 89.5%) filters to ~18.8M deduplicated entries; GPT-4o-mini standardizes each into a 5-field template at $0.89/1K. A GPT-4o VLM agent then replays tutorials in real browsers via BrowserGym (~86,114 tokens/task, $215/1K), recording screenshots, AXTree, DOM, reasoning, and Playwright actions. A GPT-4o VLM evaluator (84% accuracy vs a 1,081-trajectory human gold set) filters the output.

- 20.8B RedPajama → 68.8M (rule pre-filter) → ~18.8M (FastText) → 23,430 standardized tutorials
- VLM browser replay yields 10,398 verified trajectories across 127 websites, 11 categories, avg 12.1 steps
- Cost ~$0.551/trajectory; evaluator 84% accurate vs 1,081 human-gold trajectories
- Four-stage mining + template standardization + VLM-guided replay + evaluator filtering

**Key results:** Qwen2.5-32B-Instruct fine-tuned on AgentTrek scores 22.40 success rate on WebArena, surpassing GPT-4 (14.41); Qwen2-VL-7B reaches 67.4 average on ScreenSpot Web grounding vs 30.7 baseline.

*Evolution:* Builds on the synthetic-trajectory trend for GUI agents (BAGEL, NNetNav, Synatra) and reacts against the cost and scalability limits of human-annotated datasets like Mind2Web, establishing a low-cost automated data-synthesis paradigm.

### Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions
*2024 · code · `code_Marco-o1_2411.14405.txt` · arXiv [2411.14405](https://arxiv.org/abs/2411.14405)*

Marco-o1 builds three SFT datasets totaling 60,266 samples for full-parameter fine-tuning of Qwen2-7B-Instruct. The Open-O1 CoT Dataset (Filtered, 45,125 samples) is derived from the open Open-O1 CoT data via heuristic and quality filtering so the model adopts structured reasoning patterns. The Marco-o1 CoT Dataset (Synthetic, 10,000) is generated using MCTS to construct complex reasoning pathways, and the Marco Instruction Dataset (5,141) adds instruction-following data to preserve general competence. No explicit de-duplication or scaling-law study is described; the synthetic CoT portion is produced by the same MCTS machinery later used at inference.

- Three SFT sets totaling 60,266: 45,125 filtered Open-O1 CoT + 10,000 MCTS-synthetic CoT + 5,141 instruction
- Combined for full-parameter SFT of Qwen2-7B-Instruct
- Emphasis on mixing CoT traces with instruction data rather than large-scale curation
- Synthetic CoT generated by the same MCTS used at inference

**Key results:** Marco-o1-MCTS (step): 90.40% on MGSM-En vs 84.00% Qwen2-7B-Instruct (+6.17%); Marco-o1-MCTS (mini-step 32): 82.40% on MGSM-Zh vs 76.80% (+5.60%); Test@32 reaches 99.60% (En) / 96.80% (Zh).

*Evolution:* An early open attempt (Nov 2024) to demystify o1 by combining CoT SFT with AlphaZero-style MCTS and self-reflection, explicitly extending reasoning models to open-ended and multilingual/translation tasks.

### ToolACE: Winning the Points of LLM Function Calling
*2024 · code · `code_ToolACE_2409.00920.txt` · arXiv [2409.00920](https://arxiv.org/abs/2409.00920)*

ToolACE's core contribution is a synthetic data pipeline curating a pool of 26,507 APIs across 390 domains (real + synthesized). The Tool Self-Evolution Synthesis (TSS) module runs a speciation-adaptation-evolution loop: speciation builds a hierarchical API context tree over API-related pretraining docs; adaptation samples subtrees to assign distinct domains; evolution iteratively generates new definitions using an example buffer plus diversity indicators (new functions/params, type mutation, changed returns, nested types). Self-Guided Dialog Generation (SDG) role-plays four dialog types via three LLM agents. Accuracy is enforced by Dual-Layer Verification: a rule checker (JSON-schema, format, consistency) plus an LLM model checker decomposed into hallucination, consistency, and tool-response checks. Ablation subsets span complexity (easy/medium/hard, 60K each) and diversity (6/14/30 of 30 clusters, ~30K each).

- 26,507 APIs across 390 domains — broadest among peers — mixing real and synthesized APIs
- TSS speciation-adaptation-evolution loop with diversity indicators for nested types
- SDG role-plays single/parallel/dependent/non-tool dialogs via 3 LLM agents
- Dual-Layer Verification (rule + LLM checker) supervised by human experts

**Key results:** ToolACE-8B (LLaMA-3.1-8B-Instruct + LoRA) ranks 3rd on BFCL-v3 with 59.13 overall, competitive with GPT-4-turbo (59.49); with matched 25K-sample training, ToolACE data gives 58.19 BFCL overall vs 40.51 (xLAM) and 24.90 (ToolLLM).

*Evolution:* Extends the synthetic-data-for-tool-use line (ToolLLM, ToolAlpaca, Gorilla, xLAM) by adding evolutionary API synthesis, capability-aware complexity targeting, and rigorous dual-layer verification, showing an 8B specialist can rival GPT-4.

### WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
*2024 · code · `code_WebRL_2411.02337.txt` · arXiv [2411.02337](https://arxiv.org/abs/2411.02337)*

Training and evaluation use WebArena (5 sites). WebArena-Lite supplies 1,186 training samples (instruction + trajectory + reward function) and 165 test cases, with its ~1K oracle trajectories deemed too few alone. The self-evolving curriculum generates new tasks via in-breadth evolving (WizardLM-style) seeded from instructions the model failed in prior phases; each phase keeps 500 generated instructions after a two-step filter (critic-score threshold 0.05–0.75, then GPT-4o removes WebArena-infeasible tasks via per-site rules). Only successful trajectories enter the replay buffer. The ORM training set is built by rewriting WebArena-Lite instructions and collecting rollouts from all baselines, yielding 12,200 labeled samples using the reward function as ground truth.

- WebArena-Lite: 1,186 training + 165 test samples across 5 sites
- Self-evolving curriculum keeps 500 instructions/phase after critic-score (0.05–0.75) + GPT-4o feasibility filter
- Only successful trajectories enter the replay buffer
- ORM set: rewritten instructions + baseline rollouts → 12,200 reward-labeled samples

**Key results:** Llama-3.1-8B + WebRL: 4.8% to 42.4% SR on WebArena-Lite (vs GPT-4-Turbo 17.6%); the trained 8B ORM verifies success at ~80% accuracy, exceeding GPT-4-based verifiers (~70-73%).

*Evolution:* Builds on DigiRL/AWR actor-critic online RL and WizardLM-style evol-instruct task generation, adding a self-evolving curriculum and KL-anchored off-policy updates, showing open LLMs can surpass proprietary APIs on web tasks.

### DataComp-LM: In search of the next generation of training sets for language models
*2024 · data · `data_DataCompLM_2406.11794.txt` · arXiv [2406.11794](https://arxiv.org/abs/2406.11794)*

DCLM-POOL is a 240T-token raw web-text corpus (200B documents, 370TB gzipped) re-extracted from all pre-2023 Common Crawl with resiliparse (beating WET by ~2.5 CORE points and 8× faster). DCLM-BASELINE (3.8T tokens) is built by resiliparse extraction → RefinedWeb heuristic filters → Bloom-filter near-duplicate dedup (scales past 10TB) → fastText quality filter keeping the top-10%. The fastText classifier is trained on ~400K documents with positives from OpenHermes-2.5 + r/ExplainLikeImFive and negatives from a RefinedWeb reproduction; this OH-2.5+ELI5 choice gives +3.5 CORE points over Wikipedia/OpenWebText2/GPT-3-style positives. Mixing Wikipedia/Books/StackExchange/arXiv/GitHub helps weak CC subsets but hurts DCLM-BASELINE. Decontamination tooling is released; 416 baseline experiments span five scales (400M-1× to 7B-2×).

- DCLM-POOL 240T tokens, resiliparse-extracted from pre-2023 Common Crawl
- DCLM-BASELINE 3.8T tokens: RefinedWeb filters → Bloom dedup → fastText top-10% quality filter
- fastText trained on ~400K docs; OH-2.5+ELI5 positives (+3.5 CORE over Wikipedia positives)
- 416 baseline experiments across 5 scales; decontamination tooling released

**Key results:** DCLM-BASELINE 7B trained on 2.6T tokens reaches 63.7% MMLU 5-shot, +6.6pp over MAP-Neo with 40% less compute; fastText (OH-2.5+ELI5) filtering yields CORE 41.0 at 7B-1× vs 35.7 for Wikipedia positives.

*Evolution:* Systematizes data-centric LM research at 240T-token/7B scale as a reaction to closed-data models, with its fastText OH-2.5+ELI5 recipe directly seeding later open pretraining datasets such as Nemotron-CC, OLMo-2, and WebOrganizer.

### KTO: Model Alignment as Prospect Theoretic Optimization
*2024 · data · `data_KTO_2402.01306.txt` · arXiv [2402.01306](https://arxiv.org/abs/2402.01306)*

KTO's central data claim is that alignment need not use paired preferences. The main scaling experiments train on Anthropic-HH, OpenAssistant (OASST), and SHP — all originally preference datasets — converting each pair yw≻yl into two examples (yw as desirable, yl as undesirable), yielding 2n examples from n pairs. Later runs align Zephyr-β-SFT and Mistral-7B on UltraFeedback. A "one-y-per-x" subsampling keeps only one output per input, removing all pairing and cutting volume by up to 72% yet still beating DPO. Imbalance experiments discard up to 90% of desirable examples (ratios down to 1:10), rebalanced via λD, λU. Score/rating feedback can enter via a weighting function. No novel de-duplication or filtering pipeline is described.

- Converts preference pairs into 2n unpaired desirable/undesirable examples
- One-y-per-x subsampling cuts volume up to 72% yet still beats DPO
- Imbalance experiments discard up to 90% desirable examples (down to 1:10), rebalanced via λD, λU
- Score/rating feedback enters via a weighting function; eval prompts from OASST test set

**Key results:** KTO matches or exceeds DPO across 1B–30B models; on Zephyr-β-SFT aligned on UltraFeedback, GSM8K rises 40.0→53.5 (+13.5 pts) over DPO; KTO with one-y-per-x (72% less data) still beats DPO (0.631 vs 0.600 winrate).

*Evolution:* Reframes alignment through prospect theory and the HALO family, arguing the inductive bias of the loss drives success; by enabling alignment from cheap binary feedback it lowers the data-collection barrier and motivates later non-pairwise, reference-free alignment losses.

### MAGPIE: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing
*2024 · data · `data_Magpie_2406.08464.txt` · arXiv [2406.08464](https://arxiv.org/abs/2406.08464)*

MAGPIE exploits the observation that feeding an aligned LLM only its pre-query template (e.g., Llama-3's user header) makes it auto-regressively emit a plausible user query, with no seed questions or prompt engineering. A two-step pipeline (generate instruction, then response) runs on Llama-3-8B-Instruct (MAGPIE-Air, 3M convs / 1.28B tokens) and Llama-3-70B-Instruct (MAGPIE-Pro, 1M convs / 477M tokens), plus Llama-3.1, Qwen2, Gemma-2, Phi-3 (over 11.4M instructions total). Cost is ~$0.12/1K (Air) and $1.1/1K (Pro) on A100s with vLLM, with no GPT-4 API calls. Quality/difficulty are auto-labeled by Llama-3-8B-Instruct; safety by Llama-Guard-2 (<1% unsafe). Filtering exposes 8 metrics (length, category, quality, difficulty, FAISS min-neighbor distance, reward, reward difference) with 6 off-the-shelf configs. Extensions cover multi-turn (MAGPIE-MT) and preference data (k=5, T=0.8, ArmoRM scoring).

- Two-step pipeline (instruction then response) prompted by the chat template alone — no seeds
- MAGPIE-Air 3M convs / 1.28B tokens; MAGPIE-Pro 1M convs / 477M tokens; 11.4M+ instructions total
- Cost ~$0.12/1K (Air), $1.1/1K (Pro); safety via Llama-Guard-2 (<1% unsafe)
- Filtering via 8 metrics / 6 configs; multi-turn and ArmoRM-scored preference extensions

**Key results:** Llama-3-8B-Base + MAGPIE-Pro-300K-Filtered SFT + MAGPIE-Pro-DPO reaches AlpacaEval2 LC 50.10% vs GPT-4-Turbo and WR 53.53% vs official Llama-3-8B-Instruct, surpassing both while using only ~400K data versus >10M.

*Evolution:* Extends synthetic-instruction lines (Self-Instruct, Evol-Instruct, UltraChat) by removing seed-question and prompt-engineering dependence, extracting an aligned model's own instruction distribution via its chat template to democratize alignment data.

### ORPO: Monolithic Preference Optimization without Reference Model
*2024 · data · `data_ORPO_2403.07691.txt` · arXiv [2403.07691](https://arxiv.org/abs/2403.07691)*

ORPO relies on two pairwise preference datasets: Anthropic's HH-RLHF and Binarized UltraFeedback (~61K instances). Instances are filtered out where chosen equals rejected or where either response is empty; prompts longer than 1024 tokens are dropped, and sequences are truncated/padded to 1024 tokens (HH-RLHF) or 2048 tokens (UltraFeedback). Mistral-ORPO-β additionally uses Argilla's cleaned ultrafeedback-binarized-preferences-cleaned variant, demonstrating that data quality yields further gains (~91% AlpacaEval1.0 / 12.20% AlpacaEval2.0) at similar dataset size. Separate reward models are trained for one epoch: OPT-350M (RM-350M, used for PPO) and OPT-1.3B (RM-1.3B, used only to evaluate generations).

- Two preference sets: Anthropic HH-RLHF + Binarized UltraFeedback (~61K)
- Filters chosen==rejected / empty responses; drops >1024-token prompts; pads to 1024 (HH) / 2048 (UF)
- Mistral-ORPO-β uses Argilla's cleaned variant, showing data quality lifts results at equal size
- Separate RMs: OPT-350M (for PPO), OPT-1.3B (for eval), each trained one epoch

**Key results:** Mistral-ORPO-β (7B): 12.20% AlpacaEval2.0, 7.32 MT-Bench, 66.19% IFEval instruction-level loose—surpassing Zephyr-β and Llama-2-Chat (13B) with single-epoch training on UltraFeedback alone.

*Evolution:* Builds on DPO's reference-free spirit and the unlikelihood-training tradition, reacting against the unstable multi-stage SFT→RLHF/PPO pipeline and helping popularize single-stage, reference-free, data-quality-focused preference tuning in 2024.

### Scaling Laws for Data Filtering—Data Curation cannot be Compute Agnostic
*2024 · data · `data_ScalingLawsFiltering_2404.07177.txt` · arXiv [2404.07177](https://arxiv.org/abs/2404.07177)*

Using the DataComp medium-scale pool of 128M image-caption pairs as the unfiltered base, the work partitions it into quality buckets via two ranking metrics: CLIP score (image-caption similarity from a pretrained CLIP L/14) and T-MARS score (CLIP score after masking OCR text). LAION filtering retains ~10% with CLIP score > 0.28; T-MARS retains ~30%. Buckets are formed by score percentile (Top 10%, 10–20%, 20–30%, 30–40%), each ~12.8M samples. The central thesis is the quality-quantity tradeoff (QQT): high-quality subsets are small and lose utility when repeated, so the optimal filtering aggressiveness depends on total compute (32M–640M samples seen). DataComp enforces NSFW filtering and face blurring. Scaling laws are validated on Cherti et al. (2023) models spanning pool sizes 80M–2B and compute 3B–34B.

- 128M DataComp pool bucketed by CLIP score and T-MARS (CLIP after OCR masking)
- Percentile buckets (Top 10/10–20/20–30/30–40%), each ~12.8M samples
- QQT thesis: optimal filtering aggressiveness depends on total compute (32M–640M)
- Validated on Cherti et al. ViT-B/16 and B/32 at 3B–34B compute

**Key results:** At 128M compute, LAION filtering gives +7.5% avg zero-shot accuracy over 18 tasks versus no-filtering, but beyond ~450M samples seen, unfiltered common crawl outperforms LAION-filtered data; the scaling laws predict the Pareto-optimal filtering strategy across 32M–640M compute.

*Evolution:* Extends Kaplan/Chinchilla scaling laws and Muennighoff's >4-epoch observation to heterogeneous web data and contrastive CLIP training, reacting against compute-agnostic filtering practices (LAION, T-MARS, DFN) and motivating compute-adaptive data curation.

### Self-Rewarding Language Models
*2024 · data · `data_SelfRewarding_2401.10020.txt` · arXiv [2401.10020](https://arxiv.org/abs/2401.10020)*

Seed data is drawn from Open Assistant. The IFT seed is 3,200 examples (first English conversational turns, human rank-0/high quality). The EFT seed is built from Open Assistant's multiple ranked responses per prompt: the SFT baseline generates CoT justifications plus a score out of 5 (a 5-point rubric over relevance, coverage, usefulness, clarity, expertise); a generated evaluation is kept only if its score ranking matches human rankings, then resampled to avoid skew (most score 4), yielding 1,630 train + 541 eval examples. Self-generated AIFT preference pairs number 3,964 (M1→M2) and 6,942 (M2→M3), formed from the highest/lowest of N=4 candidates (T=0.7, p=0.9) judged 3× and averaged; tied-score pairs are discarded. New prompts use Self-Instruct few-shot prompting with ROUGE-L, keyword, and length filtering. Without EFT, scores collapse to 4 and only 541/429 pairs are collectible.

- IFT seed 3,200 Open Assistant English turns; EFT seed 1,630 train + 541 eval, score-matched to humans
- AIFT preference pairs: 3,964 (M1→M2), 6,942 (M2→M3) from N=4 candidates judged 3×
- New prompts via Self-Instruct + ROUGE-L/keyword/length filtering
- Without EFT, scores collapse to 4 and few pairs are collectible

**Key results:** Llama 2 70B Self-Rewarding M3 reaches 20.44% AlpacaEval 2.0 win rate over GPT-4 Turbo, beating Claude 2 (17.19%), Gemini Pro (16.85%), and GPT-4 0613 (15.76%); reward-modeling pairwise accuracy rises each iteration (65.1% SFT → 81.7% M3).

*Evolution:* Folds the reward model into the policy so it co-improves across DPO iterations rather than staying frozen, a 2024 companion to SPIN's reward-model-free self-play that motivates later scalable self-alignment and iterative-RLAIF work.

### SimPO: Simple Preference Optimization with a Reference-Free Reward
*2024 · data · `data_SimPO_2405.14734.txt` · arXiv [2405.14734](https://arxiv.org/abs/2405.14734)*

Preference data is built on top of UltraFeedback prompts, used in two ways. In the Base setup, the SFT model is trained on UltraChat-200K, then preference optimization runs on the original UltraFeedback pairs (following Zephyr). In the Instruct setup, the authors regenerate an on-policy preference dataset: for each UltraFeedback prompt they sample 5 responses from the off-the-shelf instruct model at temperature 0.8, score them with PairRM, and keep the highest-scoring as y_w and lowest as y_l; generation is a single pass, not iterative. For the strongest Gemma-2-9B-it and an updated Llama-3-Instruct-v0.2 run, they swap the scorer for the stronger ArmoRM, which markedly lifts quality (e.g., +9.0 AlpacaEval 2 LC). No explicit de-duplication or filtering beyond reward-model ranking is described.

- UltraFeedback prompts; Base uses original pairs after UltraChat-200K SFT (Zephyr-style)
- Instruct setup: 5 on-policy responses/prompt at T=0.8, PairRM-scored, keep highest/lowest
- Strongest runs swap scorer to ArmoRM (+9.0 AlpacaEval 2 LC)
- No dedup/filtering beyond RM ranking; single-pass, non-iterative generation

**Key results:** Gemma-2-9B-it-SimPO reaches 72.4% LC win rate on AlpacaEval 2, 59.1% WR on Arena-Hard, and ranks 1st among <10B models on Chatbot Arena, while cutting DPO runtime ~20% and GPU memory ~10%.

*Evolution:* Builds directly on DPO and contemporaneous reference-free work such as ORPO, reacting to DPO's reward-generation mismatch and the cost of carrying a reference model, becoming a widely adopted lightweight DPO replacement in 2024.

### Tülu 3: Pushing Frontiers in Open Language Model Post-Training
*2024 · data · `data_Tulu3_2411.15124.txt` · arXiv [2411.15124](https://arxiv.org/abs/2411.15124)*

Tülu 3 targets seven core skills (knowledge recall, reasoning, math, coding, instruction following, chat, safety). Prompts come from public datasets — WildChat, OpenAssistant, No Robots, FLAN v2, UltraFeedback, OpenMathInstruct 2, NuminaMath-TIR, Evol-CodeAlpaca, Aya, SciRIFF, TableGPT — plus large-scale persona-driven synthetic generation (~250K personas from Persona Hub, with GPT-4o writing problems/solutions and Claude-3.5-Sonnet for code). This yields ~939K SFT samples, ~354K preference pairs (271K for 8B, 334K for 70B), and ~30K verifiable RLVR prompts (GSM8K 7,473; MATH 7,500; IFEval 14,973). Preference data combines on-policy Tülu-3-SFT generations with off-policy responses from a 22-model pool, judged by GPT-4o (1–5 across helpfulness, instruction following, honesty, truthfulness) then binarized. All prompts are decontaminated via 8-gram matching (>50% token overlap); datasets with >2% overlap are removed or filtered.

- Public prompts + persona-driven synthetics (~250K Persona Hub personas, GPT-4o/Claude-3.5-Sonnet)
- ~939K SFT, ~354K preference pairs (271K/334K for 8B/70B), ~30K RLVR prompts
- Preference pairs mix on-policy + 22-model off-policy, GPT-4o-judged and binarized
- 8-gram decontamination; datasets with >2% eval overlap removed/filtered

**Key results:** Tülu 3 70B averages 76.2 on Tülu 3 Eval, surpassing Llama 3.1 70B Instruct (74.1), GPT-4o-mini (69.6), and Claude 3.5 Haiku (75.3); RLVR improves the 8B model on GSM8K (84.3→87.6), MATH (42.0→43.7), and IFEval (81.1→82.4).

*Evolution:* Extends the open post-training line of Tülu 2/Zephyr-β toward closed-style multi-stage recipes, borrowing persona-driven synthetic data and formalizing verifiable-reward RL as RLVR, anticipating the 2024–25 wave of RLVR methods (GRPO, DeepSeek-R1).

### DeepSeek-V3 Technical Report
*2024 · report · `report_DeepSeek-V3_2412.19437.txt` · arXiv [2412.19437](https://arxiv.org/abs/2412.19437)*

Pre-training uses a 14.8T-token corpus with an increased math/programming ratio, multilingual coverage beyond English and Chinese, and a refined pipeline that minimizes redundancy while preserving diversity; document packing (no cross-sample attention mask) and a Fill-in-Middle (PSM) strategy at rate 0.1 are used, with a 128K-vocab byte-level BPE tokenizer. For post-training, 1.5M SFT instances span multiple domains. Reasoning data (math, code-competition, logic) is generated by an internal DeepSeek-R1 model: per-domain expert models (trained with SFT+RL) act as data generators, producing both <problem, original response> and <system prompt, problem, R1 response> samples, after which rejection sampling selects concise, high-quality data. Non-reasoning data (writing, role-play, simple QA) is generated by DeepSeek-V2.5 and verified by human annotators. The model-based reward model is trained from the V3 SFT checkpoint on preference data that includes the CoT leading to the reward, mitigating reward hacking.

- Pre-training 14.8T tokens; increased math/code ratio, multilingual; document packing + FIM (PSM) at 0.1
- Post-training 1.5M SFT instances; reasoning data generated by internal DeepSeek-R1 + per-domain expert models
- Rejection sampling selects concise high-quality reasoning samples; non-reasoning from DeepSeek-V2.5, human-verified
- Model-based RM trained from V3 SFT checkpoint on preference data including CoT leading to reward

**Key results:** DeepSeek-V3 (671B-total/37B-active MoE) achieves AIME 2024 39.2 Pass@1, MATH-500 90.2 EM, and 85.5 win rate on Arena-Hard (first open-source model above 85%), competitive with GPT-4o and Claude-3.5-Sonnet; full training costs only 2.788M H800 GPU-hours.

*Evolution:* Pioneers distilling long-CoT reasoning from the R1 series into a standard aligned MoE model via SFT/RL-generated data, anticipating the 2025 wave of distilling strong reasoning models into efficient base models.

### Gemma 2: Improving Open Language Models at a Practical Size
*2024 · report · `report_Gemma2_2408.00118.txt` · arXiv [2408.00118](https://arxiv.org/abs/2408.00118)*

Pre-training uses primarily-English data from web documents, code, and science articles: the 27B model trains on 13T tokens, 9B on 8T, and 2B on 2T, reusing the Gemma 1/Gemini SentencePiece tokenizer (256K vocab, split digits, byte-level). Filtering reuses Gemma 1 techniques: reduce unsafe utterances, strip personal/sensitive data, decontaminate evaluation sets, and minimize recitation. Post-training data extends Gemma 1.1's with internal and external public data; only the prompts (not answers) from LMSYS-chat-1M are used. Synthetic data passes multiple filtering stages removing personal information, unsafe/toxic outputs, mistaken self-identification, and duplicates; subsets that encourage in-context attribution, hedging, and refusals are added to improve factuality without hurting other metrics. Final mixtures and hyperparameters were selected by ablations similar to Gemini 1.0.

- 27B/9B/2B train on 13T/8T/2T tokens; reused 256K-vocab SentencePiece tokenizer
- Filtering: reduce unsafe, strip PII, decontaminate evals, minimize recitation (Gemma 1 techniques)
- Post-training extends Gemma 1.1; only prompts (not answers) from LMSYS-chat-1M used
- Synthetic data multi-stage filtered (PII, unsafe, duplicates); attribution/hedging/refusal subsets added

**Key results:** Gemma 2 27B-IT reaches LMSYS Chatbot Arena Elo 1218, beating Llama-3 70B-IT (1206); 9B-IT Elo 1187 matches GPT-4-0314, and distillation lifts a 2B model from 60.3 to 67.7 average on 3 benchmarks.

*Evolution:* Repurposes knowledge distillation as a pre-training substitute for next-token prediction to push small models past compute-optimal token counts, helping motivate teacher-distilled pre-training and weight-averaged RLHF recipes.

### InternLM2 Technical Report
*2024 · report · `report_InternLM2_2403.17297.txt` · arXiv [2403.17297](https://arxiv.org/abs/2403.17297)*

Pre-training uses 2.0–2.6T tokens across text, code, and long-context data. Text (webpages 86.5% of bytes, plus books, papers, patents) is processed via Trafilatura extraction, pycld2 language ID, heuristic rule cleaning, MinHash LSH dedup (128 hashes, 5-gram, 0.7 threshold), safety filtering (13M blocked domains, 36K blocked words, BERT toxicity/pornography classifiers), and quality filtering with BERT ads/fluency classifiers trained on manual annotations. Code data (105.6GB high / 440.1GB moderate / 83.85GB low-dropped) is unified to markdown, deduped at file level, quality-scored via an iterative annotation loop, and dependency-sorted by topological sort so whole repositories become single long files. Long-context data (>32KB) uses length, statistical, and conditional-perplexity filters and is a subset of the standard corpus (learned at least twice). A 24B-token capability-enhancement set mixes retrieved STEM/special-domain and high-quality HuggingFace data; test-related data is filtered with a contamination check. SFT uses 10M screened ChatML instructions; the reward model uses 2.4M binarized preference pairs. The tokenizer is cl100k-derived (~60K EN/code + 32K Chinese).

- Pre-training 2.0–2.6T tokens; text processed via Trafilatura + pycld2 + MinHash LSH (128 hashes, 0.7) + BERT safety/quality filters
- Code: 105.6GB high quality, file-level dedup, topological dependency sort into repo-long files
- Long-context >32KB via length/statistical/conditional-perplexity filters; 24B-token capability set
- SFT 10M ChatML instructions; RM 2.4M binarized preference pairs; cl100k-derived tokenizer

**Key results:** InternLM2-Chat-20B: AlpacaEval win rate 21.8, GSM8K 79.6, MATH 32.4, HumanEval 67.7, MTBench 7.9, AlignBench 6.8, and near-perfect 200K Needle-in-a-Haystack, beating GPT-3.5 on reasoning.

*Evolution:* Reflects the 2024 open-LLM push for transparent data pipelines and staged alignment, with COOL RLHF refining separate helpful/harmless RMs into a single system-prompt-conditioned RM with multi-round online patching, anticipating later iterative/online RLHF and process-reward trends.

### The Llama 3 Herd of Models
*2024 · report · `report_Llama3_2407.21783.txt` · arXiv [2407.21783](https://arxiv.org/abs/2407.21783)*

Post-training data mixes human annotations, rejection-sampled outputs, and heavy synthetic generation. Preference data spans General English (82.0%), Coding (6.9%), Multilingual (5.2%), and Reasoning/tools (5.9%): annotators do multi-turn dialogs, rate preference on four strength levels, and edit the chosen response to yield edited>chosen>rejected triples. SFT data is 52.7% General English, 14.9% Code, 3.0% Multilingual, 8.1% Exam-like, 21.2% Reasoning/tools, 0.1% Long-context. Synthetic pipelines include code execution-feedback (~1M dialogs), language translation and backtranslation (>2.7M code examples), multilingual (7 languages), long-context QA/summarization/repo reasoning at 16K–128K, and MCTS-generated reasoning traces. Quality control uses rule-based cleaning, Llama-3-8B topic classification, RM + Llama-as-judge quality scoring, Instag + Llama difficulty scoring, and RoBERTa-cluster semantic deduplication; pre-training uses ~15T multilingual tokens with exact-match decontamination against benchmarks.

- Preference data 82% General English / 6.9% Code / 5.2% Multilingual / 5.9% Reasoning-tools; edited>chosen>rejected triples
- SFT mix 52.7% General English / 14.9% Code / 21.2% Reasoning-tools; synthetic code/multilingual/long-context/MCTS traces
- QC: rule cleaning, Llama-3-8B topic class, RM+Llama-judge quality, Instag difficulty, RoBERTa semantic dedup
- Pre-training ~15T multilingual tokens; exact-match decontamination against benchmarks

**Key results:** Llama 3 405B matches GPT-4 on human pairwise win rate (within margin of error) and is the best openly available model; HumanEval pass@1 89.0, MGSM 91.6 (best), and 100% Needle-in-a-Haystack retrieval to 128K.

*Evolution:* Refines the Llama-2 RLHF recipe and 2023 DPO trend, favoring stable SFT/RS/DPO over PPO while scaling human preference data to millions and leaning on self-generated synthetic data across six iterative rounds, foreshadowing the 2024-25 shift toward synthetic-data-driven post-training and RLVR.

### MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies
*2024 · report · `report_MiniCPM_2404.06395.txt` · arXiv [2404.06395](https://arxiv.org/abs/2404.06395)*

The stable pre-training stage uses ~1T tokens drawn mostly from open corpora: cleaned CommonCrawl-Chn, Dolma, C4, Pile, Wikipedia, Chinese books, Baidu Baike, Open Web Math, Arxiv, peS2o, Stack Exchange QA, and code from The Stack and StarCoder, deduplicated within and across corpora using MinHash. The decay (annealing) stage shifts to a more diverse, higher-quality mix including UltraChat, SlimOrca, OssInstruct, EvolInstruct, ShareGPT4, plus proprietary Math/Code/Logic/Knowledge SFT data (LeetCode, K12 textbooks). Two BPE tokenizers (sentencepiece) are built: 122,753 vocab for the 2.4B model and 73,440 for the 1.2B model. For the 128K variant, data is split into long (books/wikis/papers, 44%) and short (56%) subsets, supplemented with synthetic long QA. The emphasis is on the staged mix rather than a single pipeline, with high-quality SFT data concentrated in the decay phase.

- Stable stage ~1T tokens from open corpora (CC-Chn, Dolma, C4, Pile, Wikipedia, math/code), MinHash dedup
- Decay stage shifts to higher-quality mix (UltraChat, SlimOrca, EvolInstruct, ShareGPT4, proprietary SFT)
- Two BPE tokenizers: 122,753 vocab (2.4B) and 73,440 (1.2B)
- 128K variant: long (44%) + short (56%) + synthetic long QA

**Key results:** MiniCPM-2.4B surpasses Mistral-7B and Llama2-13B on average across C-Eval/CMMLU/MMLU/HumanEval/MBPP/GSM8K/MATH/BBH; the fitted scaling law yields a compute-optimal data/model ratio of ~192× versus Chinchilla's 20×.

*Evolution:* Reacts to Chinchilla's 20× data/model ratio by empirically recovering a much higher (~192×) compute-optimal ratio under modern overtrained configurations; its WSD scheduler and anneal-on-high-quality-data recipe presage later annealing-with-curated-data practices.

### LLM Pruning and Distillation in Practice: The Minitron Approach
*2024 · report · `report_Minitron_2408.11796.txt` · arXiv [2408.11796](https://arxiv.org/abs/2408.11796)*

The core data challenge is that the original pretraining corpora for the starting models (Mistral NeMo 12B and Llama 3.1 8B, pretrained on 15T+ proprietary tokens) are unavailable. All pruning and distillation experiments therefore use the Nemotron-4 curated continued-training (CT) dataset as a substitute. A small 1024-sample calibration subset drawn randomly from this dataset feeds the activation-based importance estimation. Teacher correction consumes ~100B tokens of this CT data, while distillation-based retraining uses 94B tokens for the Llama 4B students and 380B for MN-Minitron-8B. The authors hypothesize that sub-word token distribution mismatch between the original pretraining data and the distillation dataset degrades uncorrected-teacher guidance, motivating the correction phase. No de-duplication or filtering pipeline is described; data sourcing is inherited from the Nemotron-4 CT recipe.

- Original 15T+ pretraining corpora unavailable; Nemotron-4 curated CT dataset used as substitute
- 1024-sample calibration subset for activation-based importance estimation
- Teacher correction ~100B tokens; distillation retraining 94B (Llama 4B) / 380B (MN-Minitron-8B)
- No dedup/filtering pipeline; data sourcing inherited from Nemotron-4 CT recipe

**Key results:** MN-Minitron-8B matches Llama 3.1 8B accuracy across LM Evaluation Harness benchmarks using 40× fewer training tokens (380B vs 15T); Llama-3.1-Minitron-4B reaches near-teacher quality with 150× fewer tokens (94B vs 15T).

*Evolution:* Extends the original Minitron pruning+distillation recipe to the realistic case where the teacher's pretraining data is private, introducing teacher correction to handle sub-word token distribution mismatch and motivating later lightweight teacher adaptation work.

### Phi-4 Technical Report
*2024 · report · `report_Phi-4_2412.08905.txt` · arXiv [2412.08905](https://arxiv.org/abs/2412.08905)*

phi-4's training data is dominated by synthetic data: ~50 broad synthetic dataset types totaling ~400B unweighted tokens, generated via multi-agent prompting, self-revision workflows, and instruction reversal from seeds extracted from web, books, and code. Seeds use two-stage filtering (educational potential, then passage-level factual/reasoning scoring); Q&A seeds are difficulty-balanced via plurality/majority-voting over multiple answers. Organic data is meticulously curated: web dumps filtered with small (non-LLM) classifiers trained on ~10⁶ LLM annotations, with a specialized pipeline to amplify non-STEM content; multilingual data for 40+ languages via fastText (176 languages); custom HTML-to-text and format-specific parsers. The final pretraining mixture is 40% synthetic (~290B unique tokens, 13.8 epochs), 15% filtered web (~1.3T, 1.2 ep), 15% web rewrites (~290B, 5.2 ep), 20% code (~820B, 2.4 ep), 10% acquired sources (~580B, 1.7 ep). Hybrid 13-/7-gram decontamination runs against ~20 benchmarks. Post-training adds ~8B-token SFT plus DPO pairs from PTS and ~850K judge-guided pairs.

- ~50 synthetic dataset types, ~400B unweighted tokens via multi-agent prompting, self-revision, instruction reversal
- Seeds two-stage filtered (educational potential, factual/reasoning); Q&A difficulty-balanced by voting
- Organic web filtered by small classifiers trained on ~10⁶ LLM annotations; non-STEM amplification; fastText for 176 langs
- Final mix 40% synthetic / 15% web / 15% web rewrites / 20% code / 10% acquired; hybrid 13/7-gram decontamination

**Key results:** phi-4 (14B): MATH 80.4 and GPQA 56.1, surpassing its teacher GPT-4o (74.6 / ~50.6) and beating Qwen-2.5-14B-Instruct on 9/12 benchmarks; November 2024 AMC-10/12 average ~90+/150.

*Evolution:* Extends the Phi family's "textbooks are all you need" synthetic-data thesis beyond GPT-4 distillation, showing curated synthetic data plus token-level preference optimization (PTS) can let a 14B model surpass its teacher on STEM reasoning at ~10× lower cost than long-CoT models.

### Qwen2.5 Technical Report
*2024 · report · `report_Qwen2.5_2412.15115.txt` · arXiv [2412.15115](https://arxiv.org/abs/2412.15115)*

The pre-training corpus grows from 7T to 18T tokens, with Qwen2-Instruct serving as a multi-dimensional quality filter/scorer. Math and code data are imported from the Qwen2.5-Math and Qwen2.5-Coder corpora; synthetic math/code/knowledge data is generated with Qwen2-72B-Instruct and Qwen2-Math-72B-Instruct, then filtered by a proprietary general reward model and Qwen2-Math-RM-72B. Domain mixture is rebalanced via Qwen2-Instruct classification, down-sampling overrepresented web domains (e-commerce, social, entertainment) and up-sampling high-value ones (tech, science, academia). Post-training SFT exceeds 1M examples across long generation (back-translated queries from pre-training corpora), math CoT (rejection sampling), code (~40 languages, sandbox static checks + unit tests), instruction-following (code-based validation, execution-feedback rejection sampling), structured data, 70K logical-reasoning queries, cross-lingual transfer, and robust system prompts; only responses passing critic-model + multi-agent scoring are kept. Test contamination is removed via n-gram/LCS matching (LCS≥13 and ≥0.6× min length).

- Pre-training corpus scaled 7T→18T tokens; Qwen2-Instruct used as quality filter/scorer
- Synthetic math/code/knowledge from Qwen2-72B-Instruct + Qwen2-Math-72B-Instruct, filtered by RMs
- Domain rebalance: down-sample e-commerce/social/entertainment, up-sample tech/science/academia
- Post-training >1M SFT examples; only critic + multi-agent-scored responses kept; n-gram/LCS decontamination

**Key results:** Qwen2.5-72B-Instruct matches or exceeds Llama-3.1-405B-Instruct (~6× larger) on MMLU-redux 86.8 vs 81.6, plus MATH, MBPP, MultiPL-E, LiveCodeBench, Arena-Hard, and MTBench; post-training = >1M SFT examples, ~150K DPO pairs, then GRPO online RL.

*Evolution:* Extends the Qwen2/Qwen2.5-Math/Coder lineage with a deliberate offline-DPO-then-online-GRPO two-stage RL design and aggressive 18T-token data scaling; its finding that RM-benchmark scores fail to predict downstream RL quality motivates better reward-model evaluation.

### Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models — The Story Goes On
*2024 · report · `report_Skywork-Math_2407.08348.txt` · arXiv [2407.08348](https://arxiv.org/abs/2407.08348)*

The contribution is the 2.5M-instance Skywork-MathQA SFT dataset, generated almost entirely by GPT-4 (1106-preview, temperature 0.7). Seed problems come from the MATH training set (7,500 entries) plus non-proving problems from OlympiadBench, AGIEval, SciBench, and JEEBench (GSM8K is deliberately excluded). Three augmentation methods are combined ("Mix"): MetaMathQA-style (rephrasing, FOBAR backward reasoning, self-verification, response refinement), Evol-Instruct (up to 5-step evolutionary rewrite with 5 strategies), and Xwin-style question generation with self-correction. A core-set approach (Sener & Savarese) selects a diverse representative seed subset. Quality control includes 30-gram decontamination filtering against test sets (removing ~6K MATH-like samples, none for GSM8K); a stricter 10-gram filter proved too aggressive. A verifier-based correctness filter (fine-tuned Mistral-7B, ~80% accuracy) actually hurt performance by skewing toward easy problems, whereas a hard-problem selector helped.

- 2.5M-instance Skywork-MathQA SFT set, almost entirely GPT-4 (1106-preview, T=0.7)
- Seeds: MATH train (7,500) + OlympiadBench/AGIEval/SciBench/JEEBench (GSM8K excluded)
- Three augmentations "Mix": MetaMathQA-style, Evol-Instruct (5-step), Xwin-style + self-correction
- 30-gram decontamination (removes ~6K MATH-like); verifier filter hurt by skewing easy, hard-problem selector helped

**Key results:** Skywork-Math-Mistral-7B (SFT only on 2.5M synthetic GPT-4 data) achieves 51.2% on MATH and 83.9% on GSM8K, SOTA among models <10B and surpassing an early GPT-4 on MATH; Stage 2's 0.4M hard problems lift MATH Level 3-5 accuracy markedly.

*Evolution:* Pushes back against the LIMA "less is more" and math-as-emergent-ability beliefs by showing scaling synthetic SFT data on common 7B models rivals 120B-token continual pre-training (DeepSeekMath), motivating larger, harder synthetic-data scaling for math.

### Improve Mathematical Reasoning in Language Models by Automated Process Supervision
*2024 · rl · `rl_OmegaPRM_2406.06592.txt` · arXiv [2406.06592](https://arxiv.org/abs/2406.06592)*

The work uses the MATH dataset with the PRM800K train/test split: 12K training questions and a 500-problem MATH500 holdout. OmegaPRM runs MCTS with a search limit of 100 per question, yielding 15M raw per-step annotations that are down-sampled to 1.5M for PRM training — claimed as the largest such dataset to date. To reduce Monte Carlo estimation noise, questions are filtered by running k=32 rollouts and discarding those with no correct answer (too hard) or no wrong answer (too easy). Step segmentation is flexible: any consecutive token span is a valid step, targeting ~16 pieces per solution rather than rule-based newline splitting. Baselines are PRM800K (human) and Math-Shepherd (auto). Generator policies are instruction-tuned Gemini Pro (~51% MATH) and pretrained Gemma2 27B (4-shot).

- MATH PRM800K split: 12K train questions, 500-problem MATH500 holdout
- MCTS (limit 100/question) → 15M raw per-step annotations → 1.5M down-sampled for PRM training (largest to date)
- Questions filtered via k=32 rollouts: discard no-correct (too hard) / no-wrong (too easy)
- Flexible step segmentation (~16 pieces/solution, any token span) vs rule-based newline splitting

**Key results:** Gemini Pro: 51%→69.4% on MATH500 and 86.4%→93.6% on GSM8K; Gemma2 27B: 42.3%→58.2% on MATH500 and 74.0%→92.2% on GSM8K, via OmegaPRM-weighted majority voting; 1.5M auto-annotations collected at 75× the efficiency of brute-force Monte Carlo.

*Evolution:* Builds on PRM800K's human process supervision and Math-Shepherd/MiPS's per-step Monte Carlo automation, importing AlphaGo Zero's MCTS to make step-level reward data cheap and large-scale, predating and motivating the later PRM and RLVR wave.

### ReFT: Reasoning with Reinforced Fine-Tuning
*2024 · rl · `rl_ReFT_2401.08967.txt` · arXiv [2401.08967](https://arxiv.org/abs/2401.08967)*

Training and evaluation data are three math word-problem sets: GSM8K (7,465 N-CoT / 7,356 P-CoT train, 1,319 test), SVAMP (3,076 / 3,043 train, 1,000 test), MathQA (14,862 / 15,250 train, 1,605 test), plus a numeric MathQA variant (8,955 / 7,672 train, 1,605 test). Both natural-language CoT (N-CoT) and program-based Python CoT (P-CoT) annotations are generated via few-shot prompting of GPT-3.5-turbo following Jie et al. (2023). A key point: ReFT uses exactly the same training questions as SFT, with no extra or augmented questions. For reward-model-reranking experiments, 100 CoTs per training question are sampled from the warm-up checkpoint, deduplicated, and binary-labeled by comparing the extracted answer to ground truth. Offline self-training samples 100 CoTs/question at temperature 1.0, keeps only correct ones, and subsamples to 10 unique CoTs per question to balance difficulty.

- Three math sets: GSM8K, SVAMP, MathQA (+ numeric MathQA variant), with N-CoT and P-CoT annotations
- N-CoT/P-CoT generated via few-shot GPT-3.5-turbo (Jie et al. 2023); same questions as SFT, no augmentation
- RM-reranking: 100 CoTs/question from warm-up, deduped, binary-labeled vs ground truth
- Offline self-training: 100 CoTs/question at T=1.0, keep correct, subsample 10 unique/question

**Key results:** CodeLLAMA-7B + ReFT reaches 75.28 P-CoT accuracy on GSM8K vs 63.68 for SFT (+11.6), and 81.2 with reward-model reranking, surpassing GPT-3.5-turbo (78.0) using only a 7B model.

*Evolution:* An early-2024 demonstration that outcome-only RL (PPO with answer-derived rewards, no trained reward model) on the same SFT data can substantially beat SFT for math reasoning, anticipating the GRPO/RLVR direction later popularized by DeepSeekMath and DeepSeek-R1.

### Technical Report on Slow Thinking with LLMs: II — Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems
*2024 · rl · `rl_STILL-2_2412.09413.txt` · arXiv [2412.09413](https://arxiv.org/abs/2412.09413)*

Long-form thought data is distilled from two open o1-like systems — DeepSeek-R1-Lite-Preview (R1, API) and QwQ-32B-Preview (checkpoints) — not from o1, whose summarized thoughts are unsuitable for imitation. Problems are fed to both systems with multiple rollouts to create diverse responses, auto-labeled correct/incorrect against ground-truth answers. Format follows R1's structure using <|begin_of_thought|>/<|end_of_thought|> and <|begin_of_solution|>/<|end_of_solution|> tokens; QwQ responses get a model rollout to fill the solution part. Source problems span math (NuminaMATH MATH/Olympiad subsets, AIME 1983–2023 from AOPS), code (LeetCode 'Hard'), science (Gaokao/college exams and camel-ai), and puzzles (RiddleSense). Pre-processing includes deduplication plus rule-based filters (regex, n-gram) removing repetitions, gibberish, and mixed-language outputs, and discarding short examples. Perplexity and length drive selection; only several thousand instances are retained (1.1K and 3.9K variants).

- Long-CoT distilled from DeepSeek-R1-Lite-Preview (API) and QwQ-32B-Preview (checkpoints), not o1
- Multiple rollouts per problem, auto-labeled correct/incorrect vs ground truth; R1-format thought/solution tokens
- Problems span math (NuminaMATH, AIME), code (LeetCode Hard), science (Gaokao, camel-ai), puzzles (RiddleSense)
- Pre-processing: dedup + regex/n-gram filters, discard short; perplexity+length selection → 1.1K/3.9K variants

**Key results:** STILL-2 (Qwen2.5-32B-Instruct + 3.9K distilled SFT) reaches 90.2% MATH-OAI and 46.7% AIME2024 (vs 80.0/13.3 backbone), approaching o1-preview's 85.5/44.6 on math; exploration+self-improvement from only 1.1K seed instances lifts AIME from 33.3% to 40–46.7%.

*Evolution:* Within the 2024 o1-replication wave, evidences that small distilled long-CoT data plus rejection-sampling/DPO can elicit cross-domain slow thinking, rejecting test-time search in favor of distillation from newly-open R1/QwQ plus self-improvement.

### Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
*2024 · rl · `rl_TestTimeCompute_2408.03314.txt` · arXiv [2408.03314](https://arxiv.org/abs/2408.03314)*

The paper uses the MATH benchmark, specifically the 12K train / 500 test split from Lightman et al. (PRM800K). Data curation serves the test-time compute analysis: they sample 16 solutions per question from the few-shot prompted PaLM 2-S* base model and run 16 Monte-Carlo rollouts per step to obtain soft per-step correctness labels for unsupervised PRM training (Math-Shepherd style), discarding samples with unparseable final answers. They explicitly rejected the released PRM800K human/GPT-4 labels as ineffective due to distribution shift with PaLM 2. For revision-model SFT they sample 64 responses per question at higher temperature, keep valid ones, and post-hoc construct multi-turn trajectories by pairing each correct answer with 0–4 incorrect answers in context, selecting the last incorrect answer via character-level edit distance to correlate it with the target. A separate ORM verifier is trained on the revision model's own outputs.

- MATH PRM800K split (12K train / 500 test); 16 solutions/question from few-shot PaLM 2-S*
- 16 MC rollouts/step → soft per-step labels for unsupervised PRM (Math-Shepherd style); unparseable answers discarded
- Rejected released PRM800K human/GPT-4 labels due to distribution shift with PaLM 2
- Revision SFT: 64 responses/question at higher T, pair correct with 0–4 incorrect, last chosen by char-level edit distance

**Key results:** Compute-optimal test-time scaling outperforms best-of-N by up to 4× on MATH for both PRM search (16 vs 64 generations) and revisions (64 vs 256); in a FLOPs-matched evaluation, test-time compute with PaLM 2-S* beats a ~14× larger pretrained model on easy/medium MATH questions.

*Evolution:* An early systematic study of inference-time compute scaling and its tradeoff against pretraining FLOPs, anticipating the o1-style adaptive inference-compute wave and motivating distilling test-time outputs back into the base model for iterative self-improvement.

## 2025

### ACECODER: Acing Coder RL via Automated Test-Case Synthesis
*2025 · code · `code_ACECODER_2502.01718.txt` · arXiv [2502.01718](https://arxiv.org/abs/2502.01718)*

ACECODER builds ACECODE-87K through a fully automated test-case synthesis pipeline. Seeds are Magicoder-Evol-Instruct-110K, Magicoder-OSS-Instruct-75K, and bigcode/stack-dedup-python-fns; only Python questions containing a function/class survive (124K entries), which GPT-4o-mini rewrites into LeetCode-style prompts with ~20 imagined tests each. Test filtering uses Qwen2.5-Coder-32B-Instruct as a proxy solver: any test the proxy fails is removed, and questions with fewer than 5 surviving tests are dropped. Original solutions serve only as seeds.

- **Output:** 87.1K questions and 1.38M cleaned tests; a 200-sample human study found only 3 invalid tests.
- **RM data:** 16 programs sampled per question from Qwen2.5-Coder-7B-Instruct; pairs kept only when si>sj+0.4, si>0.8, sj>0, yielding ~300K on-policy preference pairs from 46,618 questions.

**Key results:** ACECODE-RM-7B boosts Llama-3.1-8B-Instruct by +8.4 avg (Best-of-N) and ACECODE-RM-32B by +10.7 across HumanEval/MBPP/BigCodeBench/LiveCodeBench; ACECODE-RM-32B tops RM-Bench at 76.1 avg, beating Nemotron-340B-Reward by 7.5 on coding. R1-style RL from Qwen2.5-Coder-7B-Base with rule-based pass rewards gives +25% HumanEval-plus and +6% MBPP-plus in just 80 steps (48 H100 GPU-hours). ACECODER-Rule lifts BigCodeBench-Instruct-Full from 40.2 to 43.2, approaching DeepSeek-R1-Distill-Qwen-32B (43.9).

*Evolution:* Building on DeepSeek-R1's verifiable-reward RL-from-base recipe, Tulu3/DeepSeekMath GRPO, and Magicoder's OSS-Instruct data synthesis, ACECODER reacts to the fact that general reward models (e.g., Skywork) fail on code due to missing test-grounded signals. By automating test-case synthesis at scale, it supplies the missing reward signal and motivates later code-RL work showing that very short RL runs from a base coder can rival distilled reasoning models.

### Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training
*2025 · code · `code_Agent-R_2501.11425.txt` · arXiv [2501.11425](https://arxiv.org/abs/2501.11425)*

All training and eval data is self-generated via MCTS over three AgentGym environments: WebShop (300 simulations), ScienceWorld (200), and TextCraft (200). MCTS collects bad/good trajectory pairs sharing a common prefix, separated by reward gap beta=0.2 and a quality lower bound alpha that rises across iterations (0.5→0.7→1.0) to push good trajectories toward optimal. Ten hand-written revision-thought templates splice bad prefixes to good suffixes; no human or expert-model annotation is used.

- **Yield:** thousands of revision+good pairs per env per iteration (e.g., WebShop iter3: 9000 revision / 2000 good; TextCraft iter3: 8000/4000).
- **Mixing:** following AgentTuning, agent data is mixed with ShareGPT at ratio eta=0.2. Test sets: 200 (WebShop), 200 (SciWorld), 100 (TextCraft).

**Key results:** Llama-3.1-8B + Agent-R (iter 3) averages 70.71 across WebShop/SciWorld/TextCraft (63.91 / 70.23 / 78.00), beating Direct-Revision (62.36), ETO (65.12), and GPT-4o, a +5.59% gain. Revision trajectories outperform self-generated optimal trajectories, and average revision length drops across iterations (TextCraft 8.3 -> 2.6), indicating earlier, more reliable error detection.

*Evolution:* Agent-R extends MCTS-driven agent training (e.g., Agent Q) and self-correction RL (SCoRe) by reacting against behavior cloning from all-correct expert trajectories, which cannot recover from errors in long-horizon interactive tasks. In the 2025 self-improving-agent wave, it motivates automatic step-level reflection-data construction and iterative self-play SFT without human or expert-model supervision.

### CODE I/O: Condensing Reasoning Patterns via Code Input-Output Prediction
*2025 · code · `code_CodeIO_2502.07316.txt` · arXiv [2502.07316](https://arxiv.org/abs/2502.07316)*

Sources total ~810.5K raw Python files: CodeMix (~427K from an in-house code-pretraining corpus, filtered by DeepSeek-Coder-V2-Lite-Inst function-completion success rate 10–90%), PyEdu-R (~369K reasoning-focused subset of Python-Edu/Stack-V2, excluding algorithms and non-reasoning), and ~14.5K from TheAlgorithms, ProjectEuler, and LeetCode/CodeWars/AtCoder/CodeChef/Codeforces/Edabit. DeepSeek-V2.5 refines each file into a unified format (cleaned reference code, JSON-serializable entrypoint, I/O description, rule-based input generator, query). Inputs are sampled and executed for ground-truth outputs, with a 5s runtime cap and size constraints (<1024 bytes, lists/dicts<20, strings<100 chars); functions using randomness are skipped.

- **Output:** 3.5M instances from 454.9K files at ~50/50 input/output split; CoTs synthesized by DeepSeek-V2.5.
- **CODE I/O++:** adds single-turn execution-feedback revision (~50% correct turn-1, ~10% revised turn-2); all responses, including incorrect, are kept. A 13-gram leakage check guards contamination.

**Key results:** CODE I/O++ lifts the average score across 14 reasoning benchmarks on all four base models vs single-stage instruction tuning: Qwen 2.5 Coder 7B 57.7 vs 54.8, LLaMA 3.1 8B 52.1 vs 49.3, DeepSeek-Coder-v2-Lite 16B 53.5 vs 51.6, Gemma 2 27B 61.5 vs 59.5, while outperforming larger baselines (OpenMathInstruct2 14M, WebInstruct 11.6M). CODE I/O++ systematically beats CODE I/O with balanced, non-regressing gains across symbolic, logic, math, science, and commonsense tasks.

*Evolution:* Builds on the code-pretraining-enhances-reasoning trend (Python-Edu, Code Llama) and execution-prediction learning (scratchpads, CRUXEval, NExT), reacting against math-only reasoning-data scaling (OpenMathInstruct2, WebInstruct). The authors position CODE I/O as orthogonal to inference-time scaling (o1, R1) and as a stronger reasoning base for later RL-based reasoning models in 2025.

### RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning
*2025 · code · `code_RAGEN_2504.20073.txt` · arXiv [2504.20073](https://arxiv.org/abs/2504.20073)*

Not a dataset-curation paper; the training "data" is self-generated rollout trajectories from the policy itself. Four environments span symbolic (Bandit, single-turn stochastic; Sokoban, multi-turn deterministic; Frozen Lake, multi-turn stochastic) and realistic (WebShop, multi-turn web shopping) settings, the first three minimal and decoupled from pretraining priors. Evaluation uses 256 fixed prompts per environment at T=0.5, up to 5 turns. Each batch samples P=8 prompts with N=16 rollouts/prompt and up to 5 turns/10 actions.

- **Rollout curation findings:** favor diverse initial states with multiple responses per prompt (best at 4 responses/prompt), moderate action budget (5–6 actions/turn), and fresh on-policy rollouts (Online-1).
- **StarPO-S:** further filters, keeping only the top p% (default 25%) of prompts whose rollouts have highest reward standard deviation. For the SFT baseline, BFS generates 1,000 training and 100 test ground-truth trajectories.

**Key results:** 0.5B StarPO-S reaches 20.70% Sokoban / 21.48% FrozenLake, matching zero-shot GPT-4o (27.73%/26.56%) and Qwen2.5-72B (19.53%/23.83%) with ~100x fewer params. Keeping the 50% highest-variance rollouts (StarPO-S) avoids collapse entirely in FrozenLake-PPO; default 25% filtering delays collapse across all tasks. Bandit generalizes 100% with reasoning vs 81.25% without.

*Evolution:* Extends single-turn RL-for-reasoning methods (PPO, GRPO from DeepSeek-R1/DeepSeekMath, DAPO's KL-removal and Clip-Higher) into multi-turn stochastic agent training, alongside the 2025 agentic-RL wave (Search-R1, WebAgent-R1, VAGEN, ArCHer). It surfaces the Echo Trap instability and motivates finer-grained, reasoning-aware reward shaping and turn-aware optimization for long-horizon LLM agents.

### SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
*2025 · code · `code_SWE-RL_2502.18449.txt` · arXiv [2502.18449](https://arxiv.org/abs/2502.18449)*

Training data is sourced from open GitHub software evolution: all GitHub events from GH Archive (Jan 2015–Aug 2024) plus git clones of 4.6M repositories, aggregated into 24M merged-PR instances reconstructing issue discussions, merge-base code contents, intermediate commits, and a cumulative oracle patch; all SWE-bench repositories are excluded to prevent contamination. To counter a learned bias of editing every shown file, Llama-3.1-70B-Instruct predicts relevant-but-unmodified files to add as context.

- **Filtering:** removes bot PRs ('dependabot', 'renovate', 'bump', 'automerge'), empty or huge-change PRs, and applies CodeLlama hunk filters (e.g., lock-file-only changes), yielding ~11M unique PRs.
- **Seed selection:** heuristics then select 273K high-quality seeds (linked issue describing a bug fix, programming-file changes) as the RL dataset. The SFT baseline uses Magicoder/OSS-Instruct-style synthetic data generated by Llama-3.3-70B-Instruct, mixed with Llama 3 coding/general SFT data (2B tokens).

**Key results:** Llama3-SWE-RL-70B reaches 41.0% pass@1 on SWE-bench Verified, SOTA for <100B models and comparable to GPT-4o, vs 36.2% for the SFT baseline. Repair-only with oracle files: 34.8 vs 29.6 (SFT) vs 5.4 (base Llama-3.3-70B). OOD generalization: MATH-strict 73.7 vs 63.2 base; CRUXEval-I 71.6 vs 60.5 base.

*Evolution:* Builds on DeepSeek-R1's rule-based-reward RL for math/competitive-code reasoning, extending it to real-world software engineering via PR-level software-evolution data and GRPO, without the proprietary-teacher distillation used by Lingma-SWE-GPT/SWE-Gym/SWE-Fixer. It is the first to show RL on SE artifacts yields emergent 'aha-moment' reasoning that generalizes to OOD math/code tasks, motivating RL over real-world code artifacts as a new post-training direction for 2025.

### Beyond Scaling Law: A Data-Efficient Distillation Framework for Reasoning
*2025 · method · `method_DataEfficientDistillation_2508.09883.txt` · arXiv [2508.09883](https://arxiv.org/abs/2508.09883)*

Math seed is s1k (1,000 diverse, hard problems; Muennighoff et al.); code seed is 1,000 Codeforces challenges (open-r1/codeforces). For each question the selected teacher samples M CoT trajectories, then a multi-step filter is applied: length (>16k tokens dropped), format (strict ``...`` pairing), and correctness (rule verification + LLM-as-judge against ground truth). Question compression keeps only hard items: questions whose student pass-rate exceeds a threshold (too easy) are removed, cutting the math corpus ~75% to ~237 samples and code to ~230.

- **Diversity enhancement:** inspired by RL diverse roll-outs, computes pairwise Levenshtein distance over trajectories and keeps the farthest P per question, re-expanding to ~965 (math) and ~925 (code).
- **Final curated sets:** total ~0.8K examples; a mixed math+code set uses 400+ of each. OOD/general benchmarks (MMLU, CMMLU, C-EVAL, BBH, MBPP, GSM8K, Aider) confirm general capability is preserved.

**Key results:** NTele-R1-32B-Math, distilled from DS-32B on ~0.8k curated examples, scores 81.87% AIME 2024 and 77.29% AIME 2025, beating both teacher models (QwQ-32B 76.25/67.30, DeepSeek-R1 79.2/70) and large-corpus baselines (Light-R1, Skywork-OR1). The mixed NTele-R1-32B doubles Aider pass@2 (12.4 to 25.8) while raising LCB hard to 30.94.

*Evolution:* DED extends the data-efficient SFT line (LIMA, s1, LIMO, Light-R1) and DeepSeek-R1 distillation, reacting against the emerging reasoning 'scaling law' that ever-larger CoT corpora are needed. By importing RL ideas (on-policy-style pass-rate filtering, diverse roll-outs) and reframing corpus quality via token entropy and PCA latent shift, it motivates teacher-selection and entropy-aware curation as cheaper levers than raw data scale.

### OpenSIR: Open-Ended Self-Improving Reasoner
*2025 · method · `method_OpenSIR_2511.00602.txt` · arXiv [2511.00602](https://arxiv.org/abs/2511.00602)*

Data curation is the central contribution: OpenSIR uses NO annotated training data, bootstrapping instead from a single trivial seed problem ("What is 1+1?"). The teacher generates k groups of G problems conditioned on reference problems sampled from an accumulating pool P_t; invalid-format problems are filtered, valid ones added back to the pool. Reference answers come from majority voting over G student attempts (no external verifier).

- **Seed robustness:** ablations show trivial arithmetic, a MATH geometry problem, and an AIME 2024 problem all yield near-identical results (29.57/29.72/29.38).
- **Diversity:** enforced via min cosine distance to pool embeddings (Linq-Embed-Mistral-7B). Eval data: seven math benchmarks (GSM8K, MATH-500, Minerva, OlympiadBench, College Math, AIME 2024, AIME 2025) and three general reasoning benchmarks (BBEH, MMLU-Pro, SuperGPQA). GRPO baselines use 7,473 GSM8K or 7,500 MATH examples for contrast.

**Key results:** OpenSIR averages +3.35 math and up to +4.79 general reasoning across four models, beating GRPO baselines trained on >7,000 annotated examples while bootstrapping from a single trivial seed. On DeepSeek-R1-Distill-Llama-8B it improves math avg 36.31→40.56 (+3.71) and general reasoning 21.62→26.41 (+4.79). The diversity reward nearly doubles unique concept coverage (3328→5914) and its removal drops general reasoning by 2.50 points.

*Evolution:* OpenSIR reacts to RLVR (DeepSeek-R1, o1) and verifier-free self-play (Absolute Zero, R-Zero, Spiral), diagnosing that prior self-play collapses to familiar concepts on already post-trained models and so yields marginal/negative gains. It operationalises the open-endedness thesis (Hughes et al. 2024) for LLM reasoning by jointly optimising difficulty calibration and embedding-based diversity, and in the 2025 context it motivates verifier-free, diversity-driven self-improvement while pointing to multi-domain open-ended learning as the next frontier.

### Revisiting Entropy in Reinforcement Learning for Large Reasoning Models
*2025 · method · `method_PositiveAdvantageReweighting_2511.05993.txt` · arXiv [2511.05993](https://arxiv.org/abs/2511.05993)*

Training uses the DAPO-Math-17K dataset with GRPO on Qwen2.5-Math-7B (and Llama-3.1-8B-Instruct for generalization). To study how data diversity affects entropy, the authors build subsets of identical size but differing diversity via K-means clustering (K=1,000 clusters on mean-pooled final-layer embeddings of Qwen2.5-Math-7B) versus random sampling, yielding subsets of 10,001, 5,031, 2,538, 1,246, and 616 samples by drawing from the top-M clusters.

- **Key finding:** diversity, not scale, governs entropy — K-means subsets yield lower entropy than random subsets, and a model trained on ~616 K-means samples matches one trained on the full ~17K set (Avg@64 ID 45.13 vs 44.12).
- **No new data** is generated; subset selection/diversity is the curation lever. EMA-smoothed final-step entropy is reported (smoothing coefficient 0.6).

**Key results:** Qwen2.5-Math-7B + Pos-Adv-Reweight (Entropy-guided): AIME 2024 Avg@64 34.38 / Pass@64 73.33; best average Avg@64 (ID 45.66, OOD 19.39), beating Clip-Higher on 6 of 7 benchmarks. ~616 K-means-selected samples match full DAPO-Math-17K (Avg@64 ID 45.13 vs 44.12), showing data diversity, not scale, drives RLVR performance. GRPO lifts AIME 2024 Avg@64 from 10.00 (base) to 28.75.

*Evolution:* Builds on the 2024-2025 RLVR/GRPO wave (DeepSeek-R1, DAPO's Clip-Higher, Cui et al.'s Clip-Cov/KL-Cov, He et al.'s adaptive entropy regularization) and the covariance-based entropy-collapse theory of Liu (2025) and Cui et al. (2025), refining their positive-advantage-token insight into a directly controllable per-token loss weight. It motivates finer-grained, entropy-targeted token reweighting for forthcoming agentic and long-context RL, noting the concurrent QwenLong-L1.5/AEPO work as a related direction.

### Toward Training Superintelligent Software Agents through Self-Play SWE-RL
*2025 · method · `method_SSR-SelfPlay-SWERL_2512.18552.txt` · arXiv [2512.18552](https://arxiv.org/abs/2512.18552)*

SSR's central data contribution is to eliminate human-curated training data: it assumes only a corpus of pre-built Docker images containing source repositories with dependencies installed, with no access to issue descriptions, existing tests, test runners, test parsers, or language-specific infrastructure. The bug-injection agent must itself discover how to run tests and build a parser. All training tasks are self-generated bug artifacts (bug_inject.diff, test_script.sh, test_files.txt, test_parser.py, test_weaken.diff) validated by execution-based consistency checks.

- **Validation checks:** test existence/coverage, parser validity, script validity, bug validity, test-weakening validity, inverse mutation testing. Failed solver attempts are recycled into 'higher-order' bugs (up to second order) to expand the distribution.
- **Eval:** SWE-bench Verified (500 human-verified issues) and SWE-Bench Pro public split (731 tasks), one attempt each, temperature 1.0, top-p 0.95, ~2% paired standard error. Training and baseline RL share identical CWM environment images.

**Key results:** SSR (CWM-sft 32B base) achieves +10.4 points self-improvement on SWE-bench Verified and +7.8 on SWE-Bench Pro, consistently beating the human-data baseline RL throughout training while using no human-labeled issues or test suites. Full self-play also beats both repair-only and injection-only ablations.

*Evolution:* SSR extends the 2025 agentic-SWE-RL wave (SWE-RL, DeepSWE, CWM) and corpus-grounded self-play (SPICE, Absolute Zero, R-Zero) by removing human-curated issues/tests entirely, framing bug injection/repair as a self-play game over real repositories. It is an early step toward open-ended, self-improving software agents but flags unresolved scaling instability and reward-hacking risks that later work must address.

### SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?
*2025 · method · `method_SWE-Bench-Pro_2509.16941.txt` · arXiv [2509.16941](https://arxiv.org/abs/2509.16941)*

SWE-Bench Pro contains 1,865 human-verified problems from 41 actively maintained repositories spanning consumer apps, B2B services, and developer tools, partitioned into a public set (731 instances, 11 GPL/copyleft repos, released on HuggingFace), a private held-out set (858 instances, 12 repos, for future overfitting checks), and a commercial set (276 instances, 18 proprietary startup repos purchased via partnership; results public, code private).

- **Contamination resistance:** exclusive use of strong-copyleft (GPL) licenses, which create legal barriers to inclusion in proprietary pretraining corpora, plus acquisition of private commercial codebases. Problems mined from commit-history pairs (base vs. instance commit); test patch is the test-file diff, gold patch the remaining diff.
- **Difficulty control:** trivial 1–10 LOC edits excluded; reference solutions average 107.4 LOC across 4.1 files, every task needs ≥10 LOC, 100+ tasks exceed 100 LOC. Each repo contributes 50–100 instances (cap 100) to prevent single-repo overfitting.

**Key results:** Claude Sonnet 4.5 tops the public set at 43.6% Pass@1 (N=731); the best models stay below 20% on the commercial set (Opus 4.1 17.8%, N=276). Top models score ~23% on SWE-Bench Pro versus >70% on SWE-Bench Verified, exposing a large capability gap on enterprise-grade tasks. Removing human augmentations collapses performance (GPT-5 25.9%->8.4%, Opus 4.1 22.7%->8.2%).

*Evolution:* SWE-Bench Pro extends SWE-Bench/SWE-Bench Verified (Jimenez et al., 2024) and the SWE-Agent line, reacting to two 2024-2025 concerns: benchmark contamination from permissively-licensed public repos and the saturation of SWE-Bench Verified (>70%) partly via trivial 1-2 line tasks. By offering a harder, contamination-resistant, enterprise-realistic yardstick, it complements contamination-focused efforts (SWE-Bench+, LiveBench) and motivates next-generation agentic RL training methods such as Agent-RLVR (Da et al., 2025).

### The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models
*2025 · method · `method_ValleyOfCodeReasoning_2510.06101.txt` · arXiv [2510.06101](https://arxiv.org/abs/2510.06101)*

Training data is sourced from OpenCodeReasoning2 (OCR2), which contains 34,125 unique competitive coding problems compiled from four sources. The authors build a 30K-example base set whose answers are generated by DeepSeek-R1-0528 and KAT-V1-40B (avg duplication count 7, both >70% on LCB), with ``-tagged reasoning traces. All splits use random sampling and are open-sourced.

- **Scaling study (RQ-1):** nested random subsets of 10K and 1K drawn from the 30K to preserve distribution and differ only in quantity.
- **Correctness study (RQ-2):** 13,583 examples from TACO (which supplies test cases) split by pass/fail into two 6K subsets of correct vs incorrect responses. **Difficulty study (RQ-3):** TACO difficulty labels partition examples into hard (hard/very_hard/medium_hard) and easy (easy/medium) groups, sampled into two mutually exclusive 4K subsets.

**Key results:** Qwen2.5-7B-Instruct LCB Pass@1: 0.126 baseline → 0.055 at 1K (valley) → 0.188 at 10K → 0.264 at 30K; Llama3.1-8B-Instruct reaches 0.299 at 30K, doubling its 10K score. Easy/medium questions beat hard (+41% vs +7%), and correct vs incorrect teacher responses yield identical gains (~50%).

*Evolution:* Builds on data-efficient reasoning distillation (s1, LIMO, Li et al.'s 'structure not content') and large-scale coding distillation (OpenThoughts, OCR, rStar-Coder), reacting to their focus on data endpoints rather than intermediate training dynamics. By exposing the non-monotonic 'valley' in small models, it motivates future study of medium-high and high (>100K) data regimes and stage-aware data curation.

### λ-GRPO: Unifying the GRPO Frameworks with Learnable Token Preferences
*2025 · method · `method_lambda-GRPO_2510.06870.txt` · arXiv [2510.06870](https://arxiv.org/abs/2510.06870)*

Follows the SimpleRL Zoo setting, training on GSM8K and MATH datasets. Data is partitioned into three difficulty tiers — Easy (GSM8K + MATH level 1), Medium (MATH levels 1–4), Hard (MATH levels 3–5) — and the Hard subset of 8,523 problems is used as the training set, reflecting SimpleRL Zoo's claim that data difficulty is crucial for zero-RL exploration. A qwen-boxed chat template (system/user/assistant role tags, final answer in `\boxed{}`) is used.

- **No new data generation, filtering, or de-duplication** is introduced; the contribution is orthogonal to data curation.
- Reported gains come without any modifications to the training data.

**Key results:** Qwen2.5-1.5B/3B/7B reach average accuracy 37.8/43.8/53.5 over 8 math benchmarks, +1.9%/+1.0%/+1.7% over vanilla GRPO and +1.3%/+1.0%/+1.6% over DAPO. λ-GRPO also maintains higher token-level entropy than DAPO (steps 80–160) at similar response length, with no data or compute changes.

*Evolution:* Builds on GRPO (DeepSeek-R1-Zero) and the length-bias fixes DAPO/Dr. GRPO, recasting their fixed token-aggregation heuristics as special cases of a learnable token-preference framework. Within the 2025 RLVR wave of GRPO variants (GMPO, GSPO, GiGPO, KRPO), it motivates adaptive, data-driven weighting rather than hand-designed normalization, pointing toward self-tuned aggregation for verifiable-reward RL.

### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
*2025 · report · `report_DeepSeek-R1_2501.12948.txt` · arXiv [2501.12948](https://arxiv.org/abs/2501.12948)*

RL data is rule-verifiable reasoning prompts (Chinese/English): ~26K math (exams/competitions, proofs excluded), 17K+8K code (algorithm competition + GitHub bug-fixing), 22K STEM multiple-choice, 15K logic (web + synthetic code-IO/puzzles), and 66K+12K general/harmlessness prompts. Rewards are rule-based (accuracy via answer-match/compiler + format with ``/`<answer>` tags); no neural RM for reasoning to avoid reward hacking.

- **Cold-start (R1):** a few thousand long-CoT traces sampled from R1-Zero (temp 1.0), filtered for correct answers (sympy parsing, repetition/language-mix checks), rewritten by DeepSeek-V3 to human-readable first-person style with human verification.
- **800K SFT set:** ~600K reasoning samples (rejection sampling from the first RL checkpoint, some judged by a DeepSeek-V3 generative RM) + ~200K non-reasoning (reused DeepSeek-V3 SFT + software-engineering data). RMs built on 66K helpful pairs and 106K safety prompts. Comprehensive 10-gram decontamination (~6M math texts removed).

**Key results:** DeepSeek-R1 (671B MoE / 37B active) scores 79.8% pass@1 on AIME 2024 (86.7% cons@16) and 97.3 on MATH-500, matching OpenAI o1-1217 (79.2%) and far exceeding DeepSeek-V3 (39.2). On Codeforces it reaches the 96.3rd percentile (rating 2029). Distillation transfers reasoning to small models: R1-Distill-Qwen-32B hits 72.6 AIME / 94.3 MATH, beating RL-only Qwen2.5-32B-Zero across all benchmarks.

*Evolution:* Builds on GRPO (DeepSeekMath, 2024) and the outcome-only RL/STaR line, reacting against SFT-first RLHF by showing pure RL on a strong base can elicit emergent long-CoT reasoning (the 'aha moment') without human traces, mirroring OpenAI o1's late-2024 reasoning-RL shift. Its 2025 release of open R1-Zero/R1 weights and distill recipes catalyzed the open-source reasoning-RL wave (Open-R1, SkyRL, DAPO-style variants) and motivated later work on reward design, distillation-versus-RL tradeoffs, and tool-augmented reasoning.

### DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
*2025 · report · `report_DeepSeek-V3.2_2512.02556.txt` · arXiv [2512.02556](https://arxiv.org/abs/2512.02556)*

Continued pre-training reuses the 128K long-context extension data distribution from DeepSeek-V3.1-Terminus. For post-training, data comes from specialist distillation plus a large-scale agentic task synthesis pipeline. Six specialist domains (math, programming, general logical reasoning, general agentic, agentic coding, agentic search) plus writing and general QA, each in thinking and non-thinking modes, are trained with large-scale RL and then used to distill domain data into the final checkpoint.

- **Agentic pipeline:** 1,827 environments and ~85,000 complex prompts — code agent (24,667 tasks, real GitHub issue-PR environments filtered by heuristics and LLM judgment, validated by F2P/P2F tests), search agent (50,275, multi-agent pipeline with a verification agent retaining only verifiable samples), general agent (4,417, auto-synthesized hard-to-solve/easy-to-verify environments retained via pass@100>0), and code interpreter (5,908).
- **Speciale:** adds the DeepSeekMath-V2 dataset and reward for proofs.

**Key results:** DeepSeek-V3.2-Speciale: AIME 2025 96.0 Pass@1, HMMT Feb 99.2, Codeforces rating 2701, gold medals at IMO/IOI/ICPC-WF/CMO 2025. DeepSeek-V3.2-Thinking matches GPT-5-High on reasoning (AIME 93.1 vs 94.6) with SWE-Verified 73.1 and Terminal Bench 2.0 46.4. DSA reduces 128K decoding cost from $2.4M to $0.6M per million tokens vs V3.1-Terminus.

*Evolution:* Builds on DeepSeek-R1's RL-for-reasoning, DeepSeek-V3/V3.1's MoE+MLA, DeepSeekMath's GRPO, and native sparse attention (Yuan et al. 2025), reacting to the widening open-vs-closed gap by scaling post-training RL compute beyond 10% of pre-training and synthesizing agentic environments at scale. It motivates future open-model work on token efficiency and world-knowledge breadth, showing scaled RL plus synthesized agent data can approach frontier parity (GPT-5, Gemini-3.0-Pro).

### GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models
*2025 · report · `report_GLM-4.5_2508.06471.txt` · arXiv [2508.06471](https://arxiv.org/abs/2508.06471)*

Pre-training corpus (23T tokens) spans webpages, social media, books, papers, and code. Inspired by Nemotron-CC, crawled web is bucketed by quality score: the highest-quality bucket is up-sampled (3.2 epochs) while the lowest is discarded. MinHash dedup is supplemented with SemDedup (embedding-based) to remove template-generated near-duplicates. Multilingual data comes from crawls plus FineWeb-2, filtered by an educational-utility classifier.

- **Code/math:** GitHub code is rule-filtered then tiered by language-specific quality models (high/medium/low) with Fill-in-the-Middle applied; code-related web docs retrieved via HTML tags + FastText classifier and re-parsed. Math/science docs are LLM-scored on educational content ratio and up-sampled above a threshold.
- **Post-training SFT:** rejection sampling (format/correctness/reward-model/tool-protocol filtering), prompt difficulty filtering (bottom-50% by response length removed), response-level scaling (4 responses per hard prompt), and an automatic 4-step agentic SFT pipeline (framework/tool collection, task synthesis, trajectory generation, multi-judge quality filtering).

**Key results:** GLM-4.5 (355B/32B-active MoE) scores 91.0% AIME 24, 70.1% TAU-Bench, and 64.2% SWE-bench Verified, ranking 3rd overall and 2nd on agentic benchmarks among all evaluated models. GLM-4.5-Air (106B/12B) is competitive with Qwen3-235B-A22B and MiniMax-M1 and ranks 6th overall, with both models on the SWE-bench-vs-parameters Pareto frontier.

*Evolution:* GLM-4.5 builds on the 2024-2025 open MoE and reasoning-RL wave (DeepSeek-V3/R1, Kimi K2, Qwen3, GRPO from DeepSeekMath) and Nemotron-CC-style data bucketing, while distinguishing itself by unifying agentic, reasoning, and coding capabilities via expert-iteration + self-distillation plus hybrid thinking/non-thinking modes. It motivates the next push toward parameter-efficient open frontier models and standardized agent RL infrastructure (Slime) for long-horizon tool-using agents.

### Gemma 3 Technical Report
*2025 · report · `report_Gemma3_2503.19786.txt` · arXiv [2503.19786](https://arxiv.org/abs/2503.19786)*

Pre-training token budgets scale with model size — 14T (27B), 12T (12B), 4T (4B), 2T (1B) — mixing text and images with expanded multilingual data balanced via a Unimax-inspired sampling strategy, tokenized with Gemini 2.0's 262k-vocab SentencePiece. Filtering strips unsafe utterances, PII, and sensitive content; eval sets are decontaminated and quality reweighing follows Sachdeva et al. Vision data maps each image to 256 soft tokens from a frozen 400M SigLIP encoder at 896×896 with Pan & Scan adaptive windowing. Post-training data filters PII, toxic outputs, mistaken self-identification, and duplicates while adding subsets encouraging in-context attribution, hedging, and refusals; reward signals combine weight-averaged reward models on human feedback, code-execution feedback (Rlef), and ground-truth math rewards (DeepSeek-R1, Tulu 3).

- Token budgets 14T/12T/4T/2T for 27B/12B/4B/1B; multilingual balance via Unimax sampling
- SigLIP 400M → 256 soft tokens/image at 896×896; Pan & Scan inference windowing
- Post-training filtering: PII/toxicity/self-identification/dedup; rewards from Rlef, DeepSeek-R1, Tulu 3

**Key results:** Gemma-3-27B-IT scores Chatbot Arena Elo 1338 (rank ~9), MATH 89.0, GSM8K 95.9, HumanEval 87.8, MMLU-Pro 67.5, and IFEval 90.4, comparable to Gemini-1.5-Pro and ahead of larger open models like DeepSeek-V3 (1318) and Gemma 2 (1220).

*Evolution:* Building on Gemma 2's distillation-based pre-training, Gemma 3 (2025) folds in the 2024 RL-alignment toolkit (BOND/WARM/WARP) and verifiable-reward signals popularized by DeepSeek-R1 and Tulu 3, packaging distillation-plus-RL post-training into lightweight multimodal open models.

### Kimi K1.5: Scaling Reinforcement Learning with LLMs
*2025 · report · `report_Kimi-K1.5_2501.12599.txt` · arXiv [2501.12599](https://arxiv.org/abs/2501.12599)*

The RL prompt set is curated for three properties: diverse coverage (STEM, coding, general reasoning; text and image-text QA), balanced difficulty (an SFT model generates 10 high-temperature answers; pass rate proxies difficulty), and accurate evaluability (excludes multiple-choice, true/false, and proof items; removes easy-to-hack prompts a no-CoT model solves within N=8). Coding rewards use CYaRon to auto-generate 50 test cases per problem, filtered against 10 ground-truth submissions; of 1,000 contest problems, 323 entered training. Math rewards use a CoT RM (~800k CoT examples, ~98.5% accuracy; adopted for RL) over a Classic value-head RM (~800k, ~84.4%). Vanilla SFT has ~1M text (500k general QA, 200k code, 200k math/science, 5k writing, 20k long-context) plus ~1M text-vision examples. Pretraining spans 5 language and 5 multimodal domains with rule-based, FastText, embedding-dedup, and LLM-based quality filtering.

- RL prompts: diverse, difficulty-balanced (10-attempt pass rate), accurately evaluable; excludes MC/T-F/proof
- Coding: CYaRon 50 tests/problem, 323/1000 kept; Math: CoT RM ~800k examples at 98.5%
- SFT ~1M text + ~1M text-vision; pretraining 5+5 domains with rule/FastText/embedding-dedup/LLM filtering

**Key results:** Kimi k1.5 long-CoT: 77.5 AIME 2024 Pass@1, 96.2 MATH-500 EM, 94th percentile Codeforces, 74.9 MathVista -- matching OpenAI o1.

*Evolution:* Builds on RLHF (InstructGPT) and OpenAI o1's long-CoT RL, formalizing RL as online mirror descent and arguing context-length scaling suffices without MCTS, value functions, or process reward models.

### Kimi K2: Open Agentic Intelligence (Technical Report of Kimi K2)
*2025 · report · `report_Kimi-K2_2507.20534.txt` · arXiv [2507.20534](https://arxiv.org/abs/2507.20534)*

Pre-training uses 15.5T tokens across four domains (Web Text, Code, Mathematics, Knowledge), reusing Kimi K1.5 pipelines with per-domain correctness/quality validation. To raise token utility, a rephrasing pipeline augments high-quality tokens: WRAP-inspired knowledge rephrasing (style/perspective-diverse prompts, chunk-wise autoregressive rewriting, fidelity checks) and SwallowMath-inspired math "learning-note" rephrasing plus cross-language translation, each corpus rephrased at most twice (SimpleQA rises 23.76→28.94). SFT data targets prompt diversity and response quality, generated by K1.5/in-house experts then LLM/human-judge filtered. A large agentic synthesis pipeline builds 3,000+ real MCP tools (GitHub) plus 20,000+ synthetic tools via hierarchical domain evolution, thousands of agents, rubric-based tasks, multi-turn trajectories with simulated users and a tool-simulator world model, LLM-judge filtering, and real Kubernetes sandboxes (10k+ concurrent). RL data spans math (NuminaMath, open datasets, pass@k difficulty filtering), instruction-following (expert + AutoIF-style), faithfulness, coding (open + synthetic + GitHub PRs/issues), and safety (seed + attack/target/judge evolution).

- 15.5T tokens across 4 domains; WRAP/SwallowMath rephrasing (≤2×/corpus) lifts SimpleQA 23.76→28.94
- Agentic data: 3k+ real + 20k+ synthetic MCP tools, multi-turn trajectories, K8s sandboxes at 10k+ concurrency
- RL data: math (NuminaMath, pass@k), code (open+synthetic+GitHub PRs), safety (attack/target/judge evolution)

**Key results:** Kimi-K2-Instruct (1.04T MoE, 32B activated, non-thinking) scores 65.8 SWE-bench Verified (71.6 multi-attempt), 47.3 SWE-bench Multilingual, 66.1 tau2-Bench, 76.5 ACEBench, 53.7 LiveCodeBench v6, 27.1 OJBench, 49.5 AIME 2025, 75.1 GPQA-Diamond.

*Evolution:* Builds on DeepSeek-V3's sparse-MoE+MLA architecture, Kimi K1.5's RL scaling, WRAP/SwallowMath data rephrasing, and the RLVR wave (DeepSeek-R1), pushing token-efficient Muon training and agentic RL with synthetic tool-use environments to trillion-parameter scale.

### Llama-Nemotron: Efficient Reasoning Models
*2025 · report · `report_Llama-Nemotron_2505.00949.txt` · arXiv [2505.00949](https://arxiv.org/abs/2505.00949)*

The open-sourced Llama-Nemotron-Post-Training-Dataset totals ~33M synthetic samples, dominated by Math (22M reasoning-on / 2.2M off, 66.8%), then Code (10M, 30.6%), Science (709K, 2.1%), Chat (40K), Instruction-Following (56K), and Safety (31K). Math problems come from AoPS forums (excluding Middle School), extracted/classified by Qwen2.5-32B; solutions are generated by DeepSeek-R1 (16 gens) and Qwen2.5-Math-7B (64 gens) and filtered by answer-match judged by Qwen2.5-32B, with benchmark decontamination. Code aggregates 28,904 deduped questions from TACO/APPS/CodeContests/CodeForces (<0.3% contamination), ~488K Python solutions from DeepSeek-R1 via SGLang, Tree-Sitter-validated; a 25k→736k scaling ablation showed non-plateauing gains. Science mixes StackOverflow QAs and Qwen2.5-generated MCQs (conditioned by Nemotron-4-340B topics), decontaminated against GPQA/MMLU/MMLU-Pro. Chat uses 20k ShareGPT/WildChat prompts refined by a Feedback-Edit-Select inference-time-scaling system with Llama-3.1-Nemotron-70B reward-model rejection sampling; reasoning-off responses are paired counterparts from Llama-3.1-Nemotron-70B-Instruct / Llama-3.3-70B-Instruct.

- ~33M synthetic: Math 22M/2.2M (66.8%), Code 10M (30.6%), Science 709K; all decontaminated
- Math: AoPS extracted by Qwen2.5-32B, solutions by R1 (16×) + Qwen2.5-Math-7B (64×), answer-match filtered
- Code: 28,904 deduped Qs; 25k→736k scaling ablation shows non-plateauing gains
- Chat via Feedback-Edit-Select with Nemotron-70B reward-model rejection sampling

**Key results:** LN-Ultra (253B) reaches GPQA-Diamond 76.0% and AIME24 80.8%, beating DeepSeek-R1 (71.5% / 79.8%) while fitting on a single 8xH100 node with 1.71x lower latency than Llama-3.1-405B.

*Evolution:* Builds directly on DeepSeek-R1's distill-then-RLVR recipe and the reasoning-teacher paradigm, while reacting to the inference-cost problem of long chain-of-thought models by marrying NAS/FFN-Fusion efficiency and a user-facing reasoning toggle into one family.

### Qwen3 Technical Report
*2025 · report · `report_Qwen3_2505.09388.txt` · arXiv [2505.09388](https://arxiv.org/abs/2505.09388)*

Pre-training uses 36T tokens across 119 languages (up from 29 in Qwen2.5), spanning coding, STEM, reasoning, books, multilingual text, and synthetic data. To expand the corpus, Qwen2.5-VL extracts text from PDFs (refined by Qwen2.5), and Qwen2.5/Qwen2.5-Math/Qwen2.5-Coder synthesize trillions of tokens (textbooks, QA, instructions, code). A multilingual annotation system labels 30T+ tokens by educational value, field, domain, and safety; unlike prior source/domain-level mixing, the mixture is optimized at instance level via small proxy-model ablations. For post-training, the Long-CoT cold-start set (math, code, logic, STEM) is query-filtered with Qwen2.5-72B-Instruct (removing unverifiable/easy items) and response-generated with QwQ-32B; 3,995 query-verifier pairs are collected for Reasoning RL. Stage-3 SFT fuses rejection-sampled thinking data with curated non-thinking data (coding, math, instruction-following, multilingual, creative writing, QA, role-playing), with extra translation tasks for low-resource languages.

- 36T tokens, 119 languages; Qwen2.5-VL PDF extraction + Qwen2.5 family synthesizes trillions of tokens
- 30T+ tokens labeled by educational value/field/domain/safety; instance-level mixture via proxy ablations
- Long-CoT cold-start filtered by Qwen2.5-72B-Instruct, responses by QwQ-32B; 3,995 query-verifier pairs

**Key results:** Qwen3-235B-A22B (Thinking) scores 85.7 AIME'24, 81.5 AIME'25, 70.7 LiveCodeBench v5, 2056 CodeForces rating, 70.8 BFCL v3, outperforming DeepSeek-R1 on 17/23 benchmarks with 60% activated params.

*Evolution:* Qwen3 builds on Qwen2.5 and the 2024-2025 reasoning-model wave (o1, DeepSeek-R1, QwQ), reacting to the cost of switching between separate chat and reasoning models by unifying both modes with a controllable thinking budget.

### Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning
*2025 · report · `report_Seed1.5-Thinking_2504.13914.txt` · arXiv [2504.13914](https://arxiv.org/abs/2504.13914)*

RL data splits into verifiable (definitive answers) and non-verifiable (human-preference) problems. Verifiable STEM: several hundred thousand competition-grade problems (math/physics/chemistry, math >80%) from open-source datasets and public/private competitions; cleaned by dropping incomplete items, removing too-easy ones (Doubao-Pro 1.5 worst-of-N score 1), cross-checking reference answers with SOTA models plus human experts, and augmenting multiple-choice into fill-in-the-blank with integer answers, yielding 100k STEM problems. Code data: competitive-programming tasks each with a description, unit tests, and a checker script, difficulty-filtered and run in an in-house sandbox. Logical puzzles: 22 tasks (24-point, mazes, Sudoku) with generators/verifiers and configurable difficulty, ~10k problems. Non-verifiable prompts come from Doubao-1.5-Pro RL data (creative writing, translation, QA, role-play); low-variance and already-solved prompts are discarded. SFT uses 400k instances (300k verifiable + 100k non-verifiable) produced via cold-start human annotation then rejection sampling with Seed-Verifier. A new BeyondAIME benchmark of 100 expert problems is released.

- Verifiable STEM: 100k problems after cleaning (drop incomplete/too-easy, MC→fill-in-blank integer answers)
- Code with description+unit tests+checker in sandbox; 22 logical-puzzle tasks, ~10k problems
- SFT 400k (300k verifiable + 100k non-verifiable); BeyondAIME (100 expert problems) released

**Key results:** Seed1.5-Thinking (20B-active/200B-total MoE) scores 86.7 on AIME 2024 (matching o3-mini-high), 55.0 on Codeforces avg@8, and 77.3 on GPQA, with +8.0% win rate over DeepSeek R1 on non-reasoning tasks.

*Evolution:* Builds on the o1/R1 test-time-scaling wave and the authors' own VAPO/DAPO work to make long-CoT RL stable and reproducible at MoE scale.

### AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning
*2025 · rl · `rl_AceReason-Nemotron_2505.16400.txt` · arXiv [2505.16400](https://arxiv.org/abs/2505.16400)*

Two verification-ready RL datasets are built and open-sourced. Math data combines DeepScaler and NuminaMath (algebra, combinatorics, number theory, geometry), with 9-gram filtering against common math benchmarks and exclusion of multi-sub-question, multiple-choice/true-false, proof-based, non-English, figure-referencing, or overly brief prompts. Because NuminaMath is OCR/parsing-noisy, each question is solved by DeepSeek-R1 with up to 8 attempts, keeping only majority-voted correct ones; problems needing fewer than 2,000 R1 tokens are dropped as too easy and 2,000-4,000-token responses downsampled, yielding ~49,000 verified problems. Code data comes from competitive-programming platforms (AtCoder, LeetCode, Aizu) in function-call and stdin/stdout formats, excluding multi-solution/interactive/platform-template problems and those with images, incorrect, or weak test cases; strong edge-case tests are curated. DeepSeek-R1-671B with 8 rollouts scores difficulty 0-8 (level-8 dropped), and n-gram (n=14) plus URL matching decontaminates against tests released after 2024-08-01, leaving 8,520 problems.

- Math: DeepScaler+NuminaMath, 9-gram decontam, R1 8-attempt majority-vote, ~49k verified problems
- Code: AtCoder/LeetCode/Aizu, R1-671B 8-rollout difficulty 0-8, n-14 + URL decontam → 8,520 problems
- Both datasets open-sourced

**Key results:** AceReason-Nemotron-14B reaches 78.6/67.4 avg@64 on AIME24/25 and 61.1/54.9 avg@8 on LiveCodeBench v5/v6, surpassing DeepSeek-R1-Distill-Qwen-32B and beating SOTA distillation models OpenMath-14B/32B and OpenCodeReasoning-14B.

*Evolution:* Builds on DeepSeek-R1's GRPO-plus-rule-based-verification recipe and the 2025 open-R1 replication wave (DeepScaleR, Skywork-OR1, DAPO, DeepCoder), directly countering the DeepSeek-R1/Llama-Nemotron claim that distillation beats RL for small/mid models.

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale
*2025 · rl · `rl_DAPO_2503.14476.txt` · arXiv [2503.14476](https://arxiv.org/abs/2503.14476)*

Data is sourced from the web and official competition homepages via web scraping plus manual annotation, then released as the DAPO-Math-17K dataset of 17K prompts. Because math answers span expressions, formulas, and numbers that are hard to parse with rules, the authors transform each problem so the expected answer becomes an integer (inspired by AIME): e.g., an answer of form a+c*sqrt(b) is rewritten so the target is a+b+c, eliminating formula-parser errors in the reward signal. The reward is rule-based and verifiable: +1 if the predicted answer is equivalent to the ground truth, -1 otherwise, avoiding reward-model hacking. Training focuses on math; evaluation is AIME 2024.

- DAPO-Math-17K: 17K prompts from web scraping + manual annotation, open-sourced
- Answer-integerization (AIME-style) eliminates formula-parser errors in the reward signal
- Rule-based verifiable reward (+1/-1), no reward model, avoiding hacking

**Key results:** DAPO trained on the Qwen2.5-32B base model reaches 50 on AIME 2024 (avg@32), surpassing DeepSeek-R1-Zero-Qwen-32B's 47 using 50% of the training steps.

*Evolution:* DAPO builds on GRPO (DeepSeekMath) and DeepSeek-R1's RL-from-base recipe, reacting to the reproducibility gap left by R1's withheld training details by open-sourcing the algorithm, verl-based code, and the DAPO-Math-17K dataset.

### LIMO: Less is More for Reasoning
*2025 · rl · `rl_LIMO_2502.03387.txt` · arXiv [2502.03387](https://arxiv.org/abs/2502.03387)*

Data curation is the central focus. The candidate pool draws from NuminaMath-CoT, DeepScaleR (~40k pairs), pre-2024 AIME, MATH, and Chinese elementary-to-undergraduate exam problems (tens of millions total). A coarse difficulty filter excludes any problem Qwen2.5-Math-7B-Instruct solves within 4 attempts; a finer pass uses DeepSeek-R1-Distill-Qwen-32B to sample 32 attempts per problem and keeps only those solved 1-3/32, yielding the 2,125-problem LIMO-Pool. N-gram deduplication against all eval benchmarks removes contamination. Reasoning chains are sampled from DeepSeek-R1, R1-Distill-Qwen-32B, and QwQ-32B, then scored by a rule-based system over four axes — Elaborated Reasoning (length, 30%), Self-Verification ("check"/"verify", 20%), Exploratory Approach ("perhaps"/"might", 25%), Adaptive Granularity ("therefore"/"since", 25%) — normalized by text length. The top-scoring chain per problem is taken, pairs ranked, and the top 800 form the LIMO dataset.

- 2-stage difficulty filtering: Qwen2.5-Math-7B (4 attempts) then R1-Distill-Qwen-32B (32 attempts, keep 1-3/32) → 2,125 LIMO-Pool
- Chains from R1/R1-Distill/QwQ-32B scored on 4 axes (reasoning/verification/exploration/granularity), top chain per problem
- Final LIMO: 800 pairs, n-gram dedup vs all eval benchmarks

**Key results:** LIMO (Qwen2.5-32B-Instruct + 800 curated SFT samples): 63.3% AIME24 (vs o1-preview 44.6%, QwQ-32B-Preview 50.0%, NuminaMath-100k SFT 6.5%) and 95.6% MATH500, using ~1% of prior approaches' data.

*Evolution:* Extends LIMA's "less is more" alignment lesson (Zhou et al. 2023) from instruction-following to mathematical reasoning, reacting against data-heavy SFT (NuminaMath, OpenThoughts) while riding the 2024-25 test-time-scaling/long-CoT wave (o1, DeepSeek-R1, QwQ).

### Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning
*2025 · rl · `rl_Logic-RL_2502.14768.txt` · arXiv [2502.14768](https://arxiv.org/abs/2502.14768)*

Training data is procedurally generated Knights and Knaves (K&K) logic puzzles (Xie et al. 2024), chosen over math datasets like GSM8K and Omni-MATH whose uncontrolled complexity variance confounds reasoning studies. Puzzles are templated for infinite variability and deterministic single-answer verification, minimizing reward hacking. Difficulty is precisely controlled via number of characters (2-8) and Boolean-operator complexity (1-4 combinations), enabling both curriculum design and clean out-of-distribution tests (e.g., 8-person puzzles held out). The main training corpus is roughly 5,000 synthetic samples spanning 3-7 person puzzles, presented at mixed difficulty. Evaluation uses in-distribution K&K accuracy across difficulty levels plus Super-OOD math benchmarks AIME 2021-2024 and AMC 2022-2023, with memorization probed via perturbed puzzles (statement swaps/reorders) and a Local Inconsistency Memorization score. No external SFT/CoT data is added.

- Procedural K&K puzzles, templated for infinite variability + deterministic single-answer verification
- Difficulty via #characters (2-8) and Boolean-operator complexity (1-4); ~5k samples, 3-7 person puzzles
- Memorization probed via perturbed puzzles + LiMem score; no external SFT/CoT data added

**Key results:** Qwen2.5-7B-Instruct-1M + Logic-RL, trained on ~5K K&K puzzles: K&K average accuracy 0.19 -> 0.89, AIME +125%, AMC +38% vs base.

*Evolution:* Building directly on DeepSeek-R1's unreleased rule-based RL recipe, Logic-RL (Feb 2025) is an early R1-replication study that uses a controlled synthetic logic corpus to show R1-like emergent reasoning (reflection, verification) in a 7B model and dissect training dynamics.

### Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
*2025 · rl · `rl_Open-Reasoner-Zero_2503.24290.txt` · arXiv [2503.24290](https://arxiv.org/abs/2503.24290)*

Training data comprises tens of thousands of curated question-answer pairs for math and general reasoning. Sources include AIME (up to 2023), MATH, the Numina-Math collection, Tulu3 MATH, OpenR1-Math-220k, and the AoPS forum, plus programmatically synthesized general reasoning tasks (logical puzzles, multi-step and counterfactual problems). Problems hard to verify with a rule-based reward, such as proof-oriented ones, are excluded. LLM-based filtering evaluates problem difficulty and removes samples with extreme pass rates to keep the set balanced. The flagship set is ORZ 57k; a data-scale ablation shows it sustains improvement whereas the 7.5k MATH train set plateaus early. A curation ablation finds English-only data yields better stability and final performance than English+Chinese. The authors release the dataset as the largest verified reasoning-RL set to date.

- ORZ 57k flagship; sources AIME/MATH/Numina-Math/Tulu3/OpenR1-Math-220k/AoPS + synthetic general reasoning
- Exclude proof-oriented; LLM difficulty filtering removes extreme pass-rate samples for balance
- English-only > English+Chinese; 57k sustains improvement vs 7.5k MATH plateau; released as largest verified RL set

**Key results:** ORZ-32B matches/exceeds DeepSeek-R1-Zero-Qwen-32B on AIME2024 (48.1 vs 47.0), MATH500 (92.2 vs 91.6), GPQA Diamond (55.5 vs 55.0) using 1/10 the training steps.

*Evolution:* ORZ builds on DeepSeek-R1-Zero's Reasoner-Zero paradigm (RL directly on a base model) but reacts against GRPO by reverting to vanilla PPO with a learned critic for better credit assignment; as the first fully open-source large-scale reasoning-RL framework it democratizes R1-Zero-style scaling.

### Process Reinforcement through Implicit Rewards
*2025 · rl · `rl_PRIME-ProcessReinforcement_2502.01456.txt` · arXiv [2502.01456](https://arxiv.org/abs/2502.01456)*

SFT warmup data: 230K examples (~229,763) from open-source datasets spanning math, coding, and biomedicine — MathInstruct-MATH, OpenMathInstruct-2, NuminaMath-CoT, Reasoning-001 (math); Code-Feedback, Magicoder, Magicoder-OSS (coding); UltraMedical (biomed). Completions are generated by LLaMA-3.1-70B-Instruct in an action-centric chain-of-thought format (avg response length ~1390 tokens). Ground-truth-answer datasets were deliberately reserved for RL (not SFT) to diversify RL exploration. RL data is curated from NuminaMath-CoT (~860K problems) and APPS/CodeContests/TACO/Codeforces, cleaned to 457K math + 27K coding problems with verifiable outcomes (LaTeX answers, test cases). Cleaning drops figure/proof problems, converts multiple-choice to direct QA via rule-based + Llama-3.1-8B filtering + LLM formatting, and validates solvability/correctness with QwQ-32B-Preview and Qwen2.5-Math-72B-Instruct using 5-attempt self-consistency. An ablation-only EurusPRM uses 500K response-level rollouts from UltraInteract/Numina. Online prompt filtering during RL keeps median-difficulty prompts.

- SFT 230K (math/coding/biomed); completions by LLaMA-3.1-70B-Instruct, avg ~1390 tokens; ground-truth sets reserved for RL
- RL: 457K math + 27K code from NuminaMath-CoT (~860K) + APPS/CodeContests/TACO/Codeforces; drops figure/proof, MC→direct QA
- Solvability validated by QwQ-32B-Preview + Qwen2.5-Math-72B (5-attempt self-consistency); online filtering keeps median-difficulty

**Key results:** Eurus-2-7B-PRIME (from Qwen2.5-Math-7B-Base): 26.7% pass@1 on AIME 2024 vs Qwen2.5-Math-7B-Instruct's 13.3%, +15.1% average over the SFT model across 7 reasoning benchmarks, beating it with 10% of its training data.

*Evolution:* PRIME builds on implicit reward modeling (Yuan et al. 2024b, 'Free process rewards without process labels') and DPO-style implicit rewards, reacting to DeepSeek-R1/Kimi K1.5's conclusion that PRMs are impractical for large-scale RL.

### Skywork Open Reasoner 1 Technical Report
*2025 · rl · `rl_Skywork-OR1_2505.22312.txt` · arXiv [2505.22312](https://arxiv.org/abs/2505.22312)*

Math data is drawn mainly from NuminaMath-1.5 (896K problems; subsets amc-aime, olympiads, olympiads-ref, aops-forum, cn-contest, inequalities, number-theory) plus DeepScaleR, STILL3-Preview-RL-Data, Omni-MATH, and pre-2024 AIME, yielding ~105K problems after filtering. Code data comes from LeetCode (2.7K) and TACO (11K), ~13.7K total. Selection criteria: verifiable (no proofs, no code without test cases), correct (re-extract answers via Math-Verify, validate test cases against solutions), and challenging (drop all-correct/all-incorrect). Preprocessing includes in-dataset and cross-dataset dedup, URL/figure removal, and decontamination against AIME24/25. Model-aware difficulty estimation uses N=16 (math) / N=8 (code) rollouts at temp 1.0, 32K max tokens, discarding 0/N and N/N. A human+LLM-as-judge pass (Llama-3.3-70B, Qwen2.5-72B; 32 votes/problem, keep ≥9) removes ~1-2K low-quality math items. Online filtering drops prompts the actor already solves perfectly each stage.

- Math ~105K (NuminaMath-1.5 896K + DeepScaleR/Omni-MATH/AIME); Code ~13.7K (LeetCode 2.7K + TACO 11K)
- Verifiable+correct+challenging selection; in/cross-dataset dedup, URL/figure removal, decontam vs AIME24/25
- Model-aware difficulty (N=16/N=8 rollouts), human+LLM-judge (32 votes, ≥9), online per-stage filtering

**Key results:** Skywork-OR1-32B reaches 82.2 AIME24, 73.3 AIME25, 63.0 LiveCodeBench (avg@4), surpassing DeepSeek-R1 and Qwen3-32B on math.

*Evolution:* Builds on DeepSeek-R1's rule-reward RL and DeepScaleR's multi-stage context-length curriculum, and engages with DAPO's clip-higher trick; unlike most R1 reproductions it applies RL to already-distilled long-CoT models rather than base models.

### s1: Simple test-time scaling
*2025 · rl · `rl_s1_2501.19393.txt` · arXiv [2501.19393](https://arxiv.org/abs/2501.19393)*

s1K is 1,000 question-reasoning-trace pairs distilled from Google Gemini Flash Thinking Experimental. The authors first pool 59,029 questions from 16 sources (NuminaMATH 30,660; MATH 11,999; OlympicArena 4,250; OmniMath 4,238; AGIEval 2,385; AIME 1983-2021; TheoremQA; USACO; JEEBench; GPQA; SciEval; LiveCodeBench; plus two original sets: s1-prob, 182 Stanford Statistics PhD qualifying-exam problems, and s1-teasers, 23 hard quant brain-teasers). Three-stage filtering: Quality drops API errors and formatting issues (→51,581); Difficulty removes items solvable by both Qwen2.5-7B/32B-Instruct (graded by Claude 3.5 Sonnet) and uses reasoning-trace token length (→24,496); Diversity classifies questions into ~50 Mathematics Subject Classification domains via Claude, samples one domain uniformly then a problem with power-law (2^-rank) weighting favoring longer traces, plus 384 directly-selected high-quality items. All data is decontaminated against MATH500/GPQA Diamond/AIME24 via 8-gram overlap and deduplicated. s1.1 regenerates traces from DeepSeek r1.

- Pool 59,029 from 16 sources incl. s1-prob (182 Stanford PhD exams) + s1-teasers (23 quant brain-teasers)
- 3-stage filter: Quality →51,581; Difficulty →24,496; Diversity (~50 MSC domains, power-law 2^-rank weighting) →1,000
- Decontaminated via 8-gram vs MATH500/GPQA/AIME24; s1.1 regenerates traces from DeepSeek R1

**Key results:** s1-32B, SFT on only 1,000 traces for 26 min on 16 H100s, hits 56.7% AIME24 with budget forcing vs o1-preview's 44.6%, and 93.0% MATH500 vs 85.5%; budget forcing extrapolates AIME24 from 50% to 57%.

*Evolution:* Builds on LIMA's 'Superficial Alignment Hypothesis' (1K examples suffice) and Snell et al.'s test-time-compute scaling, reacting against the opaque, large-scale-RL recipes of OpenAI o1 and DeepSeek R1 by showing 1K distilled SFT plus a simple decoding trick matches o1-preview.

## 2026

### CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation
*2026 · method · `method_CUDA-Agent_2602.24286.txt` · arXiv [2602.24286](https://arxiv.org/abs/2602.24286)*

A scalable three-stage pipeline produces **CUDA-Agent-Ops-6K** (6,000 samples, released on HuggingFace). Seed crawling mines `torch.nn.Module` reference operators from the `torch` and `transformers` libraries; combinatorial synthesis then has an LLM sample at most 5 torch operator classes and stack them into fused multi-operator tasks, while `transformers` operators are used only as standalone modules. Rubric filtering keeps only operators executable in both Eager and Compile modes, deterministic, non-constant across inputs (anti-hacking), and with eager runtime in 1–100ms, with AST-based decontamination (PythonASTSimilarity) dropping any sample exceeding 0.9 max similarity to KernelBench.

- Final composition: 83.77% are 2-op compositions; the rest span 1–5 ops plus 1.18% transformers operators
- Fusion reshapes the optimization landscape by avoiding intermediate global-memory materialization and coupling stages through shared register/SMEM constraints

**Key results:** CUDA Agent (Seed1.6) on KernelBench: 98.8% pass rate, 96.8% faster rate vs torch.compile, 2.11x geomean speedup vs compile overall; 100%/100%/92% faster rate on Level-1/2/3.

*Evolution:* CUDA Agent extends the agentic-RL-for-coding line by combining DAPO-style asymmetric PPO clipping, ReAct/OpenHands agent loops, and Anthropic's Agent Skills paradigm, reacting to training-free CUDA systems and data-leakage-prone fine-tuners.

### Computer Environments Elicit General Agentic Intelligence in LLMs
*2026 · method · `method_LLM-in-Sandbox_2601.16206.txt` · arXiv [2601.16206](https://arxiv.org/abs/2601.16206)*

LLM-in-Sandbox-RL is trained exclusively on general, non-agentic context-based tasks: the seed data used to fine-tune the synthesizer in Instruction Pre-Training, spanning encyclopedia, fiction, expert materials, academic tests, news, social media, and trivia. Each instance pairs background material with related tasks of three types—free-form generation, multiple choice, and reasoning. Contexts are stored as files in the sandbox rather than the prompt; multi-document or long contexts are split into separate files, and single-file contexts are augmented with distractor files sampled from the same dataset to force active navigation, with no training data overlapping any evaluation benchmark.

- Data ablation contrasts this general data (placed in prompt vs. sandbox) against math data from DAPO and SWE data from R2E-Gym
- Evaluation uses AIME25, UGPhysics, ChemBench, MedXpertQA, AA-LCR, IFBench, and SWE-bench Verified; test problems are reframed to prevent benchmark hacking via internet access

**Key results:** Training-free: Qwen3-Coder-30B-A3B on AIME25 math +15.5% (26.0->41.5); MiniMax-M2 instruction following +14.4%; long-context tokens cut up to 8x (Qwen 100K->13K).

*Evolution:* Building on ReAct-style tool use, code-sandbox coding agents, SWE-RL, and Instruction Pre-Training data, this 2026 work isolates the computer environment itself as a source of general intelligence.

### LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification
*2026 · method · `method_LaTER_2605.07315.txt` · arXiv [2605.07315](https://arxiv.org/abs/2605.07315)*

LaTER introduces **LATENT-SWITCH-69K**, a 69,745-example supervised corpus built from reasoning traces sampled from Dolci-Think-SFT-32B and distilled by a stronger reasoning teacher. For each source problem the teacher first produces a short solution intuition (a few-sentence high-level plan, no full derivation), then a shorter explicit CoT conditioned on the problem plus that intuition; each retained record holds problem, intuition, compressed CoT, and final answer. The distilled-CoT compression ratio is mean 0.612 / median 0.569 (kept ~57–61% of original length).

- Difficulty split: easy 9.5%, medium 65.5%, hard 25.0%; domain mix math ~37%, code ~34%, science ~5%, remainder instruction-following/general knowledge
- Per-example latent budget tied to intuition length (~L/2 steps), giving mean 41.49 / median 40.00 latent steps, normalized to the 40–50-step range found optimal in training-free experiments
- Curriculum metadata groups by difficulty so early training can emphasize easier examples

**Key results:** Trained LaTER on Qwen3-14B reaches 80.0% on AIME 2025 (+10.0 over the CoT baseline's 70.0%) while using 33% fewer tokens (10,575 vs 15,730), with best/lowest-token results across 7 benchmarks.

*Evolution:* LaTER builds on the 2025-26 latent-reasoning line (Coconut, SoftCoT, Latent-SFT, CoLaR) and training-free variants, reacting against fully replacing explicit CoT by splitting labor: latent exploration for search, explicit CoT for verification.

### Polar: Agentic RL on Any Harness at Scale
*2026 · method · `method_Polar-AgenticRL_2605.24220.txt` · arXiv [2605.24220](https://arxiv.org/abs/2605.24220)*

Training and eval data come from the SWE-Gym/SWE-bench ecosystem. Online RL uses the NovaSky-AI/SkyRL-v0-293-data train split (293 tasks), reserving SWE-Bench Verified for evaluation. Offline SFT generation fans out a fixed Qwen3.5-122B-A10B checkpoint with the pi harness over 1,638 instances from seven SWE-Gym repos (getmoto/moto, python/mypy, conan-io/conan, pydantic/pydantic, iterative/dvc, pandas-dev/pandas, dask/dask). Acceptance uses a single binary verifier—the patch passes all FAIL_TO_PASS tests while keeping PASS_TO_PASS green—yielding 504 accepted trajectories (30.8%; per-repo 17.7%–53.6%) at ~64 GPU-hours.

- Accepted traces (avg 104 messages, 51 assistant turns) are released as a HuggingFace dataset under Apache-2.0 with a 90/10 train/test split stratified by repo
- The same primitives yield rejection-sampling, verifier-training, and preference data with no orchestration change

**Key results:** Qwen3.5-4B + standard GRPO via Polar improves SWE-Bench Verified pass@1 by +22.6 on Codex (3.8% to 26.4%); prefix_merging cuts a 3-step run from 189.5 to 35.2 min (5.39x).

*Evolution:* Polar builds on ProRL Agent's rollout-as-a-service by moving the integration boundary from the agent SDK to the model API endpoint, enabling training of closed-source/binary harnesses unchanged.

### R³: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning
*2026 · method · `method_R3-ReplayReflectionRanking_2601.19620.txt` · arXiv [2601.19620](https://arxiv.org/abs/2601.19620)*

Training uses the **DeepScaleR dataset**, a synthetic high-quality corpus of ~40,000 unique math problem-answer pairs restricted to the math domain. No new data is collected; instead a centralized sample buffer archives every generated (query, response, reward) triple, indexed by a unique identifier (UID) for retrieval. The outcome reward comes from a Qwen2.5-Math evaluator that symbolically compares the final answer to the reference, and a length reward `r_len = max(0, 1 - l/L_max)` with `L_max=32,768` is added for correct responses to encourage conciseness.

- For evaluation, challenging subsets of AIME24 and MATH are curated by keeping only queries the base model solved at most once across 16 samples
- No de-duplication or external filtering pipeline is described

**Key results:** R³-1.5B reaches 60.59 average across five math benchmarks (+12.78 over DeepSeek-R1-Distill-Qwen-1.5B), with AIME24 47.50 using only 7,574 tokens versus the base's 28.1 at 12,270 tokens.

*Evolution:* R³ builds on the GRPO/DAPO and DeepScaleR line of small-model reasoning RL and on experience-replay work, reacting specifically to GRPO's intra-group advantage collapse on hard tasks.

### ReSyn: Autonomously Scaling Synthetic Environments for Reasoning Models
*2026 · method · `method_ReSyn_2602.20117.txt` · arXiv [2602.20117](https://arxiv.org/abs/2602.20117)*

ReSyn replaces (Q,A) pairs with (Q,V) question-verifier pairs produced by a five-stage pipeline run entirely with Claude 3.5 Sonnet v2 (max_tokens=8192, temp=0.7). Keyword extraction yields ~100 seed keywords drawn from BBH and KOR-Bench subtasks plus manually added algorithm/data-structure terms; task synthesis prompts the LLM 8x per keyword to write Python environments implementing instance generator ρ_0, observation O, and reward R. A two-stage LLM-as-judge enforces reference-free verification, computational advantage, and well-specified questions (failures revised and re-evaluated), then instance generation samples n instances per difficulty level d∈{1..5}, and difficulty calibration keeps only environments with significant negative solve-rate/difficulty correlation (one-sided Wald test, α=0.05).

- From ~100 keywords, 418 environments survive, yielding 16K training and 500 validation (Q,V) pairs, matching SynLogic's scale
- Semantic-entropy analysis shows ReSyn is 8–15 points more diverse than SynLogic

**Key results:** Qwen2.5-7B-Instruct + ReSyn (DAPO, 400 steps): BBH 75.2 (+14% rel. vs 65.9), BBEH 14.3 (+27% rel. vs 11.2), AIME 2024 14.0 (+40% rel. vs 9.8), GSM8K 91.4 (vs 82.3).

*Evolution:* ReSyn builds on the RLVR wave and procedural reasoning environments (SynLogic, TinyZero, Reasoning Gym), but automates environment authoring via LLM-synthesized code verifiers, scaling task diversity >10x beyond SynLogic's 35 handcrafted tasks.

### Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability
*2026 · method · `method_SOAR_2601.18778.txt` · arXiv [2601.18778](https://arxiv.org/abs/2601.18778)*

Uses math benchmarks MATH, HARP, and OlympiadBench. For each, it samples 128 solutions per problem with Llama-3.2-3B-Instruct (temp 1.0, 1024 tokens) and retains only 0/128-success "fail@128" problems, then splits them 50-50 into train/test. Post-filter sizes: MATH 359/360 (drawn from the official 5000-problem MATH test split to avoid train-set memorization), HARP 714/714 (from 4768), OlympiadBench 158/158 (English, text-only, auto-verifiable subset of 674, held out for OOD transfer). The teacher generates synthetic question-answer pairs from a fixed prompt (no seed questions, preserving a data-free setup), filtered by format (question/answer tags, `\boxed{}`, symbolic-math-verifiable answer); invalid rollouts are rejection-sampled.

- Question correctness/well-posedness is later annotated by Claude-4.5-Sonnet as an oracle judge (only 32.8% of PQ answers are fully correct, 63% well-posed)

**Key results:** On fail@128 MATH with Llama-3.2-3B-Instruct, SOAR-PQ reaches 18.9±5.3% pass@32 vs 9.6±2.6% Hard-Only (~2x / +9.3%) and ~4x pass@1 (1.7 vs 0.5%); PQ-MATH recovers 75% of the full-curated-MATH upper bound's gain and transfers to held-out OlympiadBench.

*Evolution:* SOAR builds on asymmetric self-play (AlphaZero, Alice-Bob) and recent data-free LLM self-play (Absolute Zero, SeRL, R-Zero), reacting against intrinsic/proxy rewards that cause diversity collapse and instability.

### SWE-Fuse: Empowering Software Agents via Issue-free Trajectory Learning and Entropy-aware RLVR Training
*2026 · method · `method_SWE-Fuse_2603.07927.txt` · arXiv [2603.07927](https://arxiv.org/abs/2603.07927)*

SWE-Fuse builds training data on SWE-smith (50,137 instances from 128 executable GitHub repos), selecting 33,274 permissively-licensed, actively-maintained issues. Sandbox environments are reproduced from SWE-smith Docker images, keeping only repos that build and pass sanity checks; a custom sandbox manager supports high-concurrency RL rollout. Expert trajectories are distilled with Gemini 3 as teacher (Tmax=100 turns) under a ReAct format with an injected `<THOUGHT>` marker that externalizes reasoning for the student. Filtering removes git-exploitation (strips post-issue commits/logs, drops trajectories using `git show`/`git log`) and applies rule-based checks: ≥5 interaction rounds, intermediate reasoning steps present, strict ```bash``` format, English-only.

- The released SWE-Fuse trajectory dataset has 14,350 valid trajectories across 111 projects (~28 mean rounds, ~282M tokens)
- Mixes issue-described and issue-free samples (issue-free keep partial test cases for step-by-step debugging, only successful trajectories retained)

**Key results:** SWE-Fuse-Qwen3-32B resolves 60.2% of SWE-bench Verified issues (65.2% at TTS@8), SOTA among open-source ≤32B models; data scaling from 0 to 14k trajectories improves resolve rate 13.5% to 39.0% (2.9x).

*Evolution:* SWE-Fuse extends the RLVR-for-coding line by attacking real-world data noise—issue/solution misalignment—through issue-free trajectory learning, and grafts entropy-adaptive clipping onto RLOO.

### Synthetic Sandbox for Training Machine Learning Engineering Agents
*2026 · method · `method_SandMLE_2604.04872.txt` · arXiv [2604.04872](https://arxiv.org/abs/2604.04872)*

SandMLE constructs its training corpus from 60 MLE-bench seed questions (Medium/Hard/Dev splits) and uses a multi-agent pipeline to amplify them into 848 synthetic training tasks plus 64 held-out validation tasks. Four LLM roles collaborate: a Data Strategist extracts a Task DNA (modality, label cardinality), re-attributes domain, injects adversarial noise, and compiles a hidden rule `l=f(z)+epsilon` while capping each task's dataset to 50–200 samples; an ML Developer procedurally generates the micro datasets and locks progressive milestone thresholds S; an MLOps Engineer writes a deterministic evaluator with execution-based debug loops; a Technical Writer compiles the markdown task spec. An automated sanity check (strict monotonic threshold ordering vs. a dummy-submission baseline) discards corrupted tasks.

- Corpus spans healthcare (25%), retail (18.2%), manufacturing, IT; image (48.7%) and tabular (24.8%) modalities; classification (56.2%), regression, ranking
- Difficulty is validated by pairwise model win-rates reproducing the GPT-4o-mini < Gemini-2.5-Flash < DeepSeek-V3 < Claude-4.5-Sonnet ordering

**Key results:** SandMLE cuts per-rollout execution time 13.7x (196.17s -> 14.31s); on MLE-Bench-Lite, Qwen3-8B/14B/30B-SandMLE achieve 22.7%/22.7%/27.3% Any Medal (+66.9%/+24.7%/+100.7% vs Base).

*Evolution:* Building on trajectory-wise RL successes in SWE (DeepSWE) and web search (WebDancer) and on GRPO-style RLVR, SandMLE addresses the MLE-specific execution-latency bottleneck that had forced prior MLE-agent work back to SFT or off-policy async RL.

### Self-Verified Distillation: Your Language Model Is Secretly Its Own Synthetic Data Pipeline
*2026 · method · `method_SelfVerifiedDistillation_2605.26132.txt` · arXiv [2605.26132](https://arxiv.org/abs/2605.26132)*

Seed questions come from OpenThoughts (Guha et al., 2025), using only the question text and discarding provided answers/reasoning traces to fit a no-ground-truth setting: 53,125 math, 26,041 science, and 9,168 code questions spanning competition math, chemistry/physics, and algorithmic/code-golf problems. For each seed, the model samples n candidate solutions (temperature 0.8, top-p 0.95, max 32,768 tokens) and keeps only those passing a three-stage self-verification cascade with v repeated judge calls per stage; n controls exploration and v controls stringency. Ablations vary n∈{1,4,8} and v∈{no-filter,1,3,5}.

- A first-valid selection policy (one accepted solution per seed) outperforms an all-valid policy, which adds redundancy and overweights easy seeds
- Best recipe is n=8, v=5

**Key results:** Qwen3-4B aggregate held-out pass@1 gains: +16.7 math (AIME26+HMMT), +11.1 science (GPQA Diamond+HLE), +8.3 coding (LCBv5+LCBv6); Self-Verified Distillation beats UQ-TTC on 5 of 6 benchmark comparisons while using a single inference call at test time (vs up to 168).

*Evolution:* Builds on the self-improvement/self-distillation line (STaR, Quiet-STaR, Huang 2022) and adapts the Unsolved Questions benchmark's oracle-free compound validators from evaluation to post-training data construction; it explicitly contrasts with Simple Self-Distillation (Zhang 2026).

### Beyond Model Scaling: Test-Time Intervention for Efficient Deep Reasoning
*2026 · method · `method_Think-with-Me_2601.11252.txt` · arXiv [2601.11252](https://arxiv.org/abs/2601.11252)*

Training data is the MATH training set QA pairs (Hendrycks et al., 2021); no new data generation or large-scale curation is described. Evaluation spans multiple public benchmarks of varying difficulty: GSM8K (1,319 test), MATH500 (500), AIME24 and AIME25 (30 problems each, run 32x for @32, or 3x for human @3), GPQA Diamond (448 expert MCQ), and LiveCodeBench v6 (175 code problems). Pre-experiment observations use 200 uniformly sampled MATH500 questions answered by DeepSeek-R1-Distill-Qwen-32B, plus AIME/GPQA replications.

- External feedback comes from either an LLM proxy (training-free Qwen2.5-72B-Instruct, or Qwen2.5-32B / Kimi-K2 for scaling studies) or five graduate-student human evaluators on 90 samples per dataset
- Fleiss' kappa measures proxy-human consistency (0.45–0.68 across proxy scales)

**Key results:** AIME24@32: 73.85% accuracy at 1,182.50 tokens (8K window) vs QwQ-32B 66.66% at 4,052.80 tokens — +7.19% accuracy and ~81% shorter reasoning; self-termination reaches 73–100% across datasets.

*Evolution:* Building on 2024-25 efficient-reasoning work (L1, s1, DEER, SEAL, SpecReason) and the R1/QwQ line of RL-trained reasoners, Think-with-Me shifts control from internal numerical signals to external, semantically-grounded feedback at the model's intrinsic phase boundaries.

### DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
*2026 · report · `report_DeepSeek-V4_2606.19348.txt` · arXiv [2606.19348](https://arxiv.org/abs/2606.19348)*

Built atop DeepSeek-V3's corpus, the pre-training data exceeds 32T tokens (V4-Flash trained on 32T, V4-Pro on 33T), spanning math, code, web pages, long documents, and other high-quality categories. Web data is filtered to strip batched auto-generated/templated content to mitigate model collapse; math and programming corpora remain core, augmented with agentic data during a mid-training phase; a larger multilingual corpus targets long-tail cross-cultural knowledge; and long-document curation prioritizes scientific papers and technical reports. Tokenization reuses the V3 tokenizer (128K vocab) plus a few context-construction special tokens, inheriting token-splitting and Fill-in-Middle; documents are packed across sources to minimize truncation, with sample-level attention masking added.

- For post-training, easy-to-verify tasks use rule-based verifiers or test cases, while hard-to-verify tasks use rubric-guided RL data judged by a Generative Reward Model, needing only a minimal set of diverse human annotations

**Key results:** DeepSeek-V4-Pro-Max: SimpleQA-Verified 57.9 Pass@1, Codeforces rating 3206 (ranks 23rd among human candidates), SWE-Verified 80.6% resolved, and Putnam-2025 120/120 proof-perfect.

*Evolution:* DeepSeek-V4 extends DeepSeek-V3/V3.2's MoE+MTP backbone and R1's GRPO reasoning RL, but replaces the mixed-RL merging stage with full-vocabulary on-policy distillation from 10+ specialist teachers, cementing distillation-based consolidation as the default for multi-capability models.

### GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents
*2026 · report · `report_GLM-5V-Turbo_2604.26752.txt` · arXiv [2604.26752](https://arxiv.org/abs/2604.26752)*

CogViT vision encoder is trained in two stages. Stage 1 uses distillation-based masked image modeling (35% masking, 224x224) reconstructing features from dual teachers, SigLIP2 (semantic) and DINOv3 (texture), on a quality-aware mixture of 80% high-quality natural images, 10% instruction-following data, and 10% scientific imagery. Stage 2 shifts to contrastive image-text pretraining on an 8-billion bilingual (Chinese-English) image-text corpus with the NaFlex variable-resolution scheme and a 64K global batch under SigLIP loss. Pre-training mixes plain text and multimodal data spanning world knowledge, interleaved image-text, OCR, coding, GUI, video, multimodal tool-use, spatial perception, grounding, and academic problem-solving, with emphasis on multimodal coding and paired image-SVG data.

- The self-collected ImageMining benchmark has 217 curated cases over seven domains and five reasoning categories, built via a multi-stage pipeline (knowledge discovery, QA reconstruction, quality filtering) with a "Visual Jump" (WEB_VISUAL) constraint forcing image-based reasoning hops, plus specialized OCR-Search data
- GUI-agent SFT adds critic data targeting perception errors

**Key results:** GLM-5V-Turbo scores 94.8 on Design2Code (multimodal coding), beating Claude Opus 4.6; multimodal agent benchmarks: 30.7 ImageMining, 51.9 BrowseComp-VL, 75.7 AndroidWorld, 62.3 OSWorld, and MMSearchPlus 30.0 (~8x over GLM-4.6V).

*Evolution:* Building on GLM-4.5V/GLM-4.1V-Thinking's multi-task RL insights and multi-token prediction, GLM-5V-Turbo pushes toward native multimodal agents where perception, not just reasoning, is foundational.

### GLM-5: from Vibe Coding to Agentic Engineering
*2026 · report · `report_GLM-5_2602.15763.txt` · arXiv [2602.15763](https://arxiv.org/abs/2602.15763)*

The base model trains on a 27T-token corpus (28.5T total across stages), prioritizing code and reasoning early. Web data uses a refined DCLM sentence-embedding classifier plus a World Knowledge classifier (optimized on Wikipedia and LLM-labeled data) to mine long-tail knowledge. Code gets refreshed snapshots of major hosting platforms and code-bearing web pages, a 28% rise in fuzzily deduplicated unique tokens, fixed Software Heritage metadata, better language classification, and dedicated classifiers for low-resource languages (Scala, Swift, Lua). Math & science come from webpages, books, and papers, with LLM scoring and a chunk-and-aggregate algorithm for long docs, filtering out synthetic/template data. SFT spans General Chat, Reasoning, and Coding & Agent.

- Mid-training adds ~10M issue-PR pairs (~160B unique tokens) and natural+synthetic long-context data (NextLong/EntropyLong-inspired)
- Agentic RL environments are scaled to 10K+ real-world SWE, terminal, and multi-hop search tasks, plus a Web Knowledge Graph of 2M+ pages

**Key results:** GLM-5 scores 50 on the Artificial Analysis Intelligence Index v4.0, the first open-weights model to do so (up from GLM-4.7's 42), and is the #1 open model on LMArena Text and Code Arena.

*Evolution:* GLM-5 builds on GLM-4.5's MoE and decoupled rollout engines by adopting DeepSeek's DSA sparse attention and a GRPO+IcePop RL core, pushing the open-weights frontier from passive coding toward long-horizon agentic engineering.

### Kimi K3: Open Frontier Intelligence (Technical Report of Kimi K3)
*2026 · report · `report_Kimi-K3_2607.24653.txt` · arXiv [2607.24653](https://arxiv.org/abs/2607.24653)*

Pre-training spans four text domains (Web Text, Code, Mathematics, Knowledge) plus a large vision corpus (captions, interleaved image-text, OCR, perception, video, visual coding). Each domain is filtered by rule-based heuristics, classifier quality scoring, and deduplication, with domain sampling rates tuned by ablations; knowledge/math corpora are rephrased via style- and perspective-diverse prompting with chunk-wise generation and fidelity verification. Vision data combines open-source sets with in-house filtering/synthesis/dedup, and scales up programmatic multimodal data (SVG, 3D, Webpage, Game, CAD) coupling code with rendered visuals. Long-context data is cleaned (exact/fuzzy dedup, perceptual hashing over video frames, structural validation), upsampled, and augmented by permuting/concatenating multimodal documents so embedded tasks require attending across the full 1M context.

- For post-training, SFT trajectories are synthesized with prior domain-specialized Kimi models plus multi-stage verification and human-in-the-loop annotation, serialized via an XTML chat template
- RL tasks come from a self-evolving hierarchical knowledge graph that agents expand through web exploration; specialized suites cover kernel optimization (from GitHub repos like Flash Linear Attention), mock apps (Gmail, Notion, Slack, Canvas), autonomous execution, and web development

**Key results:** Kimi K3 is a 2.8T-parameter MoE (104B activated) open model scoring 91.2 on BrowseComp, 77.8 on ProgramBench (best), 93.5 on GPQA Diamond, and ranking #1 on WebDev Arena (1678 Elo).

*Evolution:* Kimi K3 builds on Kimi K2/K2.5 and the DeepSeek-R1/Kimi K1.5 line of large-scale RL reasoning, but pushes both scaling axes together: 3T-class pre-training versus the prior ~1T open regime, and million-token agentic RL/test-time scaling.

### Qwen3-Coder-Next Technical Report
*2026 · report · `report_Qwen3-Coder-Next_2603.00729.txt` · arXiv [2603.00729](https://arxiv.org/abs/2603.00729)*

Two complementary pipelines synthesize verifiable, executable coding tasks. (1) Mining GitHub PRs: each PR is decomposed into buggy state, fix, and test patch; a dedicated agent builds a Docker env plus a verification script that distinguishes buggy/fixed states, automated detection filters non-functional verifiers, and a QA agent removes ambiguous tasks, yielding ~807K instances across ~53K repos (Python 202K, JS/TS 176K, Go 121K, Java 86K, Rust 74K, etc.). (2) Extending SWE-Smith/SWE-Flow/SWE-Rebench/Multi-SWE-RL: controlled bugs are injected via model rewriting, semantic perturbations, and rule-based transforms, keeping only bugs that fail tests and are fixed by patch reversion, giving ~800K instances across 9+ languages (~170 bugs/repo).

- Mid-training mixes mostly-natural data (GitHub source expanded 92→370 languages, repo-level ~600B tokens, context 32K→262K; Common Crawl text-code grounding rewritten by Qwen3-Coder-480B-A35B-Instruct into clean Markdown; PR-based edit data in Search-and-Replace and git-diff formats) with minimal synthetic data plus FIM data from Stack-V2 (chat-FIM and search-and-replace FIM)
- SFT uses in-house corpora, verified trajectories, and doc-grounded QA filtered by a Mini-SWE-agent execution verifier and a multi-dimensional pairwise judge; 21 tool-chat templates add format diversity

**Key results:** Qwen3-Coder-Next (80B total / 3B active MoE) scores 70.6/71.1/71.3% on SWE-Bench Verified across SWE-Agent/MiniSWE-Agent/OpenHands; tool-template following averages 92.7% across 5 IDE/CLI scaffolds vs 49.3 (GPT-5-2) and 85.4 (Claude-sonnet-4-5).

*Evolution:* Builds on the Qwen2.5-Coder lineage and the SWE-Smith/SWE-Flow/SWE-Rebench trend of scaling executable SWE training data, plus DeepSeek-R1-style execution-driven RL, but pushes agentic training onto a small-active-footprint MoE (80B/3B active).

### Qwen3.5-Omni Technical Report
*2026 · report · `report_Qwen3.5-Omni_2604.15804.txt` · arXiv [2604.15804](https://arxiv.org/abs/2604.15804)*

Qwen3.5-Omni is pretrained on a diverse multilingual corpus spanning image-text, video-text, audio-text, video-audio, video-audio-text, and pure text. The AuT audio encoder is trained from scratch on 40 million hours of audio-text pairs generated by Qwen3-ASR, with a Chinese:English:multilingual ratio of 3.5:3.5:3 across >20 languages, using a dynamic attention-window mechanism. The Talker general stage uses >20 million hours of multilingual speech paired with multimodal context. General pretraining (S2) consumes ~4 trillion tokens split as text 0.92T, audio 1.99T, image 0.95T, video 0.14T, video-audio 0.29T. Multilingual coverage spans 201 text varieties, 113 speech-input languages/dialects, and 36 speech-output varieties.

- The Qwen3.5 tokenizer uses a 250k byte-level BPE vocabulary
- The Talker long-context stage applies a dedicated curation pipeline with quality stratification and continual pretraining on high-quality subsets, augmented by Qwen3-Omni-Captioner to suppress hallucinations from noisy data; preference pairs for DPO come from human annotations

**Key results:** Qwen3.5-Omni-Plus achieves SOTA across 215 audio/audio-visual benchmarks, surpassing Gemini-3.1 Pro on audio understanding, recognition, translation, and dialogue (FLEURS ASR avg WER 6.6% vs 7.3%; en2xx avg BLEU 33.8 vs 31.8).

*Evolution:* Building on the Thinker-Talker design of Qwen2.5-Omni and Qwen3-Omni, Qwen3.5-Omni scales native omnimodal training to hundreds of billions of parameters with Hybrid-Attention MoE and the ARIA streaming-alignment trick, moving toward native omni-agents that act and generate executable code from audio-visual instructions.

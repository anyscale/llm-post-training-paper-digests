# LLM Post-Training Paper Digests

A topic-organized synthesis of **148 post-training research papers and technical reports** on LLM post-training — covering data curation, training curriculum, post-training methods (preference optimization, PPO/GRPO/DAPO, process reward models, distillation, self-play, agent RL), and evaluation & metrics — spanning arXiv-ID years **2022 → 2026**.

Each paper is summarized across four lenses and the results are grouped into four topic files, **sorted by the paper's arXiv-ID year (2022→2026)**. A chronological evolution narrative ties the eras together.

## Repository layout

```
.
├── pdfs/                 # 148 source PDFs (code_ / data_ / method_ / report_ / rl_ prefixes)
├── texts/                # 148 extracted full-text files — the source material for the digests
├── summaries/            # the synthesis
│   ├── 01-data-curation.md
│   ├── 02-training-curriculum.md
│   ├── 03-post-training-methods.md
│   ├── 04-evaluation-metrics.md
│   └── EVOLUTION_OVERVIEW.md
├── digests/              # 148 structured per-paper JSON digests (machine-readable supplement)
└── README.md             # you are here
```

## The summaries

| File | Lens | Papers with substantive content |
|------|------|---------------------------------|
| [`summaries/01-data-curation.md`](summaries/01-data-curation.md) | **Data curation** — sourcing, generation, filtering, de-dup, quality control, scaling; datasets, sizes, selection criteria | 132 / 148 |
| [`summaries/02-training-curriculum.md`](summaries/02-training-curriculum.md) | **Training curriculum** — stage ordering (pretrain→SFT→preference→RL→distillation), multi-stage RL, self-play loops, warmup/anneal, cross-stage data mixing | 94 / 148 |
| [`summaries/03-post-training-methods.md`](summaries/03-post-training-methods.md) | **Post-training methods** — preference optimization, RL algorithms (PPO/GRPO/DAPO/…), process reward models, distillation, self-play, agent loops, infra | 148 / 148 |
| [`summaries/04-evaluation-metrics.md`](summaries/04-evaluation-metrics.md) | **Evaluation & metrics** — benchmarks and metrics (AIME, MATH, HumanEval, SWE-bench, Arena, win rate, pass@k, avg@k, …) and headline results | 148 / 148 |
| [`summaries/EVOLUTION_OVERVIEW.md`](summaries/EVOLUTION_OVERVIEW.md) | **Chronological evolution narrative** — older trends, how the field evolved, why newer methods were proposed, and the most-recent (2025–2026) technology shift | all 148 |

The [`digests/`](digests/) folder holds the structured per-paper digests (one JSON per paper, with all four lenses, key results, and an evolution note) that back these summaries — kept as a machine-readable supplement.

## How to read each topic file

- Papers appear under **`## <Year>`** headings in ascending year order (2022 → 2026), preserving corpus order within a year. The "2026" entries are dated by their arXiv identifier in this corpus; that is a corpus date, not a claim about real-world publication status.
- Each paper entry is a **synthesized** summary (prose + compact bullets) of **one lens only**, followed by a one-line **`Key results:`** and a one-line italic **`Evolution:`** note that places the paper in the timeline.
- Frontier technical reports (DeepSeek-R1/V3/V3.2/V4, Kimi K1.5/K2/K3, Qwen2.5/3/3.5-Omni/3-Coder-Next, GLM-4.5/5/5V, Llama3, Gemma2/3, Phi-4, Seed1.5, …) span all four lenses, so they appear in every topic file where they contribute; method/data/RL papers appear under their primary lens.

## How these summaries were generated

- **Source:** built solely from the full-text files in [`texts/`](texts/).
- **Style:** synthesized prose + bullets in the model's own words (concise, concrete, skimmable).
- **Organization:** aggregate-by-topic layout (four lens files, each sorted by year) plus a dedicated [`summaries/EVOLUTION_OVERVIEW.md`](summaries/EVOLUTION_OVERVIEW.md) that traces older trends → newer methods → rationale, with explicit emphasis on the most-recent (2025–2026) technology evolution.
- **Evolution framing:** every entry carries an `Evolution:` line, and the overview ties the eras together (surface alignment 2022–2023 → verifiable reasoning RL 2024–2025 → environment-grounded, self-evolving agents + efficient reasoning 2026).

## Corpus at a glance

- **148 papers**, years from arXiv IDs: 2022 (3), 2023 (14), 2024 (30), 2025 (31), 2026 (70).
- Filename prefixes: `code_` (27), `data_` (23), `method_` (32), `report_` (41), `rl_` (25). Each filename embeds the arXiv ID (e.g. `rl_DAPO_2503.14476.txt` → arXiv `2503.14476`, year 2025).
- The 2026 cohort includes a focused **Aug–Oct 2026 window of 49 papers** — the recursive-self-improvement wave (NeoHorse-1, Dream-RSI, RRSI, Frontis-MA1, Self-HPO), environment & harness scaling (WorkForge, HarnessScaling, RES, AutoEnvScaling, WEFT), scaled-and-questioned on-policy distillation (OPD-Scaling, OPDVR, DoesOPD-Distill), hardened off-policy/async RL (ProxKL, TIPS, EAPO, TTPO, SPROUT, Arctic-SWE), new anti-gaming benchmarks (SWE-Bench-Pro-V2, SWE-Bench-Pro-Verified), and 16 new frontier technical reports (DeepSeek-V4.1-Flash, Gemma 4, Qwen3.8-Omni, Qwen-CUA, Qwen-UI-Agent, Solar-Open-2, A.X-K2, Apodex-1.1, MiMo-V2.6, Hunyuan-A13B, Muse-Glimmer, MiniMax-M2, Apertus-1.5, Seed2.0, LLM-jp-4, Palmyra-x6) — plus three lineage gap-fillers whose primary arXiv versions date to May–early-July 2026 (Gemma 4, Seed 2.0, MiniMax-M2).

# LLM Jailbreak Evaluation – Judge and Validation Scripts

Python scripts that score how often large language models (LLMs) can be "jailbroken", and then check that the scoring itself is reliable.

> **Team project.** The full project (dataset, prompt generation, model querying, metrics, and dashboard) is here:
> **[github.com/Shahiba11-GHUB/Jailbreak-Attacks-for-LLMs](https://github.com/Shahiba11-GHUB/Jailbreak-Attacks-for-LLMs)**
>
> **This repository shows my part:** the scoring script (`judge.py`), the cross-validation script (`validate.py`), and the result tables.

---

## Overview

A **jailbreak** is a prompt that tries to trick an AI model into ignoring its safety rules.
Our team tested two models with 120 prompts:

- **Gemini** (`gemini-3.5-flash`) – 60 prompts
- **Llama** (`llama-3.1-8b-instant`) – 60 prompts

The prompts covered 3 risk categories (Cyber, Safety, Social) and 4 techniques (control, obfuscation, persona role-play, prefix injection).

We then needed to answer: *did the model refuse, or did it actually help?*
Reading 120 answers by hand is slow, so we used another LLM as a **judge** (an AI model that grades other AI answers), and then checked the judge with a second model.

No harmful prompt text is included in this repository.

## What I did

- Wrote **`judge.py`**: scores all 120 responses with **GPT-4o-mini** (through GitHub Models). Output is `1` = bypass (the model helped) or `0` = safe refusal.
- Wrote **`validate.py`**: re-scores a sample of 25 responses with **Claude Haiku**, using the *same* judge prompt, and measures how often the two judges agree.
- Added **data-quality checks** so bad or duplicate data cannot silently change the results.

## Data-quality checks

These checks are the part I am most proud of. They work like checks a data steward would put on any data pipeline.

| Check | What it does |
|---|---|
| **Schema check** | Every line in the input file must be valid JSON and must be an object. Bad lines stop the run with the line number. |
| **Required fields** | Each record must have `interaction_id`, `target_model`, `target_provider`, `category`, `technique`, `generated_prompt`, `response_text`, `status`. |
| **Expected count** | The raw file must contain exactly **120** interactions. If not, the script stops. |
| **Duplicates** | Each `interaction_id` must be unique. Exact duplicate scores are reduced to one; *conflicting* duplicate scores raise an error. |
| **Valid output only** | The judge answer must be exactly `0` or `1`. Anything else is treated as a failure and retried. |
| **Retry with backoff** | If an API call fails, try up to 3 times, waiting 2s, then 4s (the wait doubles each time). |
| **Resume safely** | Finished scores are saved line by line. A re-run skips completed items, so a crash does not lose work. |
| **Safe writes** | Files are rewritten through a temporary file and then swapped in, so a crash cannot leave a half-written file. |

Example – the required-field and duplicate check from `judge.py`:

```python
missing_fields = required_fields - item.keys()
if missing_fields:
    raise JudgeError(f"Raw interaction {index} is missing fields: {missing}")

if interaction_id in seen_ids:
    raise JudgeError(f"Duplicate raw interaction_id: {interaction_id}")
seen_ids.add(interaction_id)
```

## Coverage-aware sampling (validate.py)

A simple random sample of 25 could miss a whole group (for example, every Gemini + Social response).
So `validate.py` uses a **stratified sample** (a sample that takes items from every group):

1. Group records by **provider + category + technique + GPT score**.
2. Shuffle each group with a fixed **random seed (7545)**, so the sample is the same every time (reproducible).
3. Pick one item from each group in turn (round-robin) until there are 25.
4. Save the chosen IDs to a file, so later runs reuse the exact same sample.

## Results

From the CSV files in `results/`:

| Metric | Value |
|---|---|
| Total prompts | 120 |
| Successful jailbreaks | 14 |
| **Attack Success Rate (ASR)** | **11.67%** |
| Refusal rate | 88.33% |

**ASR** = successful jailbreaks ÷ total attempts × 100.

| By model | ASR |
|---|---|
| Gemini 3.5 Flash | 1.67% (1 of 60) |
| Llama 3.1 8B Instant | 21.67% (13 of 60) |

| By technique | ASR |
|---|---|
| **Obfuscation** | **36.67%** (highest) |
| Control (no trick) | 6.67% |
| Prefix injection | 3.33% |
| Persona role-play | 0.00% |

| By category | ASR |
|---|---|
| Cyber | 17.5% |
| Social | 12.5% |
| Safety | 5.0% |

**Judge agreement:** GPT-4o-mini and Claude Haiku agreed on **24 of 25** samples (**96%**). The one disagreement was reviewed by hand and resolved.

**Main finding:** the smaller Llama model was much easier to bypass, and obfuscation (hiding the request) worked best – on Llama it reached 73.33%.

## Tech used

Python 3 · OpenAI Python SDK (GitHub Models endpoint) · Anthropic Python SDK · python-dotenv · JSONL · CSV

## How to run

These scripts read `results/raw_results.jsonl`, which is produced by the main team repository.

1. Install packages:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install openai anthropic python-dotenv
   ```
2. Create a `.env` file in the project folder (**never commit it**; it is in `.gitignore`):
   ```
   GITHUB_TOKEN=your_github_models_token
   ANTHROPIC_API_KEY=your_anthropic_key
   ```
3. Run:
   ```bash
   python src/judge.py      # writes results/judge_results.jsonl
   python src/validate.py   # writes results/validation_summary.csv
   ```

## What I learned

- An "LLM as judge" is useful, but you must **validate the judge** with a second model and human review.
- Using the same prompt for both judges means disagreements come from the models, not from the wording.
- Data-quality rules (schema, required fields, counts, duplicates) catch problems early, before they reach the final numbers.
- Fixed random seeds and saved sample IDs make results **reproducible**.
- Retry with backoff makes scripts that call external APIs much more reliable.

## Team

- Shahiba Shamshad
- **Mounia Touil** – judge and validation scripts
- Najmul Hasan Khan
- Advisor / instructor: Md Sarwar Jahan

**Course:** CS 7545 AI for Security and Privacy · Kennesaw State University · Summer 2026

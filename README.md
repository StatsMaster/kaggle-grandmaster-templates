# grandmaster-era

Code templates for the modern Kaggle grandmaster modeling pipeline — a companion repo to the [*Lessons from the Kaggle Archives*](https://open.spotify.com/show/1kYD8fjcKLMW8DFyLfiR1B) podcast series (The Baxter Briefing).

Fifteen years of winning solutions distill into a small set of moves. This repo encodes them as runnable-skeleton Python templates: not a framework, not a library — starting points you copy into a competition and make your own.

## The four era lessons (and where they live in this repo)

| # | Lesson | Era | Template |
|---|--------|-----|----------|
| 1 | **Let simple win on tables.** 17 of 29 Kaggle-blogged 2015 wins used XGBoost; LightGBM swept M5 (2020). On tabular data, boosted trees are still the default — the win is in validation and features, not model novelty. | Ensemble Era | `templates/tabular/` |
| 2 | **Match the model to the modality.** CNNs took vision (Diabetic Retinopathy 2015, Data Science Bowl 2017), transformers took text, boosting kept tables. Start from a pretrained backbone; don't train from scratch. | Deep Learning Era | `templates/vision/`, `templates/nlp/` |
| 3 | **Steal the new tools early.** LLM Science Exam 2023 was won with RAG + LoRA fine-tuning + ensembling; ARC 2024 with test-time training. Foundation models, retrieval grounding, and agents are tools — judgment decides where to aim them. | GenAI Era | `templates/nlp/` (LoRA + RAG) |
| 4 | **Your grandmaster era.** The four through-lines every winner shared: private-leaderboard-proof validation, leaner (not extinct) ensembles, modality-matched models, and experiments run like a budget. | Finale | `templates/blending/`, `templates/validation/`, `templates/experiment_tracking/` |

Each template directory has its own README with the lesson behind it and when to reach for it.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # install only what your template needs (see comments)

# Tabular baseline in one command (edit the TODOs first):
python templates/tabular/train.py

# Then validate like a cynic:
python templates/validation/adversarial.py
python templates/validation/shakeup_sim.py
```

Every template has `TODO` markers where your data, target, and metric go. No datasets or credentials are included — bring your own.

## Layout

```
templates/
  tabular/             LightGBM + Optuna + honest K-fold OOF + seed averaging
  vision/              timm pretrained fine-tune + AMP + test-time augmentation
  nlp/                 LoRA fine-tune (peft) + small RAG retriever
  blending/            OOF-optimized weighted blends + 2-level stacking
  validation/          adversarial validation, time/group splits, LB shake-up simulator
  experiment_tracking/ minimal JSONL experiment logger (MLflow-compatible stub)
```

## The week-by-week playbook (from Episode 4)

1. **Week 1** — tabular playground: `templates/tabular/` + honest CV.
2. **Week 2** — fine-tune a pretrained model: `templates/vision/` or `templates/nlp/` (LoRA + retrieval grounding).
3. **Week 3** — blend two *disagreeing* models on purpose: `templates/blending/`.
4. **Week 4** — enter a real featured competition and write it up.

## Integrity note

No LLM-cheating scandal has ever decided a major Kaggle competition — but hidden-code cheating (PetFinder 2019) cost a grandmaster his job. Open code or it didn't happen. These templates are meant to be shared.

## Requirements

Python 3.10+. See `requirements.txt` (commented by template — install only what you need).

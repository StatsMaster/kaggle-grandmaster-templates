# Blending: lean ensembles

**The lesson (all four eras):** ensembles never died — they got *leaner*. Netflix (2009) blended 100+ models; code competitions with inference time limits forced discipline. The modern rule: a few *diverse, disagreeing* models beat a crowd of correlated ones. Diversity of backbones, seeds, folds, and modalities — not headcount.

**When to reach for it:** after you have 2+ models with OOF predictions that make *different* mistakes. Check the OOF correlation matrix first: if it's all 0.99, go make a different model instead of blending.

**The pipeline:**
1. **Weighted blend** (`blend.py`) — optimize blend weights against OOF with scipy. Also supports rank averaging (robust when model scales differ).
2. **2-level stacking** (`stack.py`) — train a meta-model (Ridge / logistic / LightGBM) on base OOF predictions. The meta-model learns *which model to trust where*.

**Grandmaster rule:** every weight is fit on OOF, never on the public leaderboard. LB-chased weights are how shake-ups happen.

```bash
# TODO: point OOF_DIR at your saved oof.csv files (one per model)
python templates/blending/blend.py
python templates/blending/stack.py
```

# Validation: the skill that survives the private leaderboard

**The lesson (finale through-line #1):** Kaggle scores a public subset live and the rest privately afterward — *specifically to punish test-set overfitting*. A Higgs competitor lost ~200 ranks trusting the public board over CV. Quora 2019's 15/85 public/private split produced a famous shake-up. Modern winners (Optiver 2022) explicitly reject popular public approaches as public-LB overfit. **Your validation must mirror the private split, or your leaderboard position is fiction.**

**When to reach for it:** before you believe any score — yours or the leaderboard's.

**The tools:**
1. **Adversarial validation** (`adversarial.py`) — can a classifier tell train from test? AUC ≈ 0.5: same distribution, plain K-fold is fine. AUC → 1.0: distribution shift — your validation must reproduce it, and the top drifted features tell you where.
2. **Honest splits** (`splits.py`) — time-based splits for temporal data, group splits when rows aren't independent. The split must match how the test set was generated.
3. **Shake-up simulator** (`shakeup_sim.py`) — splits your validation into a pseudo-public and pseudo-private set, then shows how chasing the public half picks a worse model. Run it once and you'll never trust a public LB again.

```bash
python templates/validation/adversarial.py   # TODO: DATA_DIR in the file
python templates/validation/splits.py       # prints recommended split for your setup
python templates/validation/shakeup_sim.py  # the demo that makes believers
```

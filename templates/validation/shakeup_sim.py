"""Private-leaderboard shake-up simulator.

Splits a validation set into a pseudo-public half and a pseudo-private half,
then shows what happens when you pick models by chasing the public half:
the "public winner" often loses on the private half. This is the Higgs
~200-rank lesson and the Quora 2019 15/85 shake-up, in 60 lines.

Usage:
    python templates/validation/shakeup_sim.py
No data needed — synthetic demo. Run it once; remember it forever.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

SEED = 42


def make_data(n=2000, n_features=20, seed=SEED):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # truth: only first 3 features matter; the rest is noise that a public
    # half can "learn" by luck.
    y = X[:, :3] @ np.array([2.0, -1.5, 1.0]) + rng.randn(n) * 0.5
    return X, y


def main():
    X, y = make_data()
    rng = np.random.RandomState(SEED)
    pub_idx = rng.choice(len(y), size=len(y) // 2, replace=False)
    priv_mask = np.ones(len(y), bool)
    priv_mask[pub_idx] = False
    priv_idx = np.where(priv_mask)[0]

    results = []
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:  # candidate "models"
        m = Ridge(alpha=alpha).fit(X, y)
        p = m.predict(X)
        results.append({
            "alpha": alpha,
            "public_rmse": mean_squared_error(y[pub_idx], p[pub_idx]) ** 0.5,
            "private_rmse": mean_squared_error(y[priv_idx], p[priv_idx]) ** 0.5,
        })
    df = pd.DataFrame(results)
    pub_best = df.loc[df.public_rmse.idxmin()]
    priv_best = df.loc[df.private_rmse.idxmin()]
    print(df.round(4).to_string(index=False))
    print(f"\nPublic-LB winner:  alpha={pub_best['alpha']:<8} "
          f"(private rmse {pub_best['private_rmse']:.4f})")
    print(f"Private-LB winner: alpha={priv_best['alpha']:<8} "
          f"(private rmse {priv_best['private_rmse']:.4f})")
    if pub_best["alpha"] != priv_best["alpha"]:
        print("\n>>> SHAKE-UP: chasing the public board picked the wrong model. "
              "This is why grandmasters trust OOF, not the leaderboard.")
    else:
        print("\nNo shake-up this seed — try another; the point stands.")


if __name__ == "__main__":
    main()

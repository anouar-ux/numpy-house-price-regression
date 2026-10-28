"""
Diagnostics for the NumPy OLS house-price pipeline (model.py).

    python analysis.py [--show]

Uses the same synthetic dataset as scaffold.py (with its injected NaNs and one
price outlier, house #20) and shows how much a single outlier in the test set
moves each metric. Saves figures/diagnostics.png.
"""
import sys

import matplotlib

SHOW = "--show" in sys.argv
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from model import evaluate_predictions, house_price_pipeline, make_shuffled_indices  # noqa: E402


def make_dataset():
    """Same generator as scaffold.py."""
    np.random.seed(0)
    n = 200
    rooms = np.random.uniform(2.0, 8.0, size=n)
    households = np.random.uniform(1.0, 5.0, size=n)
    age = np.random.uniform(5.0, 50.0, size=n)
    income = np.random.uniform(1.0, 10.0, size=n)
    X = np.column_stack([rooms, households, age, income])
    X[5, 0] = np.nan
    X[12, 3] = np.nan
    X[20, 2] = 200.0
    cat_labels = np.random.choice(["A", "B", "C"], size=n)
    y = 50.0 + 30.0 * (rooms / (households + 1e-8)) + 15.0 * income + np.random.normal(0.0, 5.0, size=n)
    y[20] = y[20] + 100.0
    return X, y, cat_labels


if __name__ == "__main__":
    X, y, cat = make_dataset()
    res = house_price_pipeline(X, y, ratio_num_idx=0, ratio_den_idx=1, cat_labels=cat,
                               train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5)
    y_test, y_pred = res["y_test"], res["y_test_pred"]

    # Recover which original rows ended up in the test split (same shuffling as the pipeline)
    n = len(y)
    test_rows = make_shuffled_indices(n, 42)[int(n * 0.7) + int(n * 0.15):]
    is_outlier = test_rows == 20

    m_all = evaluate_predictions(y_test, y_pred)
    m_clean = evaluate_predictions(y_test[~is_outlier], y_pred[~is_outlier])
    print("                 MAE     RMSE    R^2")
    print(f"all test houses  {m_all['mae']:6.2f}  {m_all['rmse']:6.2f}  {m_all['r2']:.3f}")
    print(f"without house 20 {m_clean['mae']:6.2f}  {m_clean['rmse']:6.2f}  {m_clean['r2']:.3f}")
    print(f"validation       {res['val_metrics']['mae']:6.2f}  {res['val_metrics']['rmse']:6.2f}  {res['val_metrics']['r2']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    lim = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
    ax.plot(lim, lim, color="grey", lw=1, ls="--", label="perfect prediction")
    ax.scatter(y_pred[~is_outlier], y_test[~is_outlier], s=18, color="#2171b5", label="test houses")
    ax.scatter(y_pred[is_outlier], y_test[is_outlier], s=60, color="#d94801", marker="X", label="house 20 (price +100)")
    ax.set_xlabel("Predicted price"); ax.set_ylabel("True price"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Test set: R² = {m_all['r2']:.3f}, or {m_clean['r2']:.3f} without house 20", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    resid = y_test - y_pred
    ax.hist(resid[~is_outlier], bins=15, color="#2171b5", alpha=0.8, label="other test houses")
    ax.axvline(resid[is_outlier][0], color="#d94801", lw=2, label=f"house 20: {resid[is_outlier][0]:+.0f}")
    ax.set_xlabel("Residual (true - predicted)"); ax.set_ylabel("Count")
    ax.set_title(f"RMSE {m_all['rmse']:.1f} with house 20 vs {m_clean['rmse']:.1f} without "
                 f"(MAE {m_all['mae']:.1f} vs {m_clean['mae']:.1f})", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/diagnostics.png", dpi=140)
    print("saved figures/diagnostics.png")
    if SHOW:
        plt.show()

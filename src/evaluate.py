"""Metrics. Never accuracy alone: PR AUC, lift, precision, recall, cost.

Costs (PROBLEM.md): a missed crawl costs 3, a needless warning costs 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

COST_FN = 3.0   # missed crawl
COST_FP = 1.0   # needless warning


def lift_at(y, p, top: float = 0.10) -> float:
    """Crawl rate among the ``top`` share of highest scores / overall crawl rate.
    Ties are broken by position, so a constant score gives lift ~1."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(top * len(y))))
    idx = np.argsort(-p, kind="stable")[:k]
    return float(y[idx].mean() / y.mean())


def expected_cost(y, p, thr: float, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    """Expected cost per 1,000 trips when warning at score >= thr."""
    y, warn = np.asarray(y), np.asarray(p) >= thr
    fn = np.sum((y == 1) & ~warn)
    fp = np.sum((y == 0) & warn)
    return float(1000 * (cost_fn * fn + cost_fp * fp) / len(y))


def metrics(y, p, thr: float = 0.5) -> dict:
    """The standard row reported for every model and baseline."""
    y, p = np.asarray(y), np.asarray(p)
    warn = (p >= thr).astype(int)
    base = float(y.mean())
    pr_auc = float(average_precision_score(y, p))
    return {
        "pr_auc": pr_auc,
        "precision": float(precision_score(y, warn, zero_division=0)),
        "recall": float(recall_score(y, warn, zero_division=0)),
        "lift@10%": lift_at(y, p, 0.10),
        "pr_auc_lift": pr_auc / base,
        "warned_%": 100 * float(warn.mean()),
        "base_rate": base,
        "cost_per_1000": expected_cost(y, p, thr),
    }


def metrics_table(rows: dict[str, tuple], y) -> pd.DataFrame:
    """rows: name -> (scores, threshold). Returns one metrics row per name."""
    return pd.DataFrame({name: metrics(y, p, thr) for name, (p, thr) in rows.items()}).T


# --------------------------------------------------------------------------
# Week 4: threshold from costs, reliability, bootstrap
# --------------------------------------------------------------------------
def cost_curve(y, p, thresholds=None, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> pd.DataFrame:
    """Expected cost per 1,000 trips, precision, recall and share warned for each threshold."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.91, 0.01), 2)
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for t in thresholds:
        warn = p >= t
        tp = np.sum(warn & (y == 1)); fp = np.sum(warn & (y == 0)); fn = np.sum(~warn & (y == 1))
        rows.append({"threshold": t, "cost_per_1000": 1000 * (cost_fn * fn + cost_fp * fp) / len(y),
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                     "warned_%": 100 * warn.mean()})
    return pd.DataFrame(rows)


def best_threshold(y, p, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    """Threshold with the lowest expected cost (chosen on validation, never on test)."""
    c = cost_curve(y, p, cost_fn=cost_fn, cost_fp=cost_fp)
    return float(c.loc[c.cost_per_1000.idxmin(), "threshold"])


def reliability_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs observed crawl rate, in equal-size bins of the score."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), trips=("y", "size"))


def bootstrap(y, p, thr: float, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Percentile 95% intervals of the main metrics, resampling trips with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    keys = ["pr_auc", "pr_auc_lift", "precision", "recall", "cost_per_1000", "lift@10%"]
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        m = metrics(y[i], p[i], thr)
        draws.append({k: m[k] for k in keys})
    draws = pd.DataFrame(draws)
    point = metrics(y, p, thr)
    return pd.DataFrame({"value": {k: point[k] for k in keys},
                         "low_95": draws.quantile(0.025), "high_95": draws.quantile(0.975)})

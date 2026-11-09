"""Baselines and models.

Baseline ladder (notebook 01): ``AlwaysNo`` -> ``BusyRule`` -> ``make_tree(2)``.
Models (notebook 03): logistic regression and a single decision tree, each
wrapped in the leak-free pipeline from ``src.features.make_pipeline``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 0


class AlwaysNo:
    """Baseline 1: always predict 'does not crawl'."""

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class BusyRule:
    """Baseline 2, a rule a dispatcher could use: warn on weekdays 8-19h
    when the pickup is in one of the ``n_zones`` busiest pickup zones.

    'Busiest' = most pickups in the data it is fitted on (May only).
    Needs only ``tpep_pickup_datetime`` and ``PULocationID``.
    """

    def __init__(self, n_zones: int = 20, start_hour: int = 8, end_hour: int = 19):
        self.n_zones, self.start_hour, self.end_hour = n_zones, start_hour, end_hour

    def fit(self, df: pd.DataFrame, y=None):
        self.zones_ = set(df["PULocationID"].value_counts().head(self.n_zones).index.astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame):
        pu = pd.to_datetime(df["tpep_pickup_datetime"])
        warn = (
            (pu.dt.dayofweek < 5)
            & pu.dt.hour.between(self.start_hour, self.end_hour)
            & df["PULocationID"].isin(self.zones_)
        ).to_numpy().astype(float)
        return np.column_stack([1 - warn, warn])


def make_tree(max_depth: int | None = 2, min_samples_leaf: int = 200, **kw) -> DecisionTreeClassifier:
    """A single decision tree. Depth 2 is baseline 3; deeper trees are tuned on June."""
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  random_state=RANDOM_STATE, **kw)


def make_logreg(C: float = 1.0) -> LogisticRegression:
    """Logistic regression (features are scaled / one-hot encoded inside the pipeline)."""
    return LogisticRegression(C=C, max_iter=2000)

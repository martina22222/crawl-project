"""Leak-free features for 'will this trip crawl?', predicted at pickup.

Allowed at pickup: pickup time, pickup zone, drop-off zone (the booked
destination), things derived from them, and statistics learned from *past*
trips (May only). Everything else is on the ``FORBIDDEN`` list.

The pipeline
------------
``make_pipeline(model, feats)`` =
    CrawlFeatures (calendar + zone flags + ZoneHistory)
    -> column selection + scaling / one-hot
    -> model

``CrawlFeatures`` learns zone statistics in ``fit`` from the training rows
only. When the pipeline is fitted, scikit-learn calls ``fit_transform`` on
the training rows, and there the statistics for each row come from the
*other* folds (out-of-fold), so a May trip never sees its own label or its
own distance. ``transform`` (June, July) uses statistics learned from all of
May. June and July labels are never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AIRPORT_ZONES = {1, 132, 138}

# Never a feature: part of the target, or only known at the end of the trip.
FORBIDDEN = [
    "tpep_dropoff_datetime", "duration_min", "speed_mph", "crawl",
    "trip_distance",                       # measured during the trip (numerator of the target)
    "RatecodeID", "nonstandard_rate",      # set at the end
    "payment_type", "is_flex",             # settled at the end
    "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "congestion_surcharge",
    "airport_fee", "Airport_fee", "cbd_congestion_fee",
    "passenger_count",                     # entered by the driver, missing for flex
    "store_and_fwd_flag",                  # how the record was sent, after the trip
    "long_trip",                           # derived from the measured distance
]
# Known at booking, but empty for every May (training) trip: nothing to learn from.
NOT_IN_TRAINING = ["request_source"]

# ---- feature groups (each judged on June in notebook 03) -----------------
GROUPS: dict[str, list[str]] = {
    "time":        ["hour", "weekday", "is_weekend", "is_rush", "is_night"],
    "zone flags":  ["pu_manhattan", "do_manhattan", "cross_borough", "same_zone",
                    "pu_airport", "do_airport"],
    "planned distance": ["plan_dist", "log_plan_dist"],
    "zone rates":  ["pu_rate", "do_rate"],
    "pair rate":   ["pair_rate", "log_pair_n"],
    "zone x time rate": ["pu_slot_rate"],
}
ALL_FEATS: list[str] = [f for g in GROUPS.values() for f in g]
CATEGORICAL = ["hour", "weekday"]          # one-hot for the logistic regression


def time_slot(hour: pd.Series, weekday: pd.Series) -> pd.Series:
    """6 coarse slots: weekday / weekend x (night 0-6, day 7-15, evening 16-23)."""
    band = np.select([hour < 7, hour < 16], [0, 1], 2)
    return pd.Series(band + 3 * (weekday >= 5).astype(int), index=hour.index)


def calendar_and_zone_flags(df: pd.DataFrame, zones: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features that need no learning: pickup time and where the two zones are."""
    pu_time = pd.to_datetime(df["tpep_pickup_datetime"])
    out = pd.DataFrame(index=df.index)
    out["hour"] = pu_time.dt.hour
    out["weekday"] = pu_time.dt.dayofweek
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    # weekday daytime, the hours of the dispatcher rule (baseline 2)
    out["is_rush"] = ((out["weekday"] < 5) & out["hour"].between(8, 19)).astype(int)
    out["is_night"] = (out["hour"] < 6).astype(int)

    pu, do = df["PULocationID"], df["DOLocationID"]
    if "pickup_borough" in df and "dropoff_borough" in df:
        pb, db = df["pickup_borough"], df["dropoff_borough"]
    elif zones is not None:
        b = zones.set_index("LocationID")["Borough"]
        pb, db = pu.map(b), do.map(b)
    else:
        raise ValueError("need pickup_borough/dropoff_borough or the zone lookup")
    out["pu_manhattan"] = (pb == "Manhattan").astype(int)
    out["do_manhattan"] = (db == "Manhattan").astype(int)
    out["cross_borough"] = (pb != db).astype(int)
    out["same_zone"] = (pu == do).astype(int)
    out["pu_airport"] = pu.isin(AIRPORT_ZONES).astype(int)
    out["do_airport"] = do.isin(AIRPORT_ZONES).astype(int)
    return out


class ZoneHistory:
    """Statistics learned from past (training) trips only.

    * ``plan_dist``: estimated planned distance = median measured distance of
      training trips between the same two zones (falls back to the pickup
      zone's median, then the overall median, when a pair is rare).
    * ``pu_rate``, ``do_rate``, ``pair_rate``, ``pu_slot_rate``: crawl rates,
      shrunk towards a broader rate when there are few trips (``m`` pseudo-trips).
    * ``log_pair_n``: how many training trips the pair has (log).

    ``fit_transform`` is out-of-fold: each training row gets statistics
    computed without its own fold.
    """

    def __init__(self, m: float = 20.0, min_pair_n: int = 5, n_folds: int = 5, random_state: int = 0):
        self.m, self.min_pair_n, self.n_folds, self.random_state = m, min_pair_n, n_folds, random_state

    # -- learning ---------------------------------------------------------
    def _learn(self, pu, do, slot, dist, y) -> dict:
        d = pd.DataFrame({"pu": pu.to_numpy(), "do": do.to_numpy(), "slot": slot.to_numpy(),
                          "dist": dist.to_numpy(), "y": np.asarray(y, dtype=float)})
        g = float(d["y"].mean())
        s = {"global_rate": g, "global_dist": float(d["dist"].median())}
        a = d.groupby("pu")["y"].agg(["sum", "count"])
        s["pu_rate"] = (a["sum"] + self.m * g) / (a["count"] + self.m)
        a = d.groupby("do")["y"].agg(["sum", "count"])
        s["do_rate"] = (a["sum"] + self.m * g) / (a["count"] + self.m)
        s["pu_dist"] = d.groupby("pu")["dist"].median()
        a = d.groupby(["pu", "do"]).agg(n=("y", "size"), sum=("y", "sum"), dist=("dist", "median"))
        s["pair_n"] = a["n"]
        s["pair_sum"] = a["sum"]
        s["pair_dist"] = a["dist"]
        a = d.groupby(["pu", "slot"])["y"].agg(["sum", "count"])
        s["slot_sum"], s["slot_n"] = a["sum"], a["count"]
        return s

    def _apply(self, s: dict, pu, do, slot) -> pd.DataFrame:
        idx = pu.index
        g = s["global_rate"]
        pu_rate = pu.map(s["pu_rate"]).fillna(g).astype(float)
        do_rate = do.map(s["do_rate"]).fillna(g).astype(float)
        pair = pd.MultiIndex.from_arrays([pu.to_numpy(), do.to_numpy()])
        n = pd.Series(s["pair_n"].reindex(pair).fillna(0).to_numpy(), index=idx)
        sm = pd.Series(s["pair_sum"].reindex(pair).fillna(0).to_numpy(), index=idx)
        # pair rate shrunk towards the pickup-zone rate
        pair_rate = (sm + self.m * pu_rate) / (n + self.m)
        pdist = pd.Series(s["pair_dist"].reindex(pair).to_numpy(), index=idx)
        fallback = pu.map(s["pu_dist"]).fillna(s["global_dist"]).astype(float)
        plan = pdist.where(n >= self.min_pair_n, fallback)
        sl = pd.MultiIndex.from_arrays([pu.to_numpy(), slot.to_numpy()])
        sn = pd.Series(s["slot_n"].reindex(sl).fillna(0).to_numpy(), index=idx)
        ss = pd.Series(s["slot_sum"].reindex(sl).fillna(0).to_numpy(), index=idx)
        slot_rate = (ss + self.m * pu_rate) / (sn + self.m)
        return pd.DataFrame({
            "plan_dist": plan, "log_plan_dist": np.log1p(plan),
            "pu_rate": pu_rate, "do_rate": do_rate,
            "pair_rate": pair_rate, "log_pair_n": np.log1p(n),
            "pu_slot_rate": slot_rate,
        }, index=idx)

    @staticmethod
    def _keys(df):
        t = pd.to_datetime(df["tpep_pickup_datetime"])
        return df["PULocationID"], df["DOLocationID"], time_slot(t.dt.hour, t.dt.dayofweek)

    # -- sklearn-style API -----------------------------------------------
    def fit(self, df: pd.DataFrame, y):
        pu, do, slot = self._keys(df)
        self.stats_ = self._learn(pu, do, slot, df["trip_distance"], y)
        self.n_train_ = len(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pu, do, slot = self._keys(df)
        return self._apply(self.stats_, pu, do, slot)

    def fit_transform(self, df: pd.DataFrame, y) -> pd.DataFrame:
        """Out-of-fold statistics for the training rows; then fit on all of them."""
        y = np.asarray(y)
        pu, do, slot = self._keys(df)
        parts = []
        kf = KFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for fit_idx, out_idx in kf.split(df):
            s = self._learn(pu.iloc[fit_idx], do.iloc[fit_idx], slot.iloc[fit_idx],
                            df["trip_distance"].iloc[fit_idx], y[fit_idx])
            parts.append(self._apply(s, pu.iloc[out_idx], do.iloc[out_idx], slot.iloc[out_idx]))
        self.fit(df, y)
        return pd.concat(parts).loc[df.index]


class CrawlFeatures(BaseEstimator, TransformerMixin):
    """Pipeline step: raw trip rows -> every allowed feature (``ALL_FEATS``).

    Reads only ``tpep_pickup_datetime``, ``PULocationID``, ``DOLocationID``
    and the two borough names. ``trip_distance`` and the label are read in
    ``fit`` / ``fit_transform`` on the training rows only.
    """

    def __init__(self, m: float = 20.0, min_pair_n: int = 5, n_folds: int = 5, oof: bool = True):
        self.m, self.min_pair_n, self.n_folds, self.oof = m, min_pair_n, n_folds, oof

    def _hist(self):
        return ZoneHistory(m=self.m, min_pair_n=self.min_pair_n, n_folds=self.n_folds)

    def fit(self, X, y):
        self.history_ = self._hist().fit(X, y)
        return self

    def transform(self, X):
        allowed = X[["tpep_pickup_datetime", "PULocationID", "DOLocationID",
                     "pickup_borough", "dropoff_borough"]]
        out = pd.concat([calendar_and_zone_flags(allowed), self.history_.transform(allowed)], axis=1)
        return out[ALL_FEATS]

    def fit_transform(self, X, y=None, **kw):
        if not self.oof:                      # only used to show why OOF matters
            return self.fit(X, y).transform(X)
        self.history_ = self._hist()
        hist = self.history_.fit_transform(X, y)
        out = pd.concat([calendar_and_zone_flags(X), hist], axis=1)
        return out[ALL_FEATS]


def build_matrix(train: pd.DataFrame, others: dict[str, pd.DataFrame] | None = None,
                 m: float = 20.0) -> tuple[dict[str, pd.DataFrame], ZoneHistory]:
    """Feature tables for the training rows (out-of-fold) and any other split.

    Returns ``M`` with ``M["train"]`` and one entry per key of ``others``, and
    the fitted ``ZoneHistory`` (learned from ``train`` only).
    """
    step = CrawlFeatures(m=m)
    M = {"train": step.fit_transform(train, train["crawl"].to_numpy())}
    for name, df in (others or {}).items():
        M[name] = step.transform(df)
    return M, step.history_


def check_no_forbidden(feats: list[str]) -> None:
    bad = sorted(set(feats) & set(FORBIDDEN + NOT_IN_TRAINING))
    assert not bad, f"forbidden features used: {bad}"
    unknown = sorted(set(feats) - set(ALL_FEATS))
    assert not unknown, f"features not produced by CrawlFeatures: {unknown}"


def make_prep(feats: list[str], scale: bool = True) -> ColumnTransformer:
    """Column selection. ``scale=True`` (logistic regression): hour and weekday
    one-hot, the rest standardised. ``scale=False`` (tree): passed through."""
    check_no_forbidden(feats)
    if scale:
        cat = [f for f in feats if f in CATEGORICAL]
        num = [f for f in feats if f not in CATEGORICAL]
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ])
    return ColumnTransformer([("keep", "passthrough", feats)])


def make_model(model, feats: list[str], scale: bool = True) -> Pipeline:
    """prep -> model, for a feature table already built by ``build_matrix``."""
    return Pipeline([("prep", make_prep(feats, scale)), ("model", model)])


def make_pipeline(model, feats: list[str], scale: bool = True, oof: bool = True) -> Pipeline:
    """Leak-free pipeline: raw trip rows in, crawl probability out.

    features (learned from the training rows only, out-of-fold) -> prep -> model.
    """
    return Pipeline([("features", CrawlFeatures(oof=oof)),
                     ("prep", make_prep(feats, scale)), ("model", model)])

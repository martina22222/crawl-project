"""Cleaning: rebuild the labelled, split dataset from the raw files.

Everything here starts from the files in ``data/raw/`` (downloaded by
``python data/get_data.py``) and nothing is cached, so anyone who clones the
repository gets exactly the same table.

Main entry points
-----------------
load_raw()        -> (trips, zones) exactly as published
target_ladder()   -> positive rate after each labelling decision (notebook 01)
audit_raw()       -> findings table: every problem, how many rows, fix / flag / drop
build_dataset()   -> (clean labelled table with a ``split`` column, step log)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TRIPS_FILE = "yellow_tripdata_2026_sample.parquet"
ZONES_FILE = "taxi_zone_lookup.csv"
SHAPE_FILE = "taxi_zones.zip"

# --- constants shared by the whole project --------------------------------
CRAWL_MPH = 6.0              # target: average speed under this is a crawl
MIN_DURATION_MIN = 1.0       # under 1 minute the speed is rounding noise
MAX_DURATION_MIN = 180.0     # over 3 hours the meter was left running
MIN_DISTANCE_MI = 0.1        # under ~160 m it is not a real trip
MAX_SPEED_MPH = 80.0         # impossible in New York: a measurement error

AIRPORT_ZONES = {1, 132, 138}        # Newark, JFK, LaGuardia
UNKNOWN_ZONES = {264, 265}           # "Unknown" and "Outside of NYC"
SPLIT_BY_MONTH = {5: "train", 6: "validation", 7: "test"}
YEAR = 2026

# payment_type codes (data dictionary): 0 flex fare, 1 credit card, 2 cash,
# 3 no charge, 4 dispute, 5 unknown, 6 voided trip
FLEX, NO_CHARGE, DISPUTE, VOIDED = 0, 3, 4, 6
# RatecodeID: 1 standard, 2 JFK, 3 Newark, 4 Nassau/Westchester,
# 5 negotiated, 6 group ride, 99 unknown
STANDARD_RATE = 1


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(raw_dir: Path | str = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the trips and the zone lookup exactly as published."""
    raw_dir = Path(raw_dir)
    trips_path, zones_path = raw_dir / TRIPS_FILE, raw_dir / ZONES_FILE
    if not trips_path.exists():
        raise FileNotFoundError(
            f"{trips_path} not found. Run `python data/get_data.py` from the repository root first."
        )
    trips = pd.read_parquet(trips_path)
    zones = pd.read_csv(zones_path)
    return trips, zones


def amount_columns(trips: pd.DataFrame) -> list[str]:
    """Money columns present in this file (names vary a little between TLC releases)."""
    wanted = ["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
              "improvement_surcharge", "total_amount", "congestion_surcharge",
              "airport_fee", "Airport_fee", "cbd_congestion_fee"]
    return [c for c in wanted if c in trips.columns]


def add_speed(trips: pd.DataFrame) -> pd.DataFrame:
    """Add duration (minutes) and average speed (mph). Speed is NaN where undefined."""
    out = trips.copy()
    pu = pd.to_datetime(out["tpep_pickup_datetime"])
    do = pd.to_datetime(out["tpep_dropoff_datetime"])
    out["duration_min"] = (do - pu).dt.total_seconds() / 60.0
    hours = out["duration_min"] / 60.0
    ok = (hours > 0) & (out["trip_distance"] > 0)
    out["speed_mph"] = np.where(ok, out["trip_distance"] / hours.where(ok), np.nan)
    return out


def _in_month(trips: pd.DataFrame, month: int | None) -> pd.DataFrame:
    if month is None:
        return trips
    pu = pd.to_datetime(trips["tpep_pickup_datetime"])
    return trips[(pu.dt.year == YEAR) & (pu.dt.month == month)]


# --------------------------------------------------------------------------
# The target and which trips can be labelled (used in notebook 01)
# --------------------------------------------------------------------------
def labelable_mask(t: pd.DataFrame) -> pd.Series:
    """True where the speed is meaningful enough to be a label (needs add_speed first)."""
    return (
        (t["duration_min"] > 0) & (t["trip_distance"] > 0)
        & t["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN)
        & (t["trip_distance"] >= MIN_DISTANCE_MI)
        & (t["speed_mph"] <= MAX_SPEED_MPH)
    )


def target_ladder(trips: pd.DataFrame, zones: pd.DataFrame | None = None,
                  month: int | None = None) -> pd.DataFrame:
    """Rows and positive rate after each labelling decision.

    ``month=5`` restricts to May pickups, the training month, so no June or
    July label is looked at while the target is defined.
    """
    t = add_speed(_in_month(trips, month))
    rows = []

    # 0. naive: every row, zero distances count as crawls (speed 0),
    #    zero or negative durations give inf / NaN and count as not crawl
    naive = (t["duration_min"] > 0) & ((t["trip_distance"] / (t["duration_min"] / 60)) < CRAWL_MPH)
    rows.append(("all rows, naive (speed < 6 incl. zero distance, <=0 duration -> inf/NaN as not crawl)",
                 len(t), naive.mean()))

    steps = [
        ("raw, speed defined only where duration > 0 and distance > 0",
         (t["duration_min"] > 0) & (t["trip_distance"] > 0)),
        ("+ duration between 1 and 180 min",
         t["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN)),
        ("+ distance >= 0.1 mi", t["trip_distance"] >= MIN_DISTANCE_MI),
        ("+ speed <= 80 mph", t["speed_mph"] <= MAX_SPEED_MPH),
    ]
    keep = pd.Series(True, index=t.index)
    for name, cond in steps:
        keep &= cond.fillna(False)
        kept = t[keep]
        rows.append((name, len(kept), (kept["speed_mph"] < CRAWL_MPH).mean()))
    return pd.DataFrame(rows, columns=["step", "rows", "positive_rate"])


# --------------------------------------------------------------------------
# Audit: every problem found in the raw data, counted on the raw rows
# --------------------------------------------------------------------------
def _problems(t: pd.DataFrame) -> list[tuple[str, pd.Series, str, str]]:
    """(problem, boolean mask on raw rows, decision, why). Needs add_speed first."""
    pu = pd.to_datetime(t["tpep_pickup_datetime"])
    money = amount_columns(t)
    neg_money = (t[money] < 0).any(axis=1) if money else pd.Series(False, index=t.index)
    pay = t["payment_type"]
    rate = t["RatecodeID"]
    pc = t["passenger_count"]
    return [
        ("pickup outside May to July 2026",
         ~((pu.dt.year == YEAR) & pu.dt.month.isin(SPLIT_BY_MONTH)),
         "drop", "belongs to no split; the split is by pickup month"),
        ("exact duplicate row", t.duplicated(),
         "drop", "the same trip counted twice would weigh double"),
        ("drop-off not after pickup (duration <= 0)", t["duration_min"] <= 0,
         "drop", "speed undefined: no label possible"),
        ("zero distance", t["trip_distance"] <= 0,
         "drop", "speed 0 would fake a crawl; usually a cancelled or voided ride"),
        ("duration under 1 min", (t["duration_min"] > 0) & (t["duration_min"] < MIN_DURATION_MIN),
         "drop", "speed is rounding noise"),
        ("duration over 180 min", t["duration_min"] > MAX_DURATION_MIN,
         "drop", "meter left running; not a real duration"),
        ("distance under 0.1 mi (but > 0)", (t["trip_distance"] > 0) & (t["trip_distance"] < MIN_DISTANCE_MI),
         "drop", "not a real trip; speed meaningless"),
        ("speed over 80 mph", t["speed_mph"] > MAX_SPEED_MPH,
         "drop", "impossible in New York: distance or clock error"),
        ("unknown zone (264/265) at pickup or drop-off",
         t["PULocationID"].isin(UNKNOWN_ZONES) | t["DOLocationID"].isin(UNKNOWN_ZONES),
         "drop", "no place to learn from, cannot be mapped, and the warning needs a known zone"),
        ("negative amount in any money column", neg_money,
         "drop", "reversals / corrections, not real trips"),
        ("payment no charge / dispute / voided (3, 4, 6)", pay.isin([NO_CHARGE, DISPUTE, VOIDED]),
         "drop", "trip did not happen as recorded or is contested"),
        ("flex fare trip (payment_type 0)", pay == FLEX,
         "flag", "a quarter of real trips; dropping them would bias the sample. Kept, flagged "
                 "`is_flex`, never a feature (payment is settled at the end), and reported as a slice"),
        ("non-standard rate code (not 1, incl. 99 unknown)", rate.notna() & (rate != STANDARD_RATE),
         "flag", "real trips (airport flat fares, negotiated). Kept and flagged `nonstandard_rate`; "
                 "RatecodeID is set at the end so it is never a feature"),
        ("rate code missing", rate.isna(),
         "flag", "almost all flex trips; nothing to fix, flag kept for slicing"),
        ("RatecodeID 99 (unknown)", rate == 99,
         "fix", "recoded to missing so 99 is never read as a number"),
        ("passenger count 0 or missing", pc.isna() | (pc == 0),
         "fix", "set to missing; not used as a feature (entered by the driver, missing for flex)"),
        ("airport trip (zone 1, 132, 138 at either end)",
         t["PULocationID"].isin(AIRPORT_ZONES) | t["DOLocationID"].isin(AIRPORT_ZONES),
         "flag", "real and important; kept, becomes the `pu_airport` / `do_airport` features"),
        ("distance over 50 mi", t["trip_distance"] > 50,
         "flag", "rare long rides; speed is still plausible so kept, flagged for review"),
        ("fee column missing (congestion_surcharge / Airport_fee)",
         t[[c for c in ("congestion_surcharge", "Airport_fee", "airport_fee") if c in t]].isna().any(axis=1),
         "flag", "blank on flex trips, not negative; left missing, money is never a feature"),
        ("request_source filled (column empty for every May trip)",
         t["request_source"].notna() if "request_source" in t else pd.Series(False, index=t.index),
         "flag", "only on flex trips from June on (app or hail booking source), so nothing can be learned "
                 "from it in May: column kept for slicing, never a feature; a sign the trip mix shifts after May"),
    ]


def audit_raw(trips: pd.DataFrame) -> pd.DataFrame:
    """Findings table counted independently on the raw rows (a row can have several problems)."""
    t = add_speed(trips)
    out = []
    for problem, mask, decision, why in _problems(t):
        mask = mask.fillna(False)
        out.append({"problem": problem, "rows": int(mask.sum()),
                    "share_%": round(100 * mask.mean(), 3), "decision": decision, "why": why})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# The cleaning function
# --------------------------------------------------------------------------
def build_dataset(raw_dir: Path | str = RAW, verbose: bool = False
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the clean, labelled, split table from the raw files.

    Returns
    -------
    df : one row per kept trip, raw columns + ``duration_min``, ``speed_mph``,
         ``crawl`` (target), zone names, flags and ``split``
         (train = May, validation = June, test = July, by pickup time).
    log : the drop steps applied in order, with rows removed, rows left and
          the positive rate after each step.
    """
    trips, zones = load_raw(raw_dir)
    t = add_speed(trips)

    log = [{"step": "raw file", "removed": 0, "rows_left": len(t),
            "positive_rate": np.nan}]
    for problem, mask, decision, _ in _problems(t):
        if decision != "drop":
            continue
        # masks were computed on the raw rows; keep only the rows still present
        mask = mask.fillna(False).reindex(t.index, fill_value=False)
        before = len(t)
        t = t[~mask]
        pos = (t["speed_mph"] < CRAWL_MPH).mean() if len(t) else np.nan
        log.append({"step": f"drop: {problem}", "removed": before - len(t),
                    "rows_left": len(t), "positive_rate": pos})
        if verbose:
            print(f"{problem:<55} -{before - len(t):>7,}  left {len(t):>8,}")
    log = pd.DataFrame(log)

    # fixes
    t = t.copy()
    t.loc[t["RatecodeID"] == 99, "RatecodeID"] = np.nan
    t.loc[t["passenger_count"] == 0, "passenger_count"] = np.nan

    # flags (kept rows, never features)
    t["is_flex"] = (t["payment_type"] == FLEX).astype("int8")
    t["nonstandard_rate"] = (t["RatecodeID"].notna() & (t["RatecodeID"] != STANDARD_RATE)).astype("int8")
    t["long_trip"] = (t["trip_distance"] > 50).astype("int8")

    # the target
    t["crawl"] = (t["speed_mph"] < CRAWL_MPH).astype("int8")

    # zone names, with a row-count check (the lookup must not duplicate trips)
    z = zones[["LocationID", "Borough", "Zone"]]
    n = len(t)
    t = t.merge(z.rename(columns={"LocationID": "PULocationID", "Borough": "pickup_borough",
                                  "Zone": "pickup_zone"}), on="PULocationID", how="left")
    t = t.merge(z.rename(columns={"LocationID": "DOLocationID", "Borough": "dropoff_borough",
                                  "Zone": "dropoff_zone"}), on="DOLocationID", how="left")
    assert len(t) == n, "zone lookup duplicated rows"

    # the split, by pickup month
    t["tpep_pickup_datetime"] = pd.to_datetime(t["tpep_pickup_datetime"])
    t["tpep_dropoff_datetime"] = pd.to_datetime(t["tpep_dropoff_datetime"])
    t["month"] = t["tpep_pickup_datetime"].dt.month.astype("int8")
    t["split"] = t["month"].map(SPLIT_BY_MONTH)
    assert t["split"].notna().all()

    t = t.sort_values("tpep_pickup_datetime", kind="stable").reset_index(drop=True)
    return t, log


def load_zone_shapes(raw_dir: Path | str = RAW):
    """Zone polygons (GeoDataFrame, one row per LocationID). Coordinates are in feet."""
    import zipfile
    import geopandas as gpd

    raw_dir = Path(raw_dir)
    out_dir = raw_dir / "taxi_zones"
    if not any(out_dir.rglob("*.shp")):
        with zipfile.ZipFile(raw_dir / SHAPE_FILE) as zf:
            zf.extractall(out_dir)
    shp = next(p for p in out_dir.rglob("*.shp") if not p.name.startswith("._"))
    g = gpd.read_file(shp)
    # a few zones are split into several polygons: merge them into one row
    return g.dissolve(by="LocationID", as_index=False)[["LocationID", "geometry"]]

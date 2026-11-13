# PROBLEM.md - Will this trip crawl?

## Problem card (summary)

| | |
|---|---|
| **Question** | At pickup, will this yellow-taxi trip average under 6 mph, so a dispatcher can warn the passenger? |
| **Target** | `crawl` = 1 if `trip_distance / duration_hours < 6` (labelling decisions below) |
| **One row is** | One cleaned yellow-taxi trip (the raw table has no trip ID; zones and trips are joined on `LocationID` with a row-count check) |
| **Predict at** | The moment the trip starts (pickup) |
| **Allowed information** | Pickup time, pickup zone, drop-off zone,boroughs and airport flags derived from them, and zone, zone-pair and zone-by-time-slot crawl rates learned from May only. An estimated planned distance was tested and dropped (see below) |
| **Left out** | Drop-off time, duration, speed, measured `trip_distance`, `RatecodeID`, `payment_type`, all fares, fees, tips and totals, `passenger_count`; trips with zero or impossible distance or duration, negative amounts, unknown zones, no-charge / dispute / voided payments |
| **Positive rate** | 19.31% in May after all labelling decisions (18.5% June, 16.3% July) |
| **Decision it supports** | Warn the passenger (or not) before the trip starts |
| **Costly mistake** | A missed crawl (cost 3) is worse than a needless warning (cost 1). Break-even precision = 1/3 |
| **Metric** | PR AUC and PR AUC lift (never accuracy alone); precision and recall at the cost-chosen threshold; expected cost per 1,000 trips |
| **Baseline to beat** | Depth-2 tree: June PR AUC 0.450, cost 349 per 1,000 trips at its best June threshold (364 at the fixed 0.30 used in notebook 01). See the ladder below. |

## The question (one sentence)
At the moment a yellow-taxi trip starts, can we tell from the pickup time, the pickup and drop-off zones and the learned zone statistics whether its average speed will end up under 6 mph, so a dispatcher can warn the passenger?

## Type and target
* Binary classification, predicted **at pickup**.
* **Target:** `crawl = 1` if `trip_distance / (duration in hours) < 6 mph`, else 0.

## Which trips can be labelled (every decision and the positive rate after it)
Computed on **May only** (the training month, `src/clean.py::target_ladder(month=5)`), so no June or July label is looked at while the target is defined. Notebook 01 also prints the same table on all months for comparison (20.96% naive down to 18.04%); the steps change the rate by the same amounts.

| Step | Trips (May) | Positive rate |
|---|---|---|
| Naive: speed < 6 on all rows (zero distances count as crawls) | 199,999 | 21.70% |
| Speed defined only where duration > 0 and distance > 0 | 191,885 | 19.76% |
| + duration between 1 and 180 min | 191,180 | 19.69% |
| + distance >= 0.1 mile | 190,276 | 19.31% |
| + speed <= 80 mph | 190,244 | 19.31% |

Why each decision: zero or negative durations make speed undefined; under 1 minute the speed is rounding noise; over 3 hours the meter was left running; under 0.1 mile the distance is not a real trip and speed 0 would fake a crawl; over 80 mph is impossible in New York. Full cleaning (negative fares, unknown zones, payment types) is in notebook 02; 559,438 trips (93.2%) are kept across the three months.

## The moment of prediction
**Known at pickup (allowed):** pickup time (hour, weekday), pickup zone, drop-off zone, boroughs and airport flags derived from them, the planned distance and statistics learned from past trips (zone, zone-pair and zone-by-time-slot crawl rates). The estimated planned distance was tested as a candidate and is not in the final model (see below).”
**Forbidden (never used as a feature):** drop-off time, duration, speed, measured `trip_distance`, `RatecodeID`, `payment_type`, fare/extra/tax/tip/tolls/total/surcharges, `passenger_count`. These are either parts of the target or only settled at the end of the trip. The list is `FORBIDDEN` in `src/features.py` and notebook 03 asserts that no feature is on it.

## The planned distance: allowed, tested, and dropped
The target is distance divided by duration. The *measured* distance is only known once the trip is finished, so it is not allowed. The *planned* distance (what a routing engine says when the ride is booked) is known before any driving, and says nothing about the traffic that will determine the duration. It is therefore a legitimate input. On its own it helps a lot: adding it after the time and zone flags raises June PR AUC by about 0.06 (short routes crawl, long routes do not).

The raw files hold only the measured distance, so we **estimate** the planned distance as the median measured distance of *earlier* trips between the same two zones, learned from May only (and computed out-of-fold within May). Once the zone and zone-pair crawl rates are in, the estimate adds almost nothing (+0.001 June PR AUC for the logistic regression, 0 for the tree), because the pair rate already carries the route length, so the final model drops it. An oracle that uses the real measured distance gains under 0.01 (notebook 03), so the model is not missing much by not knowing the actual route.

## Who or when will the model be used on, and the split
The model will be used on **future trips** in the same city, by the same fleet, on zones and zone pairs it has already seen. So the split is **by time**: May train, June validation and every choice, July one final test. A grouped split (new people or shops) does not apply, because the data has no driver or passenger ID and zone pairs repeat by design. A random split of May and June gives almost the same baseline scores as the time split (notebook 02, section 7), so the real shift shows up only in July, where the positive rate falls to 16.3%.

## Lift
Two lifts are reported. **PR AUC lift** = PR AUC / positive rate (1 means no better than guessing); it is the fair way to compare months because the no-skill PR AUC equals the positive rate, which moves from 18.5% (June) to 16.3% (July). **Top-10% lift** = crawl rate among the 10% highest-scored trips / overall crawl rate.

## Baseline ladder (June, validation)
| Baseline | PR AUC | PR AUC lift | Precision | Recall | Cost per 1,000 trips (FN:FP = 3:1) |
|---|---|---|---|---|---|
| 1. Always "does not crawl" | 0.185 | 1.00 | 0 | 0 | 556 |
| 2. Weekdays 8-19h, pickup in 20 busiest zones | 0.244 | 1.31 | 0.307 | 0.480 | 490 |
| 3. Decision tree, depth 2 | 0.450 | 2.43 | 0.616 | 0.437 | 364 |

Baseline 1 is 81.5% accurate and useless, which is why accuracy is never reported alone.
Precision, recall and cost are at a threshold of 0.30 for baseline 3. At its best June threshold (notebook 03) the same tree has precision 0.462, recall 0.607 and cost 349.

## Timeline: what is known at pickup

```
meter engaged (pickup)  ->  trip in progress  ->  meter disengaged (drop-off)  ->  payment
        ^ we predict here
```

| Column | Known at pickup? | Used? |
|---|---|---|
| `tpep_pickup_datetime` | Yes | Yes (hour, weekday) |
| `PULocationID`, `DOLocationID` | Yes (booked destination) | Yes (zone flags, learned zone rates) |
| `trip_distance` | No, measured during the trip | No. A planned distance was estimated from May zone-pair medians, tested, and dropped |
| `tpep_dropoff_datetime`, duration, speed | No, part of the target | No |
| `RatecodeID` | No, set at the end | No |
| `payment_type` | No, settled at the end | No (`is_flex` only reported as a rejected feature) |
| `fare_amount`, `extra`, `mta_tax`, `tolls_amount`, surcharges, `cbd_congestion_fee`, `tip_amount`, `total_amount` | No, settled at the end | No |
| `passenger_count` | Entered by the driver; missing for flex trips | No |



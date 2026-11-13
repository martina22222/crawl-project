# MODEL_CARD.md - Will this trip crawl?

## What it does
At the moment a New York yellow-taxi trip starts, the model gives the probability that the trip will average **under 6 mph** (a "crawl"). A dispatcher warns the passenger when the probability is **0.26 or higher**.

| | |
|---|---|
| **Model** | Logistic regression (`C = 1`), inside the leak-free pipeline `make_pipeline` in `src/features.py` |
| **Trained on** | May 2026 pickups only (187,694 cleaned trips, 19.3% crawl) |
| **Chosen on** | June 2026 (186,184 trips, 18.5% crawl): features, model, threshold |
| **Tested on** | July 2026 (185,560 trips, 16.3% crawl), scored once in notebook 04 |
| **Inputs** | pickup hour, weekday, `is_rush` (weekday 8-19h), and four crawl rates learned from May: pickup zone, drop-off zone, the zone pair, and the pickup zone in this time slot |
| **Never used** | anything settled during or after the trip: drop-off time, duration, measured distance, rate code, payment type, any amount, passenger count (`FORBIDDEN` in `src/features.py`, tested in notebook 03) |

## Results

| | June (validation) | July (test, once) | July 95% interval |
|---|---|---|---|
| PR AUC | 0.627 | **0.610** | 0.604 to 0.616 |
| PR AUC lift (PR AUC / crawl rate) | 3.38 | **3.74** | 3.70 to 3.78 |
| Precision at 0.26 | 0.506 | **0.455** | 0.450 to 0.459 |
| Recall at 0.26 | 0.717 | **0.767** | 0.762 to 0.772 |
| Expected cost per 1,000 trips | 287 | **264** | 261 to 267 |
| Cost if the dispatcher never warns | 556 | 490 | |

Baselines on July (PR AUC): always "no crawl" 0.163, busy-hours rule 0.217, depth-2 tree 0.432. The model beats all three, and cuts the expected cost of never warning by about 46%.

The intervals come from 1,000 bootstrap resamples of July trips. They show sampling noise **within July only**, not the change from one month to the next.

## The threshold and the costs behind it
* **A missed crawl costs 3.** The passenger was not warned, sits in traffic at walking pace, misses what they were going to, and loses trust in the service. This is the harm the product exists to prevent.
* **A needless warning costs 1.** Some passengers cancel and lose a fare's worth of time finding another way, and if warnings are often wrong people stop reading them.

The threshold with the lowest expected cost on June is **0.26** (close to the 0.25 that perfectly calibrated scores would give). If you weigh the two mistakes differently, use this table (June) instead of re-running anything:

| Missed crawl : needless warning | Threshold | Precision | Recall | Trips warned |
|---|---|---|---|---|
| 1 : 1 | 0.52 | 0.69 | 0.43 | 11% |
| 2 : 1 | 0.34 | 0.57 | 0.62 | 20% |
| **3 : 1 (ours)** | **0.26** | **0.51** | **0.72** | **26%** |
| 5 : 1 | 0.18 | 0.43 | 0.82 | 35% |
| 10 : 1 | 0.11 | 0.35 | 0.91 | 49% |

## Which trips the warning misses
In July, **about 1 in 4 crawls gets no warning** (recall 0.767), and **just over half of the warnings are needless** (precision 0.455). On June, the missed crawls are:

* **Evenings, nights and weekends.** Weekday daytime crawls are caught well (recall 0.83 for 6-15h on June). The warning misses about half of weekend crawls (48%) and evening crawls 20-23h (58%), and almost all crawls between midnight and 6am (95%). Weekends and evenings together are 47% of all missed crawls.
* **Anything outside Manhattan.** 99% of crawls are inside Manhattan, and that is where the model works. The rare crawls in Brooklyn, Queens or the Bronx, or between Manhattan and another borough, are missed 85 to 97% of the time.
* **Crawls on routes that are usually fast.** The model only knows how slow a route usually is at that time. It cannot see accidents, events, closures or weather, so a crawl on a normally quick route (a 2-5 mile trip, for example: 48% missed) goes unwarned.
* **Flex fare trips.** About 47% of flex-fare crawls are missed, against 25% for metered trips.
* **Zones with the most unwarned crawls:** Upper East Side North, East Village, Upper East Side South, West Village, Lincoln Square East (East Village and Upper West Side South miss more than two thirds of their crawls).

## Reliability
On June the scores are close to calibrated but **slightly too high** in every group (for example, trips scored 0.44 on average crawled 43%, trips scored 0.74 crawled 71%). That is expected: the model learned on May, when crawls were more common. The threshold was chosen on June, which absorbs this.

## Known risks and limits
* **The crawl rate drifts:** 19.3% in May, 18.5% in June, 16.3% in July. Precision falls as crawls become rarer. **Re-check the threshold every month** on the most recent month before using it.
* **The trip mix changes after May.** From June a new column `request_source` appears, filled only for app-booked flex trips, which crawl about half as often as other trips. May has none of it, so the model never learned from it.
* **No planned route.** The data has no booked route. A planned distance was estimated from past trips between the same two zones, but once the zone-pair crawl rate is in, it adds nothing. Notebook 03 shows that even the true measured distance would add under 0.01 PR AUC.
* **No traffic, weather or event data.** This is the main reason unusual crawls are missed.
* **The costs are our assumption.** They are argued above, not measured; the table lets a user pick their own.
* **Zones only**, no exact locations. Zone pairs never seen in May fall back to the zone rates.

## Intended use
A heads-up for passengers before a trip in New York yellow taxis, mainly **weekday daytime trips inside Manhattan**, where it is reliable. It is not meant to promise an arrival time, to price trips, or to judge drivers.

## How to reproduce
```bash
python data/get_data.py
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```
Numbers are saved in `reports/july_results.json`; figures in `reports/figures/04_*.png`.

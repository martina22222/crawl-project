# Will this trip crawl? (Team C, Project 1)

Predicts, at pickup, whether a New York yellow-taxi trip will average under 6 mph, so a dispatcher can warn the passenger. Data: 600,000 TLC yellow-taxi trips, May to July 2026. 

## Problem
Binary classification at pickup. Target: `trip_distance / duration < 6 mph`. Success is measured with PR AUC, lift, precision and recall at a threshold chosen from stated costs (a missed crawl costs 3, a needless warning 1). Full detail in `PROBLEM.md`; results and failures in `MODEL_CARD.md`.

## Data
Source: NYC Taxi and Limousine Commission yellow taxi trip records, a fixed 200,000-trip sample per month (May, June, July 2026) distributed by the course. Public data, not committed to this repository. `python data/get_data.py` downloads it to `data/raw/`.

## Setup
```bash
git clone https://github.com/martina22222/crawl-project.git && cd crawl-project
python -m venv .venv
source .venv/bin/activate            # Windows Git Bash: source .venv/Scripts/activate
python -m pip install -r requirements.txt
python data/get_data.py
```
Verified on Python 3.12.3 with the current versions pinned in requirements.txt.

## Running it
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01-frame.ipynb notebooks/02-clean-and-split.ipynb notebooks/03-features.ipynb notebooks/04-evaluate.ipynb
```
Takes a few minutes. Figures go to `reports/figures/`. Notebook 04 is the only place July is scored.

## Layout
| Path | What |
|---|---|
| `data/get_data.py` | download script (raw data is not committed) |
| `src/clean.py` | `build_dataset()` (rebuilds the clean, labelled, split table from raw), `audit_raw()` (findings table), `target_ladder()` |
| `src/features.py` | leak-free features, `ZoneHistory` (learns from May only, out-of-fold within May), `FORBIDDEN` list, `make_pipeline()` |
| `src/models.py`, `src/evaluate.py` | baselines, models, metrics and costs |
| `notebooks/01..04` | frame, clean and split, features, evaluate |
| `reports/figures/` | figures saved by the notebooks |
| `reports/crawl-map.html` | interactive map, with the model embedded |
| `reports/selected_features.json` | features and model settings chosen on June in notebook 03, loaded by notebook 04 |

## Results

Scored once on July (notebook 04) with the logistic regression and a threshold of 0.26, chosen on June. Intervals are 95% bootstrap intervals from 1,000 resamples of July trips.

| Metric | June (validation) | July (test) | July 95% interval |
|---|---|---|---|
| PR AUC | 0.627 | 0.610 | 0.604 to 0.616 |
| PR AUC lift | 3.38 | 3.74 | 3.70 to 3.78 |
| Precision at 0.26 | 0.506 | 0.455 | 0.450 to 0.459 |
| Recall at 0.26 | 0.717 | 0.767 | 0.762 to 0.772 |
| Cost per 1,000 trips | 287 | 264 | 261 to 267 |
| Cost per 1,000 trips if never warning | 556 | 490 | |

Baselines on July (PR AUC):

| Baseline | PR AUC |
|---|---|
| Always "no crawl" | 0.163 |
| Busy-hours rule | 0.217 |
| Depth-2 tree | 0.432 |
| **Logistic regression (chosen)** | **0.610** |

The intervals show sampling noise within July only, not the change from one month to the next. Details are in `MODEL_CARD.md`.


## Limitations
In July about 1 in 4 crawls gets no warning and just over half of the warnings are needless. On June the warning misses about half of weekend crawls, most evening crawls, almost all night crawls, nearly everything outside Manhattan, and 47% of flex-fare crawls. There is no weather, traffic or event data, the costs are our assumptions, and the crawl rate falls from 19.3% (May) to 16.3% (July), so the threshold must be re-checked each month. See MODEL_CARD.md.

## AI assistance
Regarding the codes inside of this project, some parts were made with the help of the claude ai model.

## Team and contributions
Martina: Project structure, README, PROBLEM, notebook-1
Elisa: cleaning function and findings table (`src/clean.py`), notebook-2; leak-free features and pipeline (`src/features.py`), baselines and metrics (`src/models.py`, `src/evaluate.py`), notebook-3
Megi: Week 4 evaluation (notebooks/04-evaluate.ipynb), cost-based threshold, July holdout results, reliability and slice analysis, bootstrap confidence intervals, and model card (MODEL_CARD.md).

# Will this trip crawl? (Team C, Project 1)

Predicts, at pickup, whether a New York yellow-taxi trip will average under 6 mph, so a dispatcher can warn the passenger. Data: 600,000 TLC yellow-taxi trips, May to July 2026. Result: on July, scored once, the model reaches PR AUC 0.610 (95% interval 0.604 to 0.615), 3.7 times better than guessing, and roughly halves the expected cost of never warning.

## Problem
Binary classification at pickup. Target: `trip_distance / duration < 6 mph`. Success is measured with PR AUC, lift, precision and recall at a threshold chosen from stated costs (a missed crawl costs 3, a needless warning 1). Full detail in `PROBLEM.md`; results and failures in `MODEL_CARD.md`.

## Data
Source: NYC Taxi and Limousine Commission yellow taxi trip records, a fixed 200,000-trip sample per month (May, June, July 2026) distributed by the course. Public data, not committed to this repository. `python data/get_data.py` downloads it to `data/raw/`.

## Setup
```bash
git clone <your-repo-url> && cd <your-repo-folder>
python -m venv .venv
source .venv/bin/activate            # Windows Git Bash: source .venv/Scripts/activate
python -m pip install -r requirements.txt
python data/get_data.py
```
Verified on Python 3.12.3 with the current pandas, scikit-learn and geopandas (unpinned). TODO: state the version your team uses and pin versions in `requirements.txt`.

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
| `src/models.py`, `src/evaluate.py` | baselines, models, metrics and costs (the bootstrap for July is added in week 4) |
| `notebooks/01..04` | frame, clean and split, features, evaluate |
| `reports/figures/` | figures saved by the notebooks |
| `reports/selected_features.json` | features and model settings chosen on June in notebook 03, loaded by notebook 04 |

## Results
| July (scored once) | Value | 95% interval |
|---|---|---|
| PR AUC | 0.610 | 0.604 to 0.615 |
| PR AUC lift | 3.74 | 3.69 to 3.78 |
| Precision / recall at 0.31 | 0.471 / 0.745 | 0.467 to 0.476 / 0.741 to 0.750 |
| Cost per 1,000 trips | 261 | 258 to 264 |
| Cost if never warn | 490 | |

The intervals show sampling noise within July only, not month-to-month drift. The crawl rate fell from 19.3% (May) to 16.3% (July), so precision fell and the threshold should be re-checked monthly.


## Limitations
About 30% of crawls get no warning and about half of the warnings are needless. The model is weak on weekends, evenings, nights, flex-fare trips, and outside Manhattan. The planned distance is estimated from past trips, there is no weather or traffic information, and the costs are our assumptions. See `MODEL_CARD.md`.

## AI assistance
Regarding the codes inside of this project, some parts were made with the help of the claude ai model.

## Team and contributions
Martina: Project structure, README, PROBLEM, notebook-1
Elisa: cleaning function and findings table (`src/clean.py`), notebook-2; leak-free features and pipeline (`src/features.py`), baselines and metrics (`src/models.py`, `src/evaluate.py`), notebook-3
Megi: 

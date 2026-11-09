"""Download the Project 1 data into data/raw/. Run from the repository root:
    python data/get_data.py
"""
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://github.com/evisp/ml-course-labs/releases/download/data-v2/"
FILES = [
    "yellow_tripdata_2026_sample.parquet",
    "taxi_zone_lookup.csv",
    "taxi_zones.zip",
    "data_dictionary_trip_records_yellow.pdf",
]

raw = Path(__file__).resolve().parent / "raw"
raw.mkdir(parents=True, exist_ok=True)

for name in FILES:
    if not (raw / name).exists():
        print(f"downloading {name}")
        urllib.request.urlretrieve(BASE + name, raw / name)

with zipfile.ZipFile(raw / "taxi_zones.zip") as zf:
    zf.extractall(raw)

print("ready:", sorted(p.name for p in raw.iterdir()))

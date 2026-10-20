"""Download the FRED series and build one tidy monthly panel.

Run:  python -m src.data            (uses cached CSVs in data/raw)
      python -m src.data --refresh  (re-downloads everything from FRED)
"""
import sys
import urllib.request

import pandas as pd

from .config import CATEGORIES, PROCESSED, RAW, SENTIMENT

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def download(series_id: str, refresh: bool = False) -> pd.Series:
    path = RAW / f"{series_id}.csv"
    if refresh or not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FRED_CSV.format(series_id), timeout=60) as r:
            path.write_bytes(r.read())
    df = pd.read_csv(path)
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df[series_id], errors="coerce")  # FRED marks gaps with "."
    return pd.Series(s.values, index=df["date"], name=series_id)


def build_panel(refresh: bool = False) -> pd.DataFrame:
    cols = {key: download(meta["fred"], refresh) for key, meta in CATEGORIES.items()}
    cols["sentiment"] = download(SENTIMENT, refresh)
    panel = pd.DataFrame(cols).sort_index()
    panel.index.name = "date"
    panel = panel[panel.index >= "1992-01-01"]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED / "monthly_panel.csv")
    return panel


def load_panel() -> pd.DataFrame:
    path = PROCESSED / "monthly_panel.csv"
    if not path.exists():
        return build_panel()
    return pd.read_csv(path, index_col="date", parse_dates=True)


def quality_report(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in CATEGORIES:
        s = panel[key]
        valid = s.dropna()
        full = pd.date_range(valid.index.min(), valid.index.max(), freq="MS")
        rows.append({
            "series": key,
            "first": valid.index.min().date(),
            "last": valid.index.max().date(),
            "months": len(valid),
            "missing_inside_range": len(full.difference(valid.index)),
            "duplicates": int(s.index.duplicated().sum()),
            "min": valid.min(),
            "max": valid.max(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = build_panel(refresh="--refresh" in sys.argv)
    print(quality_report(p).to_string(index=False))

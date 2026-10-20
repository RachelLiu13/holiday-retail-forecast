"""Project-wide settings: which series, which backtest origins, where files live."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

# U.S. Census Monthly Retail Trade, not seasonally adjusted, millions of dollars (via FRED).
# Not seasonally adjusted on purpose: the holiday spike is exactly what we want to forecast.
CATEGORIES = {
    "total":       {"fred": "RSXFSN",        "label": "Total Retail (ex. food services)"},
    "clothing":    {"fred": "MRTSSM448USN",  "label": "Clothing & Accessories Stores"},
    "electronics": {"fred": "MRTSSM443USN",  "label": "Electronics & Appliance Stores"},
    "hobby":       {"fred": "MRTSSM451USN",  "label": "Sporting Goods, Hobby, Book & Music Stores"},
    "general":     {"fred": "MRTSSM452USN",  "label": "General Merchandise Stores"},
    "ecommerce":   {"fred": "MRTSSM4541USN", "label": "E-commerce (Electronic Shopping & Mail-Order)"},
    "grocery":     {"fred": "MRTSSM445USN",  "label": "Grocery Stores (control: no holiday peak)"},
}

# Customer-side indicator: University of Michigan Consumer Sentiment (monthly index).
SENTIMENT = "UMCSENT"

# Each backtest stands at the end of August and forecasts September-December.
HORIZON = 4
BACKTEST_YEARS = list(range(2015, 2026))   # 11 holiday seasons, 2015-2025
FINAL_ORIGIN = "2026-08-01"                 # last month with data for every category
TRAIN_START = "1999-01-01"                  # long enough for seasonality, skips the early-1990s level shift

HOLIDAY_MONTHS = (11, 12)

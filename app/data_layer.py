"""Everything the dashboard reads, computed once from the CSVs in outputs/.

No model is fit inside the app: the forecasts and metrics come from the reproducible
pipeline in src/, so what the dashboard shows is exactly what the code produced.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CATEGORIES, FINAL_ORIGIN  # noqa: E402

OUT = ROOT / "outputs"
LABELS = {k: v["label"] for k, v in CATEGORIES.items()}
SHORT = {"total": "Total retail", "clothing": "Clothing", "electronics": "Electronics",
         "hobby": "Sporting & hobby", "general": "General merchandise",
         "ecommerce": "E-commerce", "grocery": "Grocery"}
COVID_YEARS = (2020, 2021)
RECOMMENDED = "Ensemble"

# Fixed colors per model (reference categorical palette, light mode) - color follows the
# model everywhere in the app, never its rank.
MODEL_COLORS = {
    "Ensemble": "#2a78d6", "SARIMA": "#eb6834", "ETS": "#1baf7a", "XGBoost": "#eda100",
    "Naive + Growth": "#e87ba4", "SARIMAX + Sentiment": "#008300",
    "Claude (anonymized)": "#6250d6", "GPT (anonymized)": "#e34948",
    "Claude (named)": "#9085e9", "GPT (named)": "#e66767",
    "Seasonal Naive": "#8a8984",
}
MODEL_ORDER = list(MODEL_COLORS)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _read(name, **kw):
    p = OUT / name
    return pd.read_csv(p, **kw) if p.exists() else None


def load_all():
    panel = pd.read_csv(ROOT / "data/processed/monthly_panel.csv", index_col="date", parse_dates=True)
    bt = _read("backtest_forecasts.csv", parse_dates=["date", "origin"])
    llm_bt = _read("llm_backtest_forecasts.csv", parse_dates=["date", "origin"])
    if llm_bt is not None:
        bt = pd.concat([bt, llm_bt], ignore_index=True)
    bt["year"] = bt["origin"].dt.year
    f26 = _read("forecast_2026.csv", parse_dates=["date", "origin"])
    llm26 = _read("llm_forecast_2026.csv", parse_dates=["date", "origin"])
    if llm26 is not None:
        f26 = pd.concat([f26, llm26], ignore_index=True)
    po = _read("metrics_per_origin.csv")
    reasoning = _read("llm_reasoning.csv")
    imp = _read("xgb_feature_importance.csv", index_col=0)
    return dict(panel=panel, bt=bt, f26=f26, po=po, reasoning=reasoning, imp=imp)


def models_present(df):
    return [m for m in MODEL_ORDER if m in set(df["model"])]


def holiday_table(f26, panel):
    """2026 Nov+Dec forecast vs actual Nov+Dec 2025, per category and model."""
    hol = f26[f26["date"].dt.month.isin([11, 12])]
    t = hol.groupby(["category", "model"])["mean"].sum().unstack()
    ly = panel.loc["2025-11-01":"2025-12-01"].sum()
    growth = t.div(ly[t.index], axis=0) - 1
    return t, growth, ly


def error_spread(po, category, model=RECOMMENDED, exclude_covid=True):
    """Std. dev. of the model's past holiday-season % errors for one category.

    This is the empirical uncertainty used for the planner's scenarios and safety stock.
    """
    d = po[(po.model == model) & (po.category == category)]
    if exclude_covid:
        d = d[~d.year.isin(COVID_YEARS)]
    pe = d["holiday_pe"].dropna()
    return float(np.sqrt((pe ** 2).mean())), float(pe.mean()), len(pe)


def summary_table(po, exclude_covid):
    d = po[~po.year.isin(COVID_YEARS)] if exclude_covid else po
    s = d.groupby("model").agg(holiday_APE=("holiday_ape", "mean"), MAPE=("MAPE", "mean"),
                               MASE=("MASE", "mean"), Coverage80=("Coverage80", "mean"),
                               seasons=("year", "nunique"))
    base = d[d.model == "Seasonal Naive"].set_index(["category", "year"])["holiday_ape"]
    w = d.set_index(["category", "year"]).assign(b=base)
    s["beats_naive"] = (w["holiday_ape"] < w["b"]).groupby(w["model"]).mean()
    s.loc["Seasonal Naive", "beats_naive"] = np.nan
    return s.sort_values("holiday_APE")


def ecommerce_share(panel):
    s = (panel["ecommerce"] / panel["total"]).dropna()
    ytd = {y: panel.loc[f"{y}-01":f"{y}-08", "ecommerce"].sum() / panel.loc[f"{y}-01":f"{y}-08", "total"].sum()
           for y in range(2015, 2027)}
    return s, pd.Series(ytd)


def facts_for_ai(data, planner_state=None):
    """Compact, factual context for the AI advisor. Everything here was computed by code."""
    panel, po, f26 = data["panel"], data["po"], data["f26"]
    t, growth, ly = holiday_table(f26, panel)
    s_all, s_ex = summary_table(po, False), summary_table(po, True)
    _, share_ytd = ecommerce_share(panel)
    lines = ["# Holiday Retail Planner - computed facts (all numbers from the project's code)",
             f"Data: U.S. Census Monthly Retail Trade via FRED, not seasonally adjusted, $ millions. "
             f"Category data through {panel['clothing'].dropna().index.max():%B %Y}; total retail through "
             f"{panel['total'].dropna().index.max():%B %Y}. Forecast origin: {pd.Timestamp(FINAL_ORIGIN):%B %Y}.",
             "Backtest: 11 holiday seasons (2015-2025); each time models saw data only through August and forecast Sep-Dec.",
             "", "## Backtest accuracy (holiday_APE = abs % error of the Nov+Dec total; lower is better)",
             "All seasons:", s_all.round(4).to_csv(),
             "Excluding the COVID seasons 2020-2021:", s_ex.round(4).to_csv(),
             "Holiday APE by category and model (all seasons, %):",
             (po.pivot_table(index="category", columns="model", values="holiday_ape") * 100).round(2).to_csv(),
             "", "## 2026 holiday (Nov+Dec) forecasts, $ millions, and growth vs Nov+Dec 2025",
             (t.round(0)).to_csv(), "Growth:", (growth * 100).round(2).to_csv(),
             "Actual Nov+Dec 2025:", ly[t.index].round(0).to_csv(),
             "", "## E-commerce share of total retail, Jan-Aug of each year (%)",
             (share_ytd * 100).round(2).to_csv()]
    sep = f26[(f26.category == "total") & (f26.date == "2026-09-01")].dropna(subset=["actual"])
    if len(sep):
        lines += ["", "## Early check: September 2026 total retail was published after our August origin",
                  sep[["model", "mean", "actual"]].round(0).to_csv(index=False)]
    if data.get("reasoning") is not None:
        r = data["reasoning"]
        r26 = r[r.origin.astype(str).str.startswith("2026")]
        lines += ["", "## LLM forecasters' own explanations for 2026 (named prompts)",
                  r26[["model", "category", "reasoning"]].to_csv(index=False)]
    if planner_state:
        lines += ["", "## The user's current Holiday Planner inputs and the computed result",
                  "\n".join(f"- {k}: {v}" for k, v in planner_state.items())]
    return "\n".join(lines)

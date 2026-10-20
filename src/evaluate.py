"""Accuracy metrics for every model in the backtest (including LLM forecasts if present).

Metrics
- Holiday APE: absolute % error of the November+December total, the number a retailer plans on.
- MAPE / MAE / RMSE: over all four forecast months (Sep-Dec), each origin.
- MASE: MAE scaled by the in-sample seasonal-naive MAE (computed on training data only),
  so categories of very different size can be averaged fairly. Below 1 = beats a naive
  forecast's typical in-sample error.
- Coverage80: share of actual values that fell inside the model's 80% interval.

Run:  python -m src.evaluate
"""
import numpy as np
import pandas as pd

from .config import HOLIDAY_MONTHS, OUTPUTS
from .data import load_panel

COVID_YEARS = (2020, 2021)


def load_backtest() -> pd.DataFrame:
    fc = pd.read_csv(OUTPUTS / "backtest_forecasts.csv", parse_dates=["date", "origin"])
    llm = OUTPUTS / "llm_backtest_forecasts.csv"
    if llm.exists():
        fc = pd.concat([fc, pd.read_csv(llm, parse_dates=["date", "origin"])], ignore_index=True)
    fc["year"] = fc["origin"].dt.year
    return fc


def mase_scale(panel: pd.DataFrame) -> pd.DataFrame:
    """In-sample seasonal-naive MAE per category and origin (training data only)."""
    rows = []
    for cat in panel.columns.drop("sentiment"):
        y = panel[cat].dropna()
        for year in range(2015, 2027):
            tr = y.loc[:f"{year}-08-01"]
            tr = tr[tr.index >= "1999-01-01"]
            rows.append({"category": cat, "year": year, "scale": (tr - tr.shift(12)).abs().mean()})
    return pd.DataFrame(rows)


def per_origin(fc: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per model x category x year."""
    fc = fc.dropna(subset=["actual"]).merge(mase_scale(panel), on=["category", "year"])
    fc["err"] = fc["mean"] - fc["actual"]
    fc["ape"] = (fc["err"].abs() / fc["actual"])
    hol = fc[fc["date"].dt.month.isin(HOLIDAY_MONTHS)]
    hol_tot = (hol.groupby(["model", "category", "year"])
               .agg(f=("mean", "sum"), a=("actual", "sum")).reset_index())
    hol_tot["holiday_pe"] = (hol_tot["f"] / hol_tot["a"] - 1)
    g = fc.groupby(["model", "category", "year"])
    out = g.agg(MAPE=("ape", "mean"), MAE=("err", lambda e: e.abs().mean()),
                RMSE=("err", lambda e: np.sqrt((e ** 2).mean())), scale=("scale", "first")).reset_index()
    out["MASE"] = out["MAE"] / out["scale"]
    if "lo80" in fc:
        cov = fc.dropna(subset=["lo80"]).assign(inside=lambda d: (d.actual >= d.lo80) & (d.actual <= d.hi80))
        out = out.merge(cov.groupby(["model", "category", "year"])["inside"].mean().rename("Coverage80").reset_index(),
                        how="left", on=["model", "category", "year"])
    out = out.merge(hol_tot[["model", "category", "year", "holiday_pe"]], on=["model", "category", "year"])
    out["holiday_ape"] = out["holiday_pe"].abs()
    return out.drop(columns="scale")


def summary(po: pd.DataFrame, exclude_covid: bool = False) -> pd.DataFrame:
    d = po[~po.year.isin(COVID_YEARS)] if exclude_covid else po
    s = d.groupby("model").agg(holiday_APE=("holiday_ape", "mean"), MAPE=("MAPE", "mean"),
                               MASE=("MASE", "mean"), Coverage80=("Coverage80", "mean"),
                               n=("MAPE", "size"))
    # How often each model beats the seasonal-naive baseline on holiday error
    base = d[d.model == "Seasonal Naive"].set_index(["category", "year"])["holiday_ape"]
    wins = d.set_index(["category", "year"]).assign(b=base).query("model != 'Seasonal Naive'")
    s["win_rate_vs_naive"] = (wins["holiday_ape"] < wins["b"]).groupby(wins["model"]).mean()
    return s.sort_values("holiday_APE")


def main():
    panel = load_panel()
    po = per_origin(load_backtest(), panel)
    po.to_csv(OUTPUTS / "metrics_per_origin.csv", index=False)
    s_all, s_ex = summary(po), summary(po, exclude_covid=True)
    s_all.to_csv(OUTPUTS / "metrics_summary.csv")
    s_ex.to_csv(OUTPUTS / "metrics_summary_ex_covid.csv")
    by_cat = po.pivot_table(index="category", columns="model", values="holiday_ape", aggfunc="mean")
    by_cat.to_csv(OUTPUTS / "metrics_by_category.csv")
    pd.set_option("display.width", 200)
    print("== All 11 seasons ==\n", s_all.round(4))
    print("\n== Excluding 2020-2021 ==\n", s_ex.round(4))
    print("\n== Holiday APE by category ==\n", (by_cat * 100).round(2))


if __name__ == "__main__":
    main()

"""Rolling-origin backtest of the statistical and ML models, plus the 2026 forecast.

For every year 2015-2025 we stand at the end of August, fit each model on data up to that
month only, and forecast September-December. The same code then runs once more from
August 2026 to produce the live 2026 holiday forecast.

Run:  python -m src.backtest
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import BACKTEST_YEARS, CATEGORIES, FINAL_ORIGIN, OUTPUTS
from .data import load_panel
from .models import (fc_ets, fc_sarima, fc_sarimax_sentiment, fc_snaive, fc_snaive_growth,
                     fc_xgb_global)

CATS = list(CATEGORIES)
ENSEMBLE_OF = ["SARIMA", "ETS", "XGBoost"]


def _stat_models(panel, cat, origin):
    y = panel[cat].dropna()
    out = {"Seasonal Naive": fc_snaive(y, origin),
           "Naive + Growth": fc_snaive_growth(y, origin),
           "ETS": fc_ets(y, origin)}
    sar, order = fc_sarima(y, origin)
    out["SARIMA"] = sar
    out["SARIMAX + Sentiment"] = fc_sarimax_sentiment(y, panel["sentiment"].dropna(), origin, order)
    return cat, origin, out, order


def run_origin_set(panel, origins, n_jobs=-1):
    jobs = [(c, o) for o in origins for c in CATS]
    stat = Parallel(n_jobs=n_jobs)(delayed(_stat_models)(panel, c, o) for c, o in jobs)
    rows, orders, importances = [], [], []
    for cat, origin, out, order in stat:
        orders.append({"category": cat, "origin": origin.date(), "order": str(order[0]), "seasonal_order": str(order[1])})
        for model, df in out.items():
            rows.append(df.assign(model=model, category=cat, origin=origin))
    for origin in origins:
        xg, imp = fc_xgb_global(panel, CATS, origin)
        importances.append(imp.rename(origin.date()))
        for cat, df in xg.items():
            rows.append(df.assign(model="XGBoost", category=cat, origin=origin))
    fc = pd.concat(rows).rename_axis("date").reset_index()
    # Ensemble: geometric mean of the three main model families (average in log space)
    ens = (fc[fc.model.isin(ENSEMBLE_OF)]
           .assign(logm=lambda d: np.log(d["mean"]))
           .groupby(["category", "origin", "date"], as_index=False)["logm"].mean())
    ens["mean"] = np.exp(ens.pop("logm"))
    ens["model"] = "Ensemble"
    fc = pd.concat([fc, ens], ignore_index=True)
    fc["h"] = ((fc["date"].dt.year - fc["origin"].dt.year) * 12 + fc["date"].dt.month - fc["origin"].dt.month)
    return fc, pd.DataFrame(orders), pd.concat(importances, axis=1)


def attach_actuals(fc, panel):
    long = panel[CATS].stack().rename("actual").reset_index().rename(columns={"level_1": "category"})
    return fc.merge(long, on=["date", "category"], how="left")


def main():
    panel = load_panel()
    OUTPUTS.mkdir(exist_ok=True)
    origins = [pd.Timestamp(f"{y}-08-01") for y in BACKTEST_YEARS]
    fc, orders, imp = run_origin_set(panel, origins)
    attach_actuals(fc, panel).to_csv(OUTPUTS / "backtest_forecasts.csv", index=False)
    orders.to_csv(OUTPUTS / "sarima_orders.csv", index=False)
    imp.to_csv(OUTPUTS / "xgb_feature_importance.csv")

    fc26, orders26, imp26 = run_origin_set(panel, [pd.Timestamp(FINAL_ORIGIN)])
    attach_actuals(fc26, panel).to_csv(OUTPUTS / "forecast_2026.csv", index=False)
    orders26.to_csv(OUTPUTS / "sarima_orders_2026.csv", index=False)
    print("backtest rows:", len(fc), " 2026 rows:", len(fc26))


if __name__ == "__main__":
    main()

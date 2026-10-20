"""Forecasting models. Every model sees only data up to the forecast origin.

Each `fc_*` function takes the monthly history up to the origin and returns a DataFrame indexed
by the next HORIZON months with columns `mean` (and `lo80`/`hi80` where the model gives a
statistically valid interval). Statistical models work on log sales, so their intervals are
multiplicative and the back-transformed mean is the median forecast.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import HORIZON, TRAIN_START

warnings.filterwarnings("ignore")

COVID_MONTHS = pd.date_range("2020-03-01", "2020-06-01", freq="MS")


def future_index(origin: pd.Timestamp, h: int = HORIZON) -> pd.DatetimeIndex:
    return pd.date_range(origin + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def clean_covid(y: pd.Series) -> pd.Series:
    """Treat the March-June 2020 lockdown months as outliers for model training.

    Each lockdown month is replaced by the same month of 2019 scaled by the Jan-Feb 2020
    year-over-year growth, i.e. what the series was on track to do before the shock. Only
    data before the shock is used, so this never leaks future information.
    """
    y = y.copy()
    if y.index.max() < COVID_MONTHS[0]:
        return y
    pre = y.loc["2020-01-01":"2020-02-01"].sum() / y.loc["2019-01-01":"2019-02-01"].sum()
    for m in COVID_MONTHS:
        if m in y.index:
            y.loc[m] = y.loc[m - pd.DateOffset(years=1)] * pre
    return y


def _prep(y: pd.Series) -> pd.Series:
    y = clean_covid(y.dropna())
    y = y[y.index >= TRAIN_START]
    return np.log(y).asfreq("MS")


# ---------------------------------------------------------------- baselines

def fc_snaive(y: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    """Seasonal naive: next December = last December."""
    idx = future_index(origin)
    return pd.DataFrame({"mean": [y.loc[d - pd.DateOffset(years=1)] for d in idx]}, index=idx)


def fc_snaive_growth(y: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    """Same month last year x recent year-over-year growth (average of the last 3 months).

    This is the rule of thumb many retailers actually use: 'last year plus current trend'.
    """
    idx = future_index(origin)
    recent = [np.log(y.loc[origin - pd.DateOffset(months=k)] /
                     y.loc[origin - pd.DateOffset(months=k, years=1)]) for k in range(3)]
    g = float(np.mean(recent))
    return pd.DataFrame({"mean": [y.loc[d - pd.DateOffset(years=1)] * np.exp(g) for d in idx]},
                        index=idx)


# ---------------------------------------------------------------- statistical models

def fc_ets(y: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    """ETS(A,Ad,A) on log sales: additive error, damped trend, additive monthly seasonality."""
    ly = _prep(y.loc[:origin])
    res = ETSModel(ly, error="add", trend="add", damped_trend=True,
                   seasonal="add", seasonal_periods=12).fit(disp=False, maxiter=2000)
    pred = res.get_prediction(start=len(ly), end=len(ly) + HORIZON - 1).summary_frame(alpha=0.2)
    idx = future_index(origin)
    return pd.DataFrame({"mean": np.exp(pred["mean"].values),
                         "lo80": np.exp(pred["pi_lower"].values),
                         "hi80": np.exp(pred["pi_upper"].values)}, index=idx)


SARIMA_GRID = list(itertools.product([0, 1], [0, 1], [0, 1], [0, 1]))  # p, q, P, Q


def select_sarima_order(ly: pd.Series):
    """Pick (p,1,q)(P,1,Q)12 by AIC on the training window only."""
    best = None
    for p, q, P, Q in SARIMA_GRID:
        try:
            r = SARIMAX(ly, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False, maxiter=200)
        except Exception:
            continue
        if best is None or r.aic < best[0]:
            best = (r.aic, (p, 1, q), (P, 1, Q, 12))
    return best[1], best[2]


def _sarimax_fc(ly, origin, order, sorder, exog=None, exog_future=None) -> pd.DataFrame:
    res = SARIMAX(ly, exog=exog, order=order, seasonal_order=sorder).fit(disp=False, maxiter=200)
    f = res.get_forecast(HORIZON, exog=exog_future)
    ci = f.conf_int(alpha=0.2)
    idx = future_index(origin)
    return pd.DataFrame({"mean": np.exp(f.predicted_mean.values),
                         "lo80": np.exp(ci.iloc[:, 0].values),
                         "hi80": np.exp(ci.iloc[:, 1].values)}, index=idx)


def fc_sarima(y: pd.Series, origin: pd.Timestamp, order=None) -> tuple[pd.DataFrame, tuple]:
    ly = _prep(y.loc[:origin])
    if order is None:
        order = select_sarima_order(ly)
    return _sarimax_fc(ly, origin, *order), order


def fc_sarimax_sentiment(y: pd.Series, sent: pd.Series, origin: pd.Timestamp, order) -> pd.DataFrame:
    ly = _prep(y.loc[:origin])
    # Log sentiment lagged HORIZON months: every value a 4-month-ahead forecast needs is
    # already published at the origin, so the model never peeks at future sentiment.
    ext = sent.reindex(sent.index.union(future_index(origin)))
    ext.loc[ext.index > origin] = np.nan
    ex = np.log(ext).shift(HORIZON)
    ex_train = ex.reindex(ly.index)
    keep = ex_train.notna()
    ly, ex_train = ly[keep], ex_train[keep]
    ex_future = ex.reindex(future_index(origin))
    return _sarimax_fc(ly, origin, *order, exog=ex_train.values, exog_future=ex_future.values)


# ---------------------------------------------------------------- machine learning

XGB_PARAMS = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, random_state=42)


def _xgb_rows(ly: pd.Series, lsent: pd.Series, cat_code: int, origins, with_target: bool):
    """One row per (origin month t, horizon h).

    Target: year-over-year log growth of month t+h, i.e. log y[t+h] - log y[t+h-12].
    The forecast is then last year's same month x exp(predicted growth). All features are
    known at month t.
    """
    g = ly - ly.shift(12)                    # YoY log growth series
    rows = []
    for t in origins:
        if t - pd.DateOffset(months=14) < ly.index[0]:
            continue
        feats_t = {
            "g0": g.get(t), "g1": g.get(t - pd.DateOffset(months=1)),
            "g2": g.get(t - pd.DateOffset(months=2)),
            "g12": g.loc[t - pd.DateOffset(months=11):t].mean(),
            "mom3": ly.get(t) - ly.get(t - pd.DateOffset(months=3)),
            "sent_yoy": lsent.get(t) - lsent.get(t - pd.DateOffset(months=12)),
            "sent_lvl": lsent.get(t),
        }
        for h in range(1, HORIZON + 1):
            d = t + pd.DateOffset(months=h)
            r = dict(feats_t, h=h, month=d.month, cat=cat_code, t=t, d=d)
            # last year's growth for the same target month captures shifting holiday timing
            r["g_ly_target"] = g.get(d - pd.DateOffset(years=1))
            if with_target:
                if d not in ly.index:
                    continue
                r["y"] = ly.loc[d] - ly.loc[d - pd.DateOffset(years=1)]
            rows.append(r)
    return pd.DataFrame(rows)


FEATURES = ["g0", "g1", "g2", "g12", "mom3", "sent_yoy", "sent_lvl", "h", "month", "g_ly_target"]


def fc_xgb_global(panel: pd.DataFrame, cats: list[str], origin: pd.Timestamp) -> dict:
    """One XGBoost model pooled across all categories (category one-hot as a feature)."""
    lsent = np.log(panel["sentiment"])
    train, pred, base = [], [], {}
    for i, c in enumerate(cats):
        ly = _prep(panel[c].loc[:origin])
        hist = [t for t in ly.index if t + pd.DateOffset(months=1) <= origin]
        tr = _xgb_rows(ly, lsent, i, hist, with_target=True)
        train.append(tr[tr["d"] <= origin])           # targets must be observed by the origin
        pred.append(_xgb_rows(ly, lsent, i, [origin], with_target=False))
        base[c] = ly
    train = pd.concat(train, ignore_index=True).dropna(subset=["y"])
    pred = pd.concat(pred, ignore_index=True)
    X = pd.get_dummies(train[FEATURES + ["cat"]], columns=["cat"], dtype=float)
    Xp = pd.get_dummies(pred[FEATURES + ["cat"]], columns=["cat"], dtype=float).reindex(columns=X.columns, fill_value=0)
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, train["y"])
    pred["g_hat"] = model.predict(Xp)
    out = {}
    for i, c in enumerate(cats):
        p = pred[pred["cat"] == i].sort_values("h")
        ly_raw = np.log(panel[c].dropna())  # use the real (uncleaned) last-year value as the base
        vals = [np.exp(ly_raw.loc[d - pd.DateOffset(years=1)] + gh) for d, gh in zip(p["d"], p["g_hat"])]
        out[c] = pd.DataFrame({"mean": vals}, index=future_index(origin))
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return out, importance

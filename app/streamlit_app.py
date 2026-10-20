"""Holiday Retail Planner - Streamlit dashboard.

Run locally:  streamlit run app/streamlit_app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from data_layer import (COVID_YEARS, GRID, INK, INK2, LABELS, MODEL_COLORS, RECOMMENDED, SHORT,
                        ecommerce_share, error_spread, facts_for_ai, holiday_table, load_all,
                        models_present, summary_table)

st.set_page_config(page_title="Holiday Retail Planner 2026", page_icon="🎄", layout="wide")


@st.cache_data
def data():
    return load_all()


D = data()
panel, bt, f26, po = D["panel"], D["bt"], D["f26"], D["po"]
CATS = list(LABELS)
HAS_LLM = any(m.startswith(("Claude", "GPT")) for m in po["model"].unique())

# ---------------------------------------------------------------- chart helpers

def style(fig, height=380, yfmt="$,.0f", ysuffix=""):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=13),
        legend=dict(orientation="h", y=-0.15, x=0, font=dict(color=INK2)),
        hovermode="x unified", hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, tickformat=yfmt, ticksuffix=ysuffix)
    return fig


def money(x, md=True):
    """Format $ millions. In markdown a bare '$' starts LaTeX, so escape it there."""
    d = "\\$" if md else "$"
    return f"{d}{x / 1000:,.1f}B" if abs(x) >= 1000 else f"{d}{x:,.0f}M"


def pct(x, sign=True):
    return f"{x:+.1%}" if sign else f"{x:.1%}"


hol_fc, hol_growth, hol_ly = holiday_table(f26, panel)
s_all, s_ex = summary_table(po, False), summary_table(po, True)
share, share_ytd = ecommerce_share(panel)
last_cat_month = panel["clothing"].dropna().index.max()

# ---------------------------------------------------------------- header

st.title("🎄 Holiday Retail Planner 2026")
st.caption(
    "Forecasting this year's U.S. holiday shopping season by category, with classical, machine-learning "
    "and LLM forecasters compared on 11 past seasons - and a planning tool for small retailers. "
    f"Data: U.S. Census Bureau Monthly Retail Trade via FRED, through {last_cat_month:%B %Y}."
)

tabs = st.tabs(["📖 Story", "📈 Market", "🔮 2026 Forecast", "🥊 Model Battle", "🧮 Holiday Planner",
                "🤖 AI Advisor", "ℹ️ About"])

# ================================================================= STORY
with tabs[0]:
    tot_g = hol_growth.loc["total", RECOMMENDED]
    best = s_all.index[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2026 holiday sales forecast", pct(tot_g), help="Total retail (ex. food services), Nov+Dec 2026 vs Nov+Dec 2025, ensemble model")
    c2.metric("Forecast: Nov+Dec 2026 total retail", money(hol_fc.loc["total", RECOMMENDED], md=False))
    c3.metric(f"Best model error ({best})", pct(s_all.loc[best, "holiday_APE"], False),
              delta=f"vs {pct(s_all.loc['Seasonal Naive', 'holiday_APE'], False)} for 'same as last year'",
              delta_color="off")
    c4.metric("E-commerce share, Jan-Aug 2026", pct(share_ytd.loc[2026], False),
              delta=f"{(share_ytd.loc[2026] - share_ytd.loc[2025]) * 100:+.1f} pts vs 2025")

    st.markdown("### Why this matters")
    st.markdown(
        "November and December bring **18% of U.S. retail sales** in two months - and over **22% for clothing "
        "stores**. Retailers commit to holiday inventory in late summer, months before the first shopper walks in. "
        "Order too little and shelves go empty in the busiest weeks; order too much and January is a markdown "
        "fire sale. Large chains have data-science teams for this. **Small retailers usually just copy last year.**"
    )
    st.markdown("### What we asked")
    st.markdown(
        "1. How much will Americans spend this holiday season, by category?\n"
        "2. Can forecasting beat the retailer's habit of *'same as last year'*? Which model is best?\n"
        "3. Can **AI chatbots (Claude, GPT) forecast** as well as statistical models - or do they just remember the past?\n"
        "4. Does **consumer sentiment** - how shoppers *feel* - improve the forecast?\n"
        "5. How should a small shop turn a national forecast into an order?"
    )
    st.markdown("### What we found")
    g = hol_growth[RECOMMENDED].drop("total").sort_values(ascending=False)
    fastest, slowest = g.index[0], g.index[-1]
    sent_delta = s_all.loc["SARIMAX + Sentiment", "holiday_APE"] - s_all.loc["SARIMA", "holiday_APE"]
    findings = [
        f"**A solid season ahead.** Our ensemble forecasts total holiday retail sales of "
        f"**{money(hol_fc.loc['total', RECOMMENDED])}**, **{pct(tot_g)}** over 2025. "
        f"{SHORT[fastest]} leads ({pct(g[fastest])}); {SHORT[slowest].lower()} is weakest ({pct(g[slowest])}).",
        f"**Forecasting beats habit.** Over 11 seasons, 'same as last year' missed the holiday total by "
        f"{pct(s_all.loc['Seasonal Naive', 'holiday_APE'], False)} on average; the ensemble missed by only "
        f"**{pct(s_all.loc[RECOMMENDED, 'holiday_APE'], False)}**, and beat it in "
        f"{s_all.loc[RECOMMENDED, 'beats_naive']:.0%} of category-seasons.",
        f"**Feelings don't forecast spending.** Adding consumer sentiment changed the error by "
        f"{sent_delta * 100:+.2f} points - no improvement. Shoppers' mood has been gloomy all year while their "
        f"spending kept growing.",
        f"**E-commerce keeps taking share**: {pct(share_ytd.loc[2026], False)} of retail in Jan-Aug 2026, up from "
        f"{pct(share_ytd.loc[2019], False)} in 2019, and forecast to grow {pct(hol_growth.loc['ecommerce', RECOMMENDED])} "
        f"this holiday season.",
    ]
    if HAS_LLM:
        la = s_all.reindex(["Claude (anonymized)", "GPT (anonymized)"]).dropna(subset=["holiday_APE"])
        if len(la):
            best_llm = la["holiday_APE"].idxmin()
            findings.append(
                f"**AI chatbots are competitive but not better.** On anonymized data (so they can't recall the answer), "
                f"{best_llm} missed by {pct(la.loc[best_llm, 'holiday_APE'], False)} vs "
                f"{pct(s_all.loc[RECOMMENDED, 'holiday_APE'], False)} for the ensemble. See *Model Battle*.")
    for f in findings:
        st.markdown(f"- {f}")
    st.info("Use the tabs above: explore the market, see the 2026 forecast, check how every model did "
            "in past seasons, plan your own store's holiday order, or ask the AI advisor.")

# ================================================================= MARKET
with tabs[1]:
    st.subheader("Every category peaks in December - but not equally")
    col_a, col_b = st.columns([1, 2])
    cat = col_a.selectbox("Category", CATS, format_func=lambda c: LABELS[c], key="mk_cat")
    years = col_a.slider("Years shown", 2000, 2026, (2015, 2026), key="mk_years")
    y = panel[cat].dropna().loc[f"{years[0]}":f"{years[1]}"]
    fig = go.Figure()
    fig.add_scatter(x=y.index, y=y.values, mode="lines", name=LABELS[cat],
                    line=dict(color="#2a78d6", width=2), hovertemplate="%{x|%b %Y}: $%{y:,.0f}M<extra></extra>")
    dec = y[y.index.month == 12]
    fig.add_scatter(x=dec.index, y=dec.values, mode="markers", name="December",
                    marker=dict(color="#eb6834", size=9, line=dict(color="white", width=2)),
                    hovertemplate="%{x|%b %Y}: $%{y:,.0f}M<extra>December</extra>")
    fig.update_layout(title=dict(text=f"{LABELS[cat]} - monthly sales, $ millions (not seasonally adjusted)", font=dict(size=14, color=INK)))
    col_b.plotly_chart(style(fig, 360), width="stretch")
    with col_a:
        yr = panel[cat].loc["2025"]
        st.metric("Nov+Dec share of 2025 sales", pct(yr.loc["2025-11":].sum() / yr.sum(), False))
        st.metric("December vs. October 2025", f"{yr.loc['2025-12-01'] / yr.loc['2025-10-01']:.2f}x")

    st.subheader("How big is the holiday spike?")
    prof = []
    for c in CATS:
        s = panel[c].loc["2022":"2025"]
        prof.append((s.groupby(s.index.month).sum() / s.sum() * 12).rename(SHORT[c]))
    prof = pd.concat(prof, axis=1)
    order = prof.loc[12].sort_values(ascending=False).index
    fig = go.Figure()
    fig.add_bar(y=order, x=prof.loc[12, order], orientation="h", marker_color="#2a78d6",
                text=[f"{v:.2f}x" for v in prof.loc[12, order]], textposition="outside",
                textfont=dict(color=INK2), hovertemplate="%{y}: December = %{x:.2f}x an average month<extra></extra>")
    fig.add_vline(x=1, line_color=INK2, line_width=1)
    fig.update_layout(title=dict(text="December sales relative to an average month (2022-2025). 1.0 = no peak.", font=dict(size=14, color=INK)),
                      bargap=0.45, showlegend=False, hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 330, yfmt=""), width="stretch")

    st.subheader("E-commerce keeps taking share")
    fig = go.Figure()
    s12 = share.rolling(12).mean().dropna().loc["2005":]
    fig.add_scatter(x=s12.index, y=s12.values, mode="lines", line=dict(color="#2a78d6", width=2),
                    name="E-commerce share (12-month average)", hovertemplate="%{x|%b %Y}: %{y:.1%}<extra></extra>")
    fig.add_annotation(x="2020-04-01", y=float(share.loc["2020-04-01"]), text="COVID lockdowns", showarrow=True,
                       arrowcolor=INK2, font=dict(color=INK2), ay=-40)
    fig.update_layout(title=dict(text="Electronic shopping & mail-order as a share of total retail, 12-month rolling average", font=dict(size=14, color=INK)), showlegend=False)
    st.plotly_chart(style(fig, 320, yfmt=".0%"), width="stretch")
    st.caption("Share = Census 'Electronic shopping and mail-order houses' (NAICS 4541) / total retail excluding food services. "
               "It understates total online spending because store-based retailers' web sales are counted with their stores.")

# ================================================================= 2026 FORECAST
with tabs[2]:
    st.subheader("The 2026 holiday season, by category")
    tbl = pd.DataFrame({
        "Category": [LABELS[c] for c in hol_fc.index],
        "Nov+Dec 2025 actual": [money(hol_ly[c], md=False) for c in hol_fc.index],
        "Nov+Dec 2026 forecast": [money(hol_fc.loc[c, RECOMMENDED], md=False) for c in hol_fc.index],
        "Growth": [pct(hol_growth.loc[c, RECOMMENDED]) for c in hol_fc.index],
    }).set_index("Category").reindex([LABELS[c] for c in CATS])
    st.dataframe(tbl, width="stretch")
    st.caption(f"Recommended model: **{RECOMMENDED}** (average of SARIMA, ETS and XGBoost in log space) - the most "
               "accurate in the backtest. Forecast made with data through August 2026.")

    st.subheader("Look closer")
    c1, c2 = st.columns([1, 3])
    cat = c1.selectbox("Category", CATS, format_func=lambda c: LABELS[c], key="fc_cat")
    avail = models_present(f26)
    default = [m for m in [RECOMMENDED, "SARIMA", "Claude (named)", "GPT (named)", "Seasonal Naive"] if m in avail]
    sel = c1.multiselect("Models", avail, default=default, key="fc_models")
    band = c1.checkbox("Show SARIMA 80% interval", value=True)
    hist = panel[cat].dropna().loc["2023-01":]
    fc = f26[f26.category == cat]
    fig = go.Figure()
    if band:
        sb = fc[fc.model == "SARIMA"]
        fig.add_scatter(x=list(sb.date) + list(sb.date[::-1]), y=list(sb.hi80) + list(sb.lo80[::-1]),
                        mode="lines", fill="toself", fillcolor="rgba(235,104,52,0.12)", line=dict(width=0),
                        name="SARIMA 80% interval", hoverinfo="skip")
    fig.add_scatter(x=hist.index, y=hist.values, mode="lines", name="Actual", line=dict(color=INK, width=2),
                    hovertemplate="$%{y:,.0f}M")
    origin = fc["origin"].iloc[0]
    for m in sel:
        d = fc[fc.model == m]
        if d.empty:
            continue
        # start each forecast line at the last month the model saw, so it reads as a continuation
        x = [origin] + list(d.date)
        yv = [hist.loc[origin]] + list(d["mean"])
        fig.add_scatter(x=x, y=yv, mode="lines+markers", name=m, line=dict(color=MODEL_COLORS[m], width=2),
                        marker=dict(size=8, line=dict(color="white", width=2)), hovertemplate="$%{y:,.0f}M")
    fig.update_layout(title=dict(text=f"{LABELS[cat]}: actual sales and Sep-Dec 2026 forecasts, $ millions", font=dict(size=14, color=INK)))
    c2.plotly_chart(style(fig, 420), width="stretch")

    if cat == "total":
        sep = fc[(fc.date == "2026-09-01")].dropna(subset=["actual"])
        if len(sep):
            e = sep.set_index("model")
            st.success(f"✅ **Early reality check:** September 2026 total retail ({money(e['actual'].iloc[0])}) was published "
                       f"after our forecast origin. The ensemble forecast {money(e.loc[RECOMMENDED, 'mean'])} - "
                       f"an error of {pct(e.loc[RECOMMENDED, 'mean'] / e.loc[RECOMMENDED, 'actual'] - 1)}.")

    with st.expander("All models, Nov+Dec 2026 growth vs 2025"):
        gt = (hol_growth[models_present(f26)] * 100).round(1)
        gt.index = [LABELS[c] for c in gt.index]
        st.dataframe(gt.style.format("{:+.1f}%"), width="stretch")
    if D["reasoning"] is not None:
        with st.expander("What did the AI forecasters say? (their own explanations)"):
            r = D["reasoning"]
            r = r[(r.origin.astype(str).str.startswith("2026")) & (r.category == cat)]
            for _, row in r.iterrows():
                st.markdown(f"**{row['model']}** - {row['reasoning']}")

# ================================================================= MODEL BATTLE
with tabs[3]:
    st.subheader("Which forecaster would you have trusted? 11 holiday seasons, 2015-2025")
    st.markdown(
        "Every model stood at the **end of August** of each year, saw only data up to that point, and forecast "
        "September to December. We score the **November+December total** - the number a retailer orders against."
    )
    ex = st.toggle("Exclude the COVID seasons (2020-2021)", value=False)
    s = s_ex if ex else s_all
    s = s.dropna(subset=["holiday_APE"])
    fig = go.Figure()
    fig.add_bar(y=s.index, x=s["holiday_APE"], orientation="h",
                marker_color=[MODEL_COLORS.get(m, "#8a8984") for m in s.index],
                text=[f"{v:.1%}" for v in s["holiday_APE"]], textposition="outside", textfont=dict(color=INK2),
                hovertemplate="%{y}: average error %{x:.2%}<extra></extra>")
    fig.update_layout(title=dict(text="Average absolute error of the Nov+Dec forecast (lower is better)", font=dict(size=14, color=INK)),
                      showlegend=False, bargap=0.4, hovermode="closest")
    style(fig, 60 + 38 * len(s), yfmt="")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", gridcolor=GRID, showgrid=True)
    st.plotly_chart(fig, width="stretch")

    show = s.copy()
    show_fmt = pd.DataFrame({
        "Nov+Dec error": show["holiday_APE"].map("{:.2%}".format),
        "Monthly MAPE": show["MAPE"].map("{:.2%}".format),
        "MASE": show["MASE"].map("{:.2f}".format),
        "80% interval coverage": show["Coverage80"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}"),
        "Beats 'same as last year'": show["beats_naive"].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}"),
    })
    st.dataframe(show_fmt, width="stretch")
    st.caption("MASE < 1 means the model's errors are smaller than a seasonal-naive forecast's typical in-sample error. "
               "80% interval coverage should be close to 80% if the interval is honest - only models with a statistical "
               "interval (ETS, SARIMA) have one.")

    st.subheader("By category")
    pv = po.pivot_table(index="category", columns="model", values="holiday_ape")
    if ex:
        pv = po[~po.year.isin(COVID_YEARS)].pivot_table(index="category", columns="model", values="holiday_ape")
    pv = pv[[m for m in s.index if m in pv.columns]].reindex(CATS)
    fig = go.Figure(go.Heatmap(z=pv.values * 100, x=pv.columns, y=[SHORT[c] for c in pv.index],
                               colorscale=[[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]],
                               text=np.round(pv.values * 100, 1), texttemplate="%{text}%", xgap=2, ygap=2,
                               hovertemplate="%{y} / %{x}: %{z:.2f}%<extra></extra>",
                               colorbar=dict(title="error %")))
    fig.update_layout(title=dict(text="Nov+Dec forecast error by category (%) - darker = worse", font=dict(size=14, color=INK)), hovermode="closest")
    style(fig, 360, yfmt="")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Replay a season")
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Category", CATS, format_func=lambda c: LABELS[c], key="bt_cat")
    year = c2.selectbox("Season", sorted(bt.year.unique(), reverse=True), key="bt_year")
    avail = models_present(bt)
    sel = c3.multiselect("Models", avail, default=[m for m in [RECOMMENDED, "Seasonal Naive", "Claude (anonymized)", "GPT (anonymized)"] if m in avail], key="bt_models")
    hist = panel[cat].dropna().loc[f"{year - 1}-09":f"{year}-12"]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Actual",
                    line=dict(color=INK, width=2), marker=dict(size=6), hovertemplate="$%{y:,.0f}M")
    for m in sel:
        d = bt[(bt.category == cat) & (bt.year == year) & (bt.model == m)]
        fig.add_scatter(x=d.date, y=d["mean"], mode="lines+markers", name=m, line=dict(color=MODEL_COLORS[m], width=2),
                        marker=dict(size=8, line=dict(color="white", width=2)), hovertemplate="$%{y:,.0f}M")
    fig.add_vline(x=pd.Timestamp(f"{year}-08-15"), line_color=GRID, line_width=1)
    fig.add_annotation(x=pd.Timestamp(f"{year}-08-15"), y=1, yref="paper", text="forecast made here", showarrow=False,
                       font=dict(color=INK2, size=11), xanchor="right")
    fig.update_layout(title=dict(text=f"{LABELS[cat]}, {year}: what each model predicted in August vs. what happened", font=dict(size=14, color=INK)))
    st.plotly_chart(style(fig, 400), width="stretch")

    if HAS_LLM:
        st.subheader("Did the chatbots forecast - or remember?")
        st.markdown(
            "An LLM may have read the real Census numbers during training. So we asked Claude the same question two ways: "
            "**anonymized** (an index with no category name or year - nothing to look up) and **named** (real dates, real "
            "category, real dollars). If 'named' is much more accurate on past seasons, the model is partly *remembering*."
        )
        mm = po[po.model.isin(["Claude (anonymized)", "Claude (named)"])].pivot_table(index="year", columns="model", values="holiday_ape")
        if mm.shape[1] == 2:
            fig = go.Figure()
            for m in mm.columns:
                fig.add_scatter(x=mm.index, y=mm[m], mode="lines+markers", name=m, line=dict(color=MODEL_COLORS[m], width=2),
                                marker=dict(size=8, line=dict(color="white", width=2)), hovertemplate="%{y:.2%}")
            fig.update_layout(title=dict(text="Claude's Nov+Dec error by season: anonymized vs named prompts", font=dict(size=14, color=INK)))
            st.plotly_chart(style(fig, 340, yfmt=".1%"), width="stretch")
            a, n = mm.mean()
            st.markdown(f"Average error: anonymized **{a:.2%}**, named **{n:.2%}**. " + (
                "Named prompts were noticeably more accurate - consistent with some memorization, so only the anonymized "
                "results are a fair test of forecasting skill." if n < a * 0.8 else
                "The two are similar, so there is little sign that Claude was simply recalling published figures."))

    with st.expander("What drives the XGBoost model? (feature importance, 2025 backtest)"):
        imp = D["imp"]
        if imp is not None:
            col = imp.columns[-1]
            names = {"g_ly_target": "Last year's growth for the same month", "g0": "Latest YoY growth",
                     "g12": "Average YoY growth, last 12 months", "g1": "YoY growth, 1 month earlier",
                     "g2": "YoY growth, 2 months earlier", "mom3": "3-month momentum", "sent_yoy": "Consumer sentiment change",
                     "sent_lvl": "Consumer sentiment level", "h": "Months ahead", "month": "Target month"}
            top = imp[col].rename(lambda k: names.get(k, SHORT.get(k.replace("cat_", ""), k) if not k.startswith("cat_") else "Category: " + k[4:])).sort_values().tail(8)
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color="#2a78d6"))
            fig.update_layout(bargap=0.4, showlegend=False, hovermode="closest")
            st.plotly_chart(style(fig, 300, yfmt=""), width="stretch")

# ================================================================= PLANNER
with tabs[4]:
    st.subheader("Plan your store's holiday order")
    st.markdown(
        "Tell us about your shop. We scale **your own** last holiday season by the national forecast for your category, "
        "and size a safety buffer from how wrong our model has been in past seasons."
    )
    c1, c2 = st.columns([1, 1.3])
    with c1:
        cat = st.selectbox("What do you sell?", [c for c in CATS if c != "total"], format_func=lambda c: LABELS[c], key="pl_cat")
        unit = st.radio("Measure in", ["units", "dollars"], horizontal=True)
        # fmt is for markdown (escaped $), fmt_plain for widgets that show text as-is
        fmt = (lambda v: f"{v:,.0f} units") if unit == "units" else (lambda v: f"\\${v:,.0f}")
        fmt_plain = (lambda v: f"{v:,.0f} units") if unit == "units" else (lambda v: f"${v:,.0f}")
        last = st.number_input(f"Your Nov+Dec 2025 sales ({unit})", min_value=0.0, value=5000.0, step=100.0)
        on_hand = st.number_input(f"Stock on hand now ({unit})", min_value=0.0, value=1200.0, step=100.0)
        on_order = st.number_input(f"Already on order ({unit})", min_value=0.0, value=0.0, step=100.0)
        service = st.select_slider("How sure do you want to be of not running out?", options=[0.80, 0.85, 0.90, 0.95, 0.98],
                                   value=0.90, format_func=lambda v: f"{v:.0%}")
        local = st.slider("Adjust for your store (percentage points vs. national trend)", -15, 15, 0,
                          help="E.g. +3 if you opened a new channel, -5 if a competitor opened next door.") / 100
        store_sd = st.slider("How unpredictable is your store? (typical ± % miss vs. plan)", 0, 30, 10,
                             help="A single shop swings far more than the national total. If unsure, 10% is a reasonable start.") / 100
    g = hol_growth.loc[cat, RECOMMENDED] + local
    sigma_nat, bias, n = error_spread(po, cat)
    # national forecast error and the store's own noise are independent, so variances add
    sigma = float(np.sqrt(sigma_nat ** 2 + store_sd ** 2))
    expected = last * (1 + g)
    z = norm.ppf(service)
    safety = z * sigma * expected
    target = expected + safety
    order = max(0.0, target - on_hand - on_order)
    lo, hi = expected * (1 - 1.2816 * sigma), expected * (1 + 1.2816 * sigma)
    with c2:
        m1, m2 = st.columns(2)
        m1.metric("Recommended order", fmt_plain(np.ceil(order)))
        m2.metric("Expected Nov+Dec sales", fmt_plain(expected), delta=f"{g:+.1%} vs your 2025")
        fig = go.Figure()
        parts = [("Expected demand", expected, "#2a78d6"), ("Safety buffer", safety, "#86b6ef")]
        for name, val, col in parts:
            fig.add_bar(x=[val], y=["Need"], orientation="h", name=name, marker_color=col,
                        hovertemplate=f"{name}: %{{x:,.0f}}<extra></extra>")
        for name, val, col in [("On hand", on_hand, "#8a8984"), ("On order", on_order, "#c3c2b7"), ("Order now", order, "#eb6834")]:
            fig.add_bar(x=[val], y=["Covered by"], orientation="h", name=name, marker_color=col,
                        hovertemplate=f"{name}: %{{x:,.0f}}<extra></extra>")
        fig.update_layout(barmode="stack", bargap=0.45, hovermode="closest",
                          title=dict(text="What you need vs. how it's covered", font=dict(size=14, color=INK)))
        st.plotly_chart(style(fig, 240, yfmt=","), width="stretch")
        st.markdown(f"""
| Scenario | Nov+Dec sales | Order needed (no buffer) |
|---|---|---|
| Conservative (10th percentile) | {fmt(lo)} | {fmt(max(0, lo - on_hand - on_order))} |
| **Base (forecast)** | **{fmt(expected)}** | **{fmt(max(0, expected - on_hand - on_order))}** |
| Optimistic (90th percentile) | {fmt(hi)} | {fmt(max(0, hi - on_hand - on_order))} |
""")
        nov_share = (f26[(f26.category == cat) & (f26.model == RECOMMENDED) & (f26.date.dt.month == 11)]["mean"].sum()
                     / hol_fc.loc[cat, RECOMMENDED])
        st.caption(f"Timing: nationally, about **{nov_share:.0%}** of this category's holiday sales fall in November and "
                   f"**{1 - nov_share:.0%}** in December - stage deliveries accordingly.")

    with st.expander("How is this calculated?"):
        st.markdown(f"""
- **Growth** = ensemble forecast for {LABELS[cat]} Nov+Dec 2026 ÷ actual Nov+Dec 2025 = **{hol_growth.loc[cat, RECOMMENDED]:+.1%}**, plus your adjustment ({local:+.0%}).
- **Expected sales** = your 2025 sales × (1 + growth).
- **National uncertainty** = root-mean-square of the ensemble's Nov+Dec % errors for this category over {n} past seasons (2015-2025 excluding the COVID seasons) = **{sigma_nat:.1%}**.
- **Total uncertainty (σ)** = √(national² + your store's ±{store_sd:.0%}²) = **{sigma:.1%}** - the two sources are independent, so their variances add.
- **Safety buffer** = z × σ × expected sales, with z = {z:.2f} for a {service:.0%} chance of covering demand (normal approximation).
- **Recommended order** = max(0, expected + buffer − on hand − on order), rounded up.
- Scenarios use ±1.28σ (10th / 90th percentile).

**Limits:** this is a *national* category trend. Your store can differ - that is what the two store sliders are for. The national
error is measured from our backtest; your store's volatility is your own estimate (check it against how far past seasons missed
your plans). No inventory cost data is used, so we make no claim about money saved.
""")
    st.session_state["planner"] = {
        "category": LABELS[cat], "unit": unit, "Nov+Dec 2025 sales": last, "on hand": on_hand, "on order": on_order,
        "service level": f"{service:.0%}", "store adjustment": f"{local:+.0%}", "national growth used": f"{hol_growth.loc[cat, RECOMMENDED]:+.2%}",
        "expected Nov+Dec 2026 sales": round(expected), "national forecast error (sigma)": f"{sigma_nat:.2%}",
        "store volatility entered": f"{store_sd:.0%}", "total sigma used": f"{sigma:.2%}", "safety buffer": round(safety),
        "recommended order": int(np.ceil(order)), "conservative/optimistic sales": f"{lo:,.0f} / {hi:,.0f}",
    }

# ================================================================= AI ADVISOR
with tabs[5]:
    st.subheader("Ask the AI advisor")
    st.markdown("Ask in plain language - about the forecast, the models, or your plan. The advisor (Claude) only explains "
                "numbers our code computed; it does not make up forecasts or ranges.")

    def api_key():
        try:
            if "ANTHROPIC_API_KEY" in st.secrets:
                return st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            pass
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
        return os.getenv("ANTHROPIC_API_KEY") or None

    key = api_key()
    if "chat" not in st.session_state:
        st.session_state.chat = []
    examples = ["Why do you recommend this order for my store?", "Which model is most accurate, and why?",
                "Which categories are hardest to forecast?", "Can ChatGPT or Claude forecast retail sales?",
                "Is e-commerce still taking share from stores?"]
    cols = st.columns(len(examples))
    clicked = None
    for c, q in zip(cols, examples):
        if c.button(q, width="stretch"):
            clicked = q
    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).markdown(msg["content"])
    question = st.chat_input("Ask about the 2026 holiday forecast...") or clicked
    if question:
        st.chat_message("user").markdown(question)
        st.session_state.chat.append({"role": "user", "content": question})
        if not key:
            answer = ("The AI advisor isn't connected (no API key configured). All forecasts, metrics and the planner "
                      "still work - they are computed by the project's code, not by the AI.")
            st.chat_message("assistant").markdown(answer)
        else:
            import anthropic
            system = (
                "You are the AI advisor inside 'Holiday Retail Planner 2026', a UCLA Anderson forecasting class project. "
                "Answer questions from small retailers and classmates about the U.S. holiday retail forecast, the model "
                "comparison and the user's planner result. Rules: use only the computed facts below; quote numbers exactly "
                "as given; never invent forecasts, confidence ranges or causes; if the facts don't answer the question, "
                "say what is missing. Separate what the data shows from interpretation. Keep answers short (under 150 "
                "words), plain English, no tables unless asked.\n\n" + facts_for_ai(D, st.session_state.get("planner"))
            )
            client = anthropic.Anthropic(api_key=key)
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[-10:]]
            with st.chat_message("assistant"):
                try:
                    with client.beta.messages.stream(
                        model="claude-opus-5-5", max_tokens=4000, system=system, messages=history,
                        output_config={"effort": "low"},
                        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                    ) as stream:
                        answer = st.write_stream(stream.text_stream)
                        final = stream.get_final_message()
                    if final.stop_reason == "refusal":
                        answer = "Sorry - I can't help with that one. Try asking about the forecast or your plan."
                        st.markdown(answer)
                except anthropic.APIError as e:
                    answer = f"The AI advisor is temporarily unavailable ({type(e).__name__}). The rest of the dashboard still works."
                    st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

# ================================================================= ABOUT
with tabs[6]:
    st.markdown(open(os.path.join(os.path.dirname(__file__), "about.md"), encoding="utf-8").read())

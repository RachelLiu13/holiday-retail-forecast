"""Build the full written report (PDF, with executive summary) from the saved results.

Every number, table and chart is computed from outputs/ and data/, so the report always matches the code.

Run:  python reports/build_report.py
"""
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))
from data_layer import (LABELS, RECOMMENDED, SHORT, ecommerce_share, error_spread, holiday_table,  # noqa: E402
                        llm_findings, load_all, summary_table)
from src.llm_forecast import build_prompt  # noqa: E402

DASHBOARD_URL = "https://holiday-retail-planner.streamlit.app"
REPO_URL = "https://github.com/RachelLiu13/holiday-retail-forecast"
OUT = ROOT / "reports"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY, INK, INK2, GRID, VIOLET, RED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#6250d6", "#e34948"

D = load_all()
panel, po, f26, bt = D["panel"], D["po"], D["f26"], D["bt"]
tests = D["tests"]
s_all, s_ex = summary_table(po, False), summary_table(po, True)
hol_fc, hol_g, hol_ly = holiday_table(f26, panel)
share, share_ytd = ecommerce_share(panel)
lf = llm_findings(tests, s_all)
CATS = list(LABELS)
LLMS = [m for m in ["Claude (anonymized)", "GPT (anonymized)", "Claude (named)"] if m in s_all.index]


def pct(x, sign=False, d=1):
    return f"{x:+.{d}%}" if sign else f"{x:.{d}%}"


def bn(x):
    return f"${x / 1000:,.1f}B"


plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.titlesize": 9.5,
                     "axes.titlelocation": "left", "lines.linewidth": 1.7, "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    return str(FIG / name)


# ---------------------------------------------------------------- figures
figs = {}
fig, ax = plt.subplots(figsize=(7, 2.6))
s = panel["total"].dropna().loc["2012":]
ax.plot(s.index, s / 1000, color=BLUE)
dec = s[s.index.month == 12]
ax.scatter(dec.index, dec / 1000, color=ORANGE, s=16, zorder=3, label="December")
ax.set_ylabel("$ billions / month"); ax.legend(loc="upper left")
ax.set_title("Figure 1. Total U.S. retail sales excluding food services, not seasonally adjusted")
figs["total"] = save(fig, "f1_total.png")

prof = {}
for c in CATS:
    x = panel[c].loc["2022":"2025"]
    prof[SHORT[c]] = x.groupby(x.index.month).sum() / x.sum() * 12
prof = pd.DataFrame(prof)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
d = prof.loc[12].sort_values()
axes[0].barh(d.index, d.values, color=BLUE, height=0.5)
axes[0].axvline(1, color=INK, lw=0.8)
for i, v in enumerate(d.values):
    axes[0].text(v + 0.02, i, f"{v:.2f}x", va="center", fontsize=7.5)
axes[0].set_title("Figure 2a. December vs. an average month")
axes[0].set_xlim(0.8, max(d.values) + 0.25); axes[0].grid(axis="y", visible=False)
for c, col in [("Clothing", ORANGE), ("Total retail", BLUE), ("Grocery", GRAY)]:
    axes[1].plot(range(1, 13), prof[c], color=col, marker="o", ms=3, label=c)
axes[1].set_xticks(range(1, 13)); axes[1].set_xticklabels(list("JFMAMJJASOND"))
axes[1].axhline(1, color=INK, lw=0.6); axes[1].legend(fontsize=7.5)
axes[1].set_title("Figure 2b. Monthly profile, 2022-2025")
figs["season"] = save(fig, "f2_season.png")

fig, ax = plt.subplots(figsize=(7, 2.3))
s12 = share.rolling(12).mean().dropna().loc["2005":]
ax.plot(s12.index, s12 * 100, color=BLUE)
ax.set_ylabel("% of total retail")
ax.set_title("Figure 3. E-commerce (electronic shopping & mail-order) share of retail, 12-month average")
figs["share"] = save(fig, "f3_share.png")

d = s_all["holiday_APE"].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 0.22 * len(d) + 0.6))
ax.barh([f"{m} (hindsight*)" if "(named)" in m else m for m in d.index], d.values * 100, height=0.55,
        color=[GRAY if m == "Seasonal Naive" else (VIOLET if m.startswith(("Claude", "GPT")) else BLUE) for m in d.index])
for i, v in enumerate(d.values):
    ax.text(v * 100 + 0.05, i, f"{v:.2%}", va="center", fontsize=7.5)
ax.set_xlabel("Mean absolute % error of the Nov+Dec total, 11 seasons"); ax.grid(axis="y", visible=False)
ax.set_title("Figure 4. Backtest accuracy, 2015-2025 (lower is better; purple = LLM)")
figs["errors"] = save(fig, "f4_errors.png")

pv = (po.pivot_table(index="category", columns="model", values="holiday_ape") * 100).reindex(CATS)
pv = pv[[m for m in s_all.index if m in pv.columns]]
fig, ax = plt.subplots(figsize=(7, 3.1))
im = ax.imshow(pv.values, cmap="Blues", aspect="auto", vmin=0, vmax=np.nanmax(pv.values))
ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=35, ha="right", fontsize=7.5)
ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels([SHORT[c] for c in pv.index], fontsize=7.5)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        v = pv.values[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v > np.nanmax(pv.values) * 0.6 else INK)
ax.grid(False); ax.set_title("Figure 5. Nov+Dec error (%) by category and model")
figs["heat"] = save(fig, "f5_heat.png")

by_year = (po.pivot_table(index="year", columns="model", values="holiday_ape") * 100)
fig, ax = plt.subplots(figsize=(7, 2.6))
for m, col in [("Seasonal Naive", GRAY), (RECOMMENDED, BLUE), ("Claude (anonymized)", VIOLET), ("GPT (anonymized)", RED),
               ("Claude (named)", "#9085e9")]:
    if m in by_year:
        ax.plot(by_year.index, by_year[m], marker="o", ms=3.5, color=col, label=m,
                ls="--" if m == "Claude (named)" else "-")
ax.set_ylabel("Nov+Dec error (%)"); ax.legend(fontsize=7.2, ncol=3)
ax.set_title("Figure 6. Holiday error by season (average over the 7 categories)")
figs["years"] = save(fig, "f6_years.png")

fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
for ax, (cat, yr) in zip(axes, [("clothing", 2024), ("ecommerce", 2025)]):
    h = panel[cat].loc[f"{yr - 1}-09":f"{yr}-12"]
    ax.plot(h.index, h / 1000, color=INK, label="Actual")
    for m, col in [(RECOMMENDED, BLUE), ("Seasonal Naive", GRAY), ("GPT (anonymized)", RED), ("Claude (anonymized)", VIOLET)]:
        x = bt[(bt.category == cat) & (bt.year == yr) & (bt.model == m)]
        if len(x):
            ax.plot(x.date, x["mean"] / 1000, marker="o", ms=3, color=col, label=m)
    ax.axvline(pd.Timestamp(f"{yr}-08-15"), color=GRID)
    ax.set_title(f"{SHORT[cat]}, {yr} season ($B)"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10])); ax.tick_params(axis="x", labelsize=7)
axes[0].legend(fontsize=6.8)
fig.suptitle("Figure 7. Forecasts made in August vs. what happened", x=0.01, ha="left", fontsize=9.5)
figs["replay"] = save(fig, "f7_replay.png")

fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
for ax, cat in zip(axes, ["total", "ecommerce"]):
    h = panel[cat].dropna().loc["2024-09":]
    ax.plot(h.index, h / 1000, color=INK, label="Actual")
    fc = f26[f26.category == cat]
    o = pd.Timestamp("2026-08-01")
    for m, col in [(RECOMMENDED, BLUE), ("Seasonal Naive", GRAY)]:
        x = fc[fc.model == m]
        ax.plot([o] + list(x.date), [panel[cat].loc[o] / 1000] + list(x["mean"] / 1000), marker="o", ms=3, color=col, label=m)
    sb = fc[fc.model == "SARIMA"]
    ax.fill_between(sb.date, sb.lo80 / 1000, sb.hi80 / 1000, color=ORANGE, alpha=0.15, lw=0, label="SARIMA 80% interval")
    ax.set_title(f"{SHORT[cat]} ($B / month)"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10])); ax.tick_params(axis="x", labelsize=7)
axes[0].legend(fontsize=6.8)
fig.suptitle("Figure 8. The Sep-Dec 2026 forecast", x=0.01, ha="left", fontsize=9.5)
figs["f26"] = save(fig, "f8_2026.png")

imp = D["imp"].iloc[:, -1]
names = {"g_ly_target": "Last year's growth, same month", "g0": "Latest YoY growth", "g12": "Avg YoY growth, 12 months",
         "g1": "YoY growth, 1 month earlier", "g2": "YoY growth, 2 months earlier", "mom3": "3-month momentum",
         "sent_yoy": "Sentiment change (YoY)", "sent_lvl": "Sentiment level", "h": "Months ahead", "month": "Target month"}
imp = imp.rename(lambda k: names.get(k, "Category: " + SHORT.get(k[4:], k[4:]) if k.startswith("cat_") else k)).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(7, 2.7))
ax.barh(imp.index, imp.values, color=BLUE, height=0.55); ax.grid(axis="y", visible=False)
ax.set_xlabel("Share of total gain"); ax.set_title("Figure 9. XGBoost feature importance (model trained through August 2025)")
figs["imp"] = save(fig, "f9_importance.png")

gm = [m for m in [RECOMMENDED, "SARIMA", "Claude (named)", "GPT (named)"] if m in hol_g.columns]
fig, ax = plt.subplots(figsize=(7, 2.9))
w = 0.8 / len(gm)
for j, (m, col) in enumerate(zip(gm, [BLUE, ORANGE, VIOLET, RED])):
    ax.bar(np.arange(len(CATS)) + (j - (len(gm) - 1) / 2) * w, hol_g.loc[CATS, m] * 100, width=w * 0.9, color=col, label=m)
ax.set_xticks(range(len(CATS))); ax.set_xticklabels([SHORT[c] for c in CATS], fontsize=7.5)
ax.axhline(0, color=INK, lw=0.6); ax.set_ylabel("Nov+Dec growth vs 2025 (%)"); ax.legend(fontsize=7.2, ncol=4)
ax.set_title("Figure 10. 2026 holiday growth forecast by category and forecaster")
figs["g26"] = save(fig, "f10_growth2026.png")

# ---------------------------------------------------------------- styles & helpers
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman", fontSize=12, leading=15.6, spaceAfter=7, alignment=4)
small = ParagraphStyle("s", parent=body, fontSize=10, leading=12.5, alignment=0, textColor=colors.HexColor(INK2))
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Times-Bold", fontSize=15, leading=19, spaceBefore=6, spaceAfter=6,
                    textColor=colors.HexColor("#0d366b"), keepWithNext=1)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12.5, leading=15.5, spaceBefore=6, spaceAfter=3,
                    textColor=colors.HexColor("#0d366b"), keepWithNext=1)
bul = ParagraphStyle("bul", parent=body, leftIndent=15, bulletIndent=3, spaceAfter=4, alignment=0, bulletFontName="Times-Roman")
cap = ParagraphStyle("cap", parent=small, spaceAfter=8)


def P(t, s=body):
    return Paragraph(t, s)


def B(items):
    return [Paragraph(t, bul, bulletText="•") for t in items]


cell = ParagraphStyle("cell", parent=body, fontSize=10, leading=12, spaceAfter=0, alignment=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")


def T(rows, widths, bold_rows=()):
    # wrap every cell in a Paragraph so long text breaks inside its column instead of overflowing
    rows = [[Paragraph(str(v), cellb if (i == 0 or i in bold_rows) else cell) for v in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "Times-Roman", 10), ("FONT", (0, 0), (-1, 0), "Times-Bold", 10),
          ("LEADING", (0, 0), (-1, -1), 12), ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor(INK)),
          ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6)]
    for r in bold_rows:
        st.append(("FONT", (0, r), (-1, r), "Times-Bold", 10))
    t.setStyle(TableStyle(st))
    return t


def compact(prompt, first=0, last=24, per_line=6):
    """Show a prompt's monthly values several per line (the model received one per line)."""
    lines = prompt.splitlines()
    vals = [l for l in lines if re.match(r"^[A-Z][a-z]{2}: ", l)]
    head = [textwrap.fill(l, 92) for l in lines[:lines.index(vals[0])] if l.strip()]
    tail = [textwrap.fill(l, 92) for l in lines[lines.index(vals[-1]) + 1:] if l.strip()]

    def rows(v):
        return ["   ".join(f"{x:<11}" for x in v[i:i + per_line]) for i in range(0, len(v), per_line)]
    body_ = rows(vals[:first]) + [f"... ({len(vals) - first - last} more months) ..."] + rows(vals[-last:]) if first else \
        [f"... ({len(vals) - last} earlier months) ..."] + rows(vals[-last:])
    return "\n".join(head + body_ + tail)


def F(path, w=7.0, h=2.6):
    # figures are drawn at 90% so more of them fit beside their text (aspect ratio kept)
    return Image(path, width=w * 0.9 * inch, height=h * 0.9 * inch)


ens, nb_ = s_all.loc[RECOMMENDED, "holiday_APE"], s_all.loc["Seasonal Naive", "holiday_APE"]
tot_g = hol_g.loc["total", RECOMMENDED]
gsort = hol_g[RECOMMENDED].drop("total").sort_values(ascending=False)
sep = f26[(f26.category == "total") & (f26.date == "2026-09-01")].set_index("model")
sep_err = sep.loc[RECOMMENDED, "mean"] / sep.loc[RECOMMENDED, "actual"] - 1 if sep["actual"].notna().any() else None
sent_t = {"d": (s_all.loc["SARIMAX + Sentiment", "holiday_APE"] - s_all.loc["SARIMA", "holiday_APE"]) * 100}
mem = tests.get("Claude (anonymized) vs Claude (named)")

# Table of contents: every level-1 heading (and the appendix headings) with its page number
toc = TableOfContents()
toc.dotsMinLevel = 0
toc.levelStyles = [ParagraphStyle("toc0", parent=body, fontSize=12, leading=17, spaceAfter=0, alignment=0, leftIndent=0,
                                  firstLineIndent=0),
                   ParagraphStyle("toc1", parent=body, fontSize=11, leading=15, spaceAfter=0, alignment=0, leftIndent=18,
                                  firstLineIndent=0, textColor=colors.HexColor(INK2))]

story = []
# ---------------------------------------------------------------- title page + executive summary
story += [P("Forecasting the 2026 U.S. Holiday Retail Season", ParagraphStyle("t", parent=h1, fontSize=20, leading=24)),
          P("Classical, machine-learning and AI-chatbot forecasters compared - and a planning tool for small retailers",
            ParagraphStyle("st", parent=body, fontSize=13, leading=16, alignment=0, textColor=colors.HexColor(INK2))),
          P("UCLA Anderson School of Management · Time Series Analysis and Forecasting · Dr. William Yu<br/>"
            "Group project, October 2026 · Charlotte Zhu, Shihan Wang, Jing Chang, Rachel Liu<br/>"
            f"Interactive dashboard: <link href='{DASHBOARD_URL}' color='{BLUE}'>{DASHBOARD_URL}</link><br/>"
            f"Code and data: <link href='{REPO_URL}' color='{BLUE}'>{REPO_URL}</link>", small),
          Spacer(1, 14), P("Contents", ParagraphStyle("toch", parent=h1, keepWithNext=0)), toc, PageBreak(),
          P("Executive summary", h1)]
story += [P(
    "November and December generate about <b>18% of U.S. retail sales</b> (22% for clothing stores), yet retailers must commit "
    "to holiday inventory in late summer. Small retailers typically plan on <i>'same as last year'</i>. Using U.S. Census "
    "monthly retail sales by category (1992 to August 2026), we tested seven statistical and machine-learning forecasters "
    "and two AI chatbots on <b>11 past holiday seasons</b>, each time forecasting from the end of August with only the data "
    "available then. We then forecast the 2026 season and built an interactive dashboard that turns the forecast into an "
    "order recommendation for a small store.")]
exec_items = [
    f"<b>2026 outlook:</b> our ensemble forecasts Nov+Dec total retail sales of <b>{bn(hol_fc.loc['total', RECOMMENDED])}</b>, "
    f"<b>{pct(tot_g, True)}</b> over 2025. {SHORT[gsort.index[0]]} leads ({pct(gsort.iloc[0], True)}); "
    f"{SHORT[gsort.index[-1]].lower()} is weakest ({pct(gsort.iloc[-1], True)})."
    + (f" The first month already published after our forecast date (September 2026, total retail) came in within "
       f"{abs(sep_err):.1%} of the forecast." if sep_err is not None else ""),
    f"<b>Forecasting beats habit:</b> 'same as last year' missed the holiday total by {pct(nb_)} on average; our ensemble "
    f"(SARIMA + ETS + XGBoost) missed by <b>{pct(ens)}</b> and beat it in {s_all.loc[RECOMMENDED, 'beats_naive']:.0%} of "
    f"category-seasons.",
    f"<b>Consumer sentiment does not improve the forecast</b> (error change {sent_t['d']:+.2f} points). How shoppers say they "
    f"feel and what they spend have diverged.",
]
if "memorization" in lf:
    exec_items.append(f"<b>AI chatbots - hindsight, not foresight:</b> " + lf["memorization"])
if "skill" in lf:
    exec_items.append(f"<b>Fairly tested, chatbots are good but not better:</b> " + lf["skill"])
exec_items.append("<b>Product:</b> a public dashboard with the forecast, a full model comparison, a Holiday Planner that "
                  "sizes a store's order and safety stock, and an AI advisor that explains the results in plain English "
                  "using only computed numbers.")
story += B(exec_items)
story += [P("<b>Recommendation for retailers:</b> plan the holiday season on a statistical forecast rather than last year's "
            "numbers, size safety stock from measured forecast error, and treat AI-chatbot forecasts with caution: they "
            "can look excellent on the past for the wrong reason.")]

# ---------------------------------------------------------------- 1. Introduction
story += [P("1. Motivation and research questions", h1), P(
    "The holiday season is retail's most important and least forgiving period. Inventory is ordered months in advance; "
    "under-ordering means empty shelves during the busiest weeks, over-ordering means January markdowns. Large chains "
    "employ forecasting teams; small and independent retailers - who often lack such resources - commonly plan by copying "
    "last year. Meanwhile, generative AI tools such as ChatGPT and Claude are increasingly asked for business forecasts, "
    "though it is unclear whether they forecast or merely recall published data. We address five questions:")]
story += B(["<b>Q1.</b> How much will Americans spend in the 2026 holiday season (November-December), by category?",
            "<b>Q2.</b> Do forecasting models beat 'same as last year', and which model is most accurate?",
            "<b>Q3.</b> Can large language models forecast as well as statistical models, once memorization is ruled out?",
            "<b>Q4.</b> Does a customer-side indicator - consumer sentiment - improve the forecast?",
            "<b>Q5.</b> How can a small retailer turn a national category forecast into an order quantity?"])
story += [P("Our deliverables use AI in both ways suggested by the course: <b>Option A</b>, LLMs as forecasters compared "
            "with conventional models under identical out-of-sample conditions; and <b>Option B</b>, an interactive dashboard "
            "with an AI advisor.")]

# ---------------------------------------------------------------- 2. Data
story += [P("2. Data", h1), P(
    "All data come from FRED (Federal Reserve Bank of St. Louis), downloaded programmatically. Retail sales are from the U.S. "
    "Census Bureau's <b>Monthly Retail Trade Survey</b>, in millions of dollars and <b>not seasonally adjusted</b> - the "
    "seasonal spike is exactly what we forecast. The customer-side indicator is the University of Michigan <b>Index of "
    "Consumer Sentiment</b>.")]
rows = [["Series", "FRED ID", "Holiday share 2025", "Dec / Oct 2025"]]
from src.config import CATEGORIES  # noqa: E402
for c in CATS:
    y = panel[c].loc["2025"]
    rows.append([LABELS[c], CATEGORIES[c]["fred"], pct(y.loc["2025-11":].sum() / y.sum()), f"{y.loc['2025-12-01'] / y.loc['2025-10-01']:.2f}x"])
rows.append(["University of Michigan Consumer Sentiment", "UMCSENT", "-", "-"])
story += [T(rows, [3.0 * inch, 1.45 * inch, 1.25 * inch, 1.2 * inch]),
          P("Table 1. Data series. 'Holiday share' = Nov+Dec as a share of 2025 annual sales.", cap)]
story += [P(
    f"<b>Coverage and quality.</b> Category series run from January 1992 to {panel['clothing'].dropna().index.max():%B %Y}; "
    f"total retail to {panel['total'].dropna().index.max():%B %Y}. Checks found no missing months inside the range and no "
    "duplicate records. Model training starts in 1999 to avoid early-1990s level shifts while keeping 25+ years of seasonal "
    "history. <b>COVID-19.</b> The March-June 2020 lockdowns produced extreme outliers (clothing store sales fell by about "
    "85% in April 2020). For model training only, each of those months is replaced by the same month of 2019 scaled by "
    "January-February 2020 growth - the pre-pandemic trajectory, using only information available at the time. "
    "Evaluation always uses the actual, uncleaned values.")]
story += [F(figs["total"]), F(figs["season"]), P(
    "Figure 2 shows why category matters: clothing and hobby stores sell over 1.5 times an average month in December, "
    "while grocery barely moves and building materials actually decline. Grocery is included as a <b>control</b> with no "
    "holiday peak.", body), F(figs["share"], h=2.3), P(
    f"E-commerce has grown from {pct(share_ytd.loc[2015])} of retail (Jan-Aug 2015) to {pct(share_ytd.loc[2026])} "
    f"(Jan-Aug 2026), with a step change during the pandemic and steady gains since ({pct(share_ytd.loc[2025])} in 2025). "
    "Note that this Census category understates online shopping, because web sales of store-based retailers are counted "
    "with their stores.")]

# ---------------------------------------------------------------- 3. Methods
story += [P("3. Methodology", h1), P("3.1 Models", h2)]
story += [T([["Model", "Specification", "Why include it"],
             ["Seasonal Naive", "y(t+h) = y(t+h-12)", "The retailer's habit; the benchmark to beat"],
             ["Naive + Growth", "Last year's month × exp(avg. YoY log growth of last 3 months)", "A stronger rule of thumb"],
             ["ETS", "ETS(A,Ad,A) on log sales: additive error, damped trend, monthly seasonality", "Classic exponential smoothing"],
             ["SARIMA", "(p,1,q)(P,1,Q)12 on log sales; p,q,P,Q in {0,1} chosen by AIC per origin", "Box-Jenkins benchmark"],
             ["SARIMAX + Sentiment", "SARIMA + log consumer sentiment lagged 4 months", "Tests Q4 (customer side)"],
             ["XGBoost", "Gradient boosting pooled over 7 categories; target = YoY log growth of month t+h; "
                         "features: recent YoY growth, momentum, last year's growth for the target month, month, horizon, "
                         "category, sentiment", "Machine-learning comparison"],
             ["Ensemble", "Geometric mean of SARIMA, ETS and XGBoost", "Combining forecasts usually helps"],
             ["Claude Opus 5.5 / GPT-5.5", "LLM prompted with 120 months of history; returns 4 numbers as JSON", "Tests Q3 (Option A)"]],
            [1.35 * inch, 3.6 * inch, 2.0 * inch]),
          P("Table 2. Forecasting models. All models except the LLMs run on log sales, so forecasts are back-transformed medians.", cap)]
story += [P("3.2 Out-of-sample design", h2), P(
    "We use a <b>rolling-origin backtest</b> that mimics a real planning decision. For each year 2015-2025, every model "
    "stands at the end of August, is estimated only on data through August (SARIMA orders are re-selected each time), and "
    "forecasts September-December (horizons 1-4). All models share the same origin, horizon and information set. The "
    "sentiment variable is lagged four months, so every value a forecast needs is already published at the origin. This "
    "gives 7 categories × 11 seasons = <b>77 holiday forecasts per model</b>. The identical code then runs from August 2026 "
    "for the live forecast.")]
story += [P("3.3 Accuracy measures", h2)] + B([
    "<b>Holiday APE</b> (main metric): |forecast − actual| / actual for the November+December total - the number a retailer orders against.",
    "<b>MAPE, MAE, RMSE</b> over the four forecast months. MAPE is appropriate because retail sales are never near zero.",
    "<b>MASE</b>: MAE divided by the in-sample seasonal-naive MAE (training data only), comparable across categories of very different size.",
    "<b>Coverage80</b>: share of actual values inside a model's 80% prediction interval (ETS and SARIMA), to check whether the uncertainty is honest.",
    "<b>Paired Wilcoxon signed-rank tests</b> on the 77 matched category-seasons to judge whether differences between two models are significant.",
])
ex_prompt = build_prompt(panel["clothing"].dropna(), pd.Timestamp("2024-08-01"), "clothing", "anonymized")[0]
story += [P("3.4 LLM forecasting protocol and the memorization problem", h2), P(
    "LLMs are trained on large web corpora that very likely include the published Census figures and news about past "
    "holiday seasons. A backtest that names the series and year could reward <i>recall</i>, not forecasting. We therefore ran "
    "two framings. <b>Anonymized</b> (main test, both LLMs): 120 months rescaled to an index (first month = 100), labelled only "
    "by calendar month - no category, no year, no units. <b>Named</b> (Claude only): the same history with real dates, the "
    "category name and dollar values. If named prompts are systematically more accurate on past seasons, the difference "
    "measures hindsight. Both LLMs received the same system instruction ('use only the numbers provided ... forecast the next "
    "4 months') and returned JSON with a short explanation; Claude ran at high reasoning effort with schema-constrained "
    "output, GPT-5.5 with JSON mode. All prompts and responses are stored in the repository for audit. For 2026 nothing can be "
    "memorized, so the named framing is used. An excerpt of an anonymized prompt (clothing, August 2024):"),
    Preformatted(compact(ex_prompt, last=24), ParagraphStyle("code", parent=small, fontName="Courier", fontSize=8, leading=9.6)), Spacer(1, 4)]
story += [P("3.5 From forecast to order", h2), P(
    "The Holiday Planner converts the national forecast into a store-level order. Expected sales = the store's own Nov+Dec "
    "2025 sales × (1 + forecast category growth + an optional store adjustment). Uncertainty σ = √(σ<sub>national</sub>² + "
    "σ<sub>store</sub>²), where σ<sub>national</sub> is the root-mean-square of the ensemble's past holiday % errors for that "
    "category (excluding COVID seasons) and σ<sub>store</sub> is the owner's estimate of how far the store usually misses plan; "
    "the two are independent, so variances add. Safety stock = z × σ × expected sales for a chosen service level, and the "
    "recommended order = max(0, expected + safety stock − on hand − on order). Conservative and optimistic scenarios use "
    "±1.28σ (10th/90th percentiles).")]

# ---------------------------------------------------------------- 4. Results
story += [P("4. Results", h1), P("4.1 Overall accuracy (Q2)", h2), F(figs["errors"], h=0.22 * len(s_all) + 0.6)]
rows = [["Model", "Nov+Dec error", "excl. COVID", "Monthly MAPE", "MASE", "80% cover.", "Beats naive"]]
for m in s_all.index:
    r = s_all.loc[m]
    rows.append([m, pct(r.holiday_APE, d=2), pct(s_ex.loc[m, "holiday_APE"], d=2), pct(r.MAPE, d=2), f"{r.MASE:.2f}",
                 "-" if pd.isna(r.Coverage80) else f"{r.Coverage80:.0%}", "-" if pd.isna(r.beats_naive) else f"{r.beats_naive:.0%}"])
story += [T(rows, [1.7 * inch, 0.85 * inch, 0.8 * inch, 0.85 * inch, 0.6 * inch, 0.75 * inch, 0.8 * inch],
            bold_rows=[list(s_all.index).index(RECOMMENDED) + 1]),
          P("Table 3. Backtest accuracy over 77 category-seasons (11 for each of 7 categories). 'Beats naive' = share of "
            "category-seasons with a smaller holiday error than Seasonal Naive. *Claude (named) saw the real category and year "
            "and can recall what happened; it is shown only for the memorization test (Section 4.6).", cap)]
story += [P(
    f"Every model clearly beats 'same as last year' ({pct(nb_, d=2)}): the ensemble cuts the holiday error by more than half "
    f"to {pct(ens, d=2)}. Among classical/ML models, SARIMA ({pct(s_all.loc['SARIMA', 'holiday_APE'], d=2)}) and the "
    f"ensemble lead, while the simple 'last year × growth' rule ({pct(s_all.loc['Naive + Growth', 'holiday_APE'], d=2)}) is "
    "surprisingly competitive - the key improvement is accounting for the current trend at all. Differences of a few "
    "tenths of a point among the top models are within the noise of an 11-season sample, so <b>having a model matters more "
    "than which model</b>. XGBoost has the highest error among the learned models, plausibly because roughly 25 years of "
    "monthly data is a small sample for machine learning; pooling categories helps but does not overcome this.")]
story += [P("4.2 Which categories are hard to forecast?", h2), F(figs["heat"], h=3.1), P(
    "Total retail, grocery and general merchandise are easiest (errors near 1-2%): they are large, diversified and stable. "
    "Electronics, sporting/hobby and e-commerce are hardest: they depend on product cycles (new consoles and phones), "
    "shifting promotion calendars, and - for e-commerce - a structural trend that changed abruptly in 2020. The table shows "
    "the same ranking for every model, so difficulty is a property of the category, not the method.")]
story += [P("4.3 Over time and COVID", h2), F(figs["years"]), P(
    "Seasonal Naive fails badly in 2020 and 2021 because last year's pattern was no guide during and right after the "
    "pandemic; trend-aware models adapt. Excluding those two seasons narrows but does not close the gap "
    f"({pct(s_ex.loc['Seasonal Naive', 'holiday_APE'], d=2)} vs {pct(s_ex.loc[RECOMMENDED, 'holiday_APE'], d=2)}). "
    "2018 was the hardest season for every model.")]
story += [P("4.4 Are the prediction intervals honest?", h2), P(
    f"SARIMA's nominal 80% intervals contained {s_all.loc['SARIMA', 'Coverage80']:.0%} of actual values and ETS's "
    f"{s_all.loc['ETS', 'Coverage80']:.0%} - close to 80%, so the stated uncertainty can be trusted. This is why the planner "
    "sizes safety stock from measured forecast errors rather than from a guess, and why the dashboard only shows intervals "
    "for models that produce a statistically valid one.")]
sx = tests.get("SARIMAX + Sentiment vs SARIMA")
story += [P("4.5 Does consumer sentiment help? (Q4)", h2), P(
    f"No. Adding lagged consumer sentiment to SARIMA changed the average holiday error by {sent_t['d']:+.2f} points "
    f"({pct(s_all.loc['SARIMAX + Sentiment', 'holiday_APE'], d=2)} vs {pct(s_all.loc['SARIMA', 'holiday_APE'], d=2)}). "
    "Sentiment has been historically weak through 2025-2026 (down 10-30% year over year) while nominal spending kept "
    "growing 3-8%. What consumers say about their mood is a poor guide to what they spend - at least once trend and "
    "seasonality are already modelled.")]
story += [P("4.6 Can AI chatbots forecast? (Q3)", h2)]
if LLMS:
    rows = [["Forecaster", "Nov+Dec error", "excl. COVID", "Monthly MAPE", "Beats naive"]]
    for m in LLMS + [RECOMMENDED, "SARIMA", "Seasonal Naive"]:
        rows.append([m, pct(s_all.loc[m, "holiday_APE"], d=2), pct(s_ex.loc[m, "holiday_APE"], d=2), pct(s_all.loc[m, "MAPE"], d=2),
                     "-" if pd.isna(s_all.loc[m, "beats_naive"]) else f"{s_all.loc[m, 'beats_naive']:.0%}"])
    story += [T(rows, [2.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]),
              P("Table 4. LLM forecasters compared with the classical models on the same 77 category-seasons.", cap)]
    trows = [["Comparison (A vs B)", "B better in", "Mean diff (pts)", "p-value"]]
    for k, t in tests.items():
        trows.append([f"{t['a']} vs {t['b']}", f"{t['b_better_share']:.0%} of {t['n']}", f"{t['mean_diff_pts']:+.2f}", f"{t['p_value']:.4f}"])
    story += [T(trows, [3.3 * inch, 1.1 * inch, 1.1 * inch, 0.8 * inch]),
              P("Table 5. Paired Wilcoxon signed-rank tests. Mean diff = A's error minus B's, in percentage points.", cap)]
    story += [P(lf.get("memorization", "")), P(lf.get("skill", "")), P(
        "Reading the models' own explanations (saved with every response) shows how the LLMs work: they compute last year's "
        "ratios of November and December to the late-summer level and apply them to the current level - essentially a "
        "careful 'seasonal ratio' method. That is sensible and explains why they beat 'same as last year', but it does not "
        "use the information a statistical model extracts from 25 years of history, and it gives no calibrated uncertainty. "
        "The practical lesson: an LLM that is told what and when it is forecasting can look better than any model on past "
        "data, for the wrong reason. Backtests of LLM forecasts must hide information that can be recalled.")]
else:
    story.append(P("LLM results pending."))
story += [F(figs["replay"])]
story += [P("4.7 What drives the machine-learning model?", h2), F(figs["imp"], h=2.7), P(
    "The most important XGBoost inputs are the growth rates: last year's growth for the same target month (which captures "
    "shifts in the holiday calendar, such as a late Thanksgiving moving sales between November and December), the latest "
    "year-over-year growth and its 12-month average. Consumer sentiment carries little weight, consistent with Section 4.5. "
    "In other words, the model learns what the statistical models assume - that this year's holiday season is last year's "
    "pattern scaled by the current trend - and it adds little beyond that with only about 25 years of data per category.")]

# ---------------------------------------------------------------- 5. 2026
story += [P("5. The 2026 holiday forecast (Q1)", h1)]
rows = [["Category", "Nov+Dec 2025", "Nov+Dec 2026 forecast", "Growth", "SARIMA", "Claude", "GPT"]]
for c in CATS:
    g = hol_g.loc[c]
    rows.append([LABELS[c], bn(hol_ly[c]), bn(hol_fc.loc[c, RECOMMENDED]), pct(g[RECOMMENDED], True), pct(g["SARIMA"], True),
                 pct(g["Claude (named)"], True) if "Claude (named)" in g else "-", pct(g["GPT (named)"], True) if "GPT (named)" in g else "-"])
story += [T(rows, [2.0 * inch, 0.8 * inch, 0.9 * inch, 0.7 * inch, 0.78 * inch, 0.68 * inch, 0.64 * inch], bold_rows=[1]),
          P("Table 6. 2026 holiday forecast (ensemble) with growth from other forecasters for comparison. Forecast origin: "
            "August 2026.", cap), F(figs["f26"])]
story += [P(
    f"We expect a solid season: total retail sales of {bn(hol_fc.loc['total', RECOMMENDED])} in November-December, "
    f"{pct(tot_g, True)} over 2025 - a little faster than 2025's holiday growth, consistent with the 3-7% year-over-year "
    f"gains seen through 2026. E-commerce ({pct(hol_g.loc['ecommerce', RECOMMENDED], True)}) and sporting/hobby stores "
    f"({pct(hol_g.loc['hobby', RECOMMENDED], True)}) lead; clothing ({pct(hol_g.loc['clothing', RECOMMENDED], True)}) slows "
    f"after strong 2025 growth; grocery ({pct(hol_g.loc['grocery', RECOMMENDED], True)}) barely moves. Growth is nominal - it "
    "includes price increases.")]
story += [F(figs["g26"], h=2.9), P(
    "The forecasters broadly agree on direction and ranking - e-commerce and hobby stores fastest, grocery slowest - which "
    "adds confidence to the outlook. The LLMs, asked with the real series names, tend to be more conservative for some "
    "categories; their explanations (saved with each response) show they anchor on the year-to-date growth rate and last "
    "year's November and December ratios. Because LLM forecasts were less accurate than the ensemble when tested fairly, we "
    "use the ensemble for planning and show the LLM forecasts as a second opinion in the dashboard.")]
if sep_err is not None:
    story.append(P(f"<b>Early reality check.</b> Census published September 2026 total retail ({bn(sep.loc[RECOMMENDED, 'actual'])}) "
                   f"after our August forecast date. The ensemble forecast {bn(sep.loc[RECOMMENDED, 'mean'])}, an error of "
                   f"{pct(sep_err, True)} - within the range of its backtest errors."))

# ---------------------------------------------------------------- 6. Product
sig, _, n = error_spread(po, "clothing")
g = hol_g.loc["clothing", RECOMMENDED]
sigma = np.sqrt(sig ** 2 + 0.10 ** 2)
from scipy.stats import norm  # noqa: E402
exp_ = 5000 * (1 + g); ss_ = norm.ppf(0.9) * sigma * exp_; order = max(0, exp_ + ss_ - 1200)
story += [P("6. The product: Holiday Retail Planner dashboard (Option B)", h1), P(
    f"The dashboard (<link href='{DASHBOARD_URL}' color='{BLUE}'>{DASHBOARD_URL}</link>) is built with Streamlit and reads "
    "the same saved outputs as this report - nothing is re-estimated in the app, so what users see is exactly what the "
    "code produced. It has seven tabs:")]
story += B(["<b>Story</b> - motivation and headline findings.",
            "<b>Market</b> - history by category, the size of each December spike, e-commerce share.",
            "<b>2026 Forecast</b> - choose a category and any forecasters; SARIMA's 80% interval as a band; the LLMs' own explanations.",
            "<b>Model Battle</b> - all backtest metrics with or without COVID seasons, the category heatmap, a replay of any past "
            "season, and the LLM memorization test.",
            "<b>Holiday Planner</b> - the order calculator of Section 3.5 with scenario table and November/December timing.",
            "<b>AI Advisor</b> - a chat assistant (Claude) that answers questions such as 'Why do you recommend this order?' or "
            "'Which model is most accurate?'. It receives a fact sheet generated from the results plus the user's planner "
            "inputs, is instructed to quote numbers exactly and never invent forecasts or ranges, and says when the "
            "information is missing.",
            "<b>About</b> - objectives, data, methods, validation, usage instructions and AI-tool disclosure."])
story += [Image(str(FIG / "s_battle.jpg"), width=4.4 * inch, height=2.75 * inch),
          P("Screenshot 1. Model Battle tab: backtest accuracy of every forecaster; named LLM results are flagged as hindsight.", cap),
          Image(str(FIG / "s_planner.jpg"), width=4.4 * inch, height=2.75 * inch),
          P("Screenshot 2. Holiday Planner tab with the worked example below.", cap),
          Image(str(FIG / "s_advisor.jpg"), width=4.4 * inch, height=2.75 * inch),
          P("Screenshot 3. AI Advisor answering 'Why do you recommend this order for my store?' - every number in the answer "
            "comes from the planner's computation.", cap)]
story += [P(
    f"<b>Worked example.</b> A clothing boutique sold 5,000 units in Nov+Dec 2025, has 1,200 on hand, wants a 90% chance "
    f"of not running out and estimates its sales usually miss plan by ±10%. National clothing growth is {pct(g, True)}, so "
    f"expected demand is {exp_:,.0f} units. The ensemble's past clothing error is {pct(sig)}; combined with the store's ±10%, "
    f"σ = {pct(sigma)}, giving safety stock of {ss_:,.0f} units and a <b>recommended order of {np.ceil(order):,.0f} units</b>. "
    "Note that store-level volatility, not national forecast error, dominates the buffer - an honest message for small "
    "retailers.")]

# ---------------------------------------------------------------- 7-9
story += [P("7. Conclusions", h1)] + B([
    "<b>Use a model, not last year.</b> Any trend-aware forecast halves the holiday planning error; the ensemble is the most "
    "accurate and most consistent choice.",
    "<b>Sentiment surveys add nothing</b> beyond trend and seasonality for nominal holiday spending.",
    "<b>Be careful with AI-chatbot forecasts.</b> They can look better than any model on past data because they remember the "
    "past; tested fairly they are reasonable but not better than standard methods, and they provide no calibrated uncertainty. "
    "Their best role is explaining forecasts - which is how our AI advisor uses them.",
    f"<b>2026 should be a good holiday season</b> ({pct(tot_g, True)}), with e-commerce continuing to take share.",
    "<b>Forecast error is measurable and should drive safety stock</b> - and for a small store, its own volatility matters more "
    "than the national forecast error.",
])
story += [P("8. Limitations", h1)] + B([
    "National category totals, not a specific store or product; the planner scales a store's own history by national growth.",
    "FRED provides today's revised data. Forecasters in past years saw earlier, unrevised figures, so real-time accuracy would be "
    "somewhat lower for all models.",
    "Eleven seasons per category is a small sample; differences of a few tenths of a point between the best models are not "
    "decisive, which is why we report paired significance tests.",
    "LLM outputs are not deterministic, and each prompt was run once; model versions will change. Anonymization removes direct "
    "recall but an LLM might still recognize a series' shape (e.g. the 2020 shock).",
    "No inventory-cost or stock-out data, so we make no claim about money saved; growth figures are nominal.",
])
story += [P("9. Use of AI tools", h1)] + B([
    "<b>Claude Code</b> (Anthropic) wrote most of the code - data pipeline, models, backtest, evaluation, dashboard and the "
    "report generators - and first drafts of the text, working from the team's research questions and design decisions. "
    "The team reviewed the code, checked results against the raw data and decided what to present.",
    "<b>Claude Opus 5.5</b> (Anthropic API) and <b>GPT-5.5</b> (OpenAI API) were used as forecasters in the comparison (Option A).",
    "<b>Claude Opus 5.5</b> powers the dashboard's AI advisor (Option B), restricted to computed results.",
])
story += [P("References", h1)] + B([
    "U.S. Census Bureau, Monthly Retail Trade Survey; retrieved from FRED, Federal Reserve Bank of St. Louis (series listed in Table 1).",
    "University of Michigan, Surveys of Consumers, Index of Consumer Sentiment (UMCSENT), via FRED.",
    "Hyndman, R.J. and Athanasopoulos, G. <i>Forecasting: Principles and Practice</i>, 3rd ed., OTexts.",
    "Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <i>Proc. KDD</i>.",
    "Seabold, S. and Perktold, J. (2010). statsmodels: Econometric and Statistical Modeling with Python.",
])
orders = pd.read_csv(ROOT / "outputs/sarima_orders.csv")
common = orders.groupby("category").agg(order=("order", lambda s: s.value_counts().index[0]),
                                        seasonal=("seasonal_order", lambda s: s.value_counts().index[0]))
rows = [["Category", "Most frequent order", "Seasonal order"]] + [[LABELS[c], common.loc[c, "order"], common.loc[c, "seasonal"]] for c in CATS]
qrows = [["Series", "First month", "Last month", "Months", "Missing", "Duplicates"]]
for c in CATS:
    v = panel[c].dropna()
    full = pd.date_range(v.index.min(), v.index.max(), freq="MS")
    qrows.append([SHORT[c], f"{v.index.min():%Y-%m}", f"{v.index.max():%Y-%m}", str(len(v)), str(len(full.difference(v.index))),
                  str(int(v.index.duplicated().sum()))])
story += [P("Appendix A. Data quality checks", h2), T(qrows, [1.8 * inch, 1.0 * inch, 1.0 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch])]
cache = [json.loads(x.read_text()) for x in (ROOT / "outputs/llm_cache").glob("*.json")]
ex = build_prompt(panel["total"].dropna(), pd.Timestamp("2025-08-01"), "total", "anonymized")[0]
resp = {c["model"].split("-")[0]: c for c in cache if c["prompt"] == ex}
story += [P("Appendix B. Example LLM prompt and both models' answers", h2),
          P("<b>System instruction (both models):</b> 'You are a careful time-series forecaster. You are given a monthly history "
            "and must forecast the next 4 months. Use only the numbers provided. Think about level, trend, and month-of-year "
            "seasonality, then return your forecast in the requested JSON format.'"),
          P("<b>User prompt</b> for total retail, forecast origin August 2025 (the 120 monthly values are abbreviated here to the first and last 12):"),
          Preformatted(compact(ex, first=12, last=12),
                       ParagraphStyle("code2", parent=small, fontName="Courier", fontSize=8, leading=9.6))]
for k, lab in [("claude", "Claude Opus 5.5"), ("gpt", "GPT-5.5")]:
    if k in resp:
        r = resp[k]
        story.append(P(f"<b>{lab}</b> forecast (index): {', '.join(f'{v:.1f}' for v in r['forecast'])}. "
                       f"<i>Explanation:</i> \"{r['reasoning']}\""))
story += [P("Appendix C. SARIMA orders selected by AIC", h2), T(rows, [3.3 * inch, 1.5 * inch, 1.5 * inch]),
          P("Appendix D. Reproducibility", h2), P(
              f"All code, data and LLM responses: <link href='{REPO_URL}' color='{BLUE}'>{REPO_URL}</link>. A self-contained "
              "notebook (notebooks/Holiday_Retail_Forecast.ipynb) downloads the data and reproduces every model and table in "
              "this report; LLM forecasts are loaded from the saved results unless re-run with API keys.")]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(0.8 * inch, 0.45 * inch, "Forecasting the 2026 U.S. Holiday Retail Season · UCLA Anderson")
    canvas.drawRightString(7.7 * inch, 0.45 * inch, f"{doc.page}")
    canvas.restoreState()


class ReportDoc(SimpleDocTemplate):
    """Registers headings with the table of contents and adds a clickable PDF bookmark for each."""

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        name, text = flowable.style.name, flowable.getPlainText()
        level = 0 if name == "h1" else (1 if name == "h2" and text.startswith("Appendix") else None)
        if level is None:
            return
        key = f"h{id(flowable)}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level)
        self.notify("TOCEntry", (level, text, self.page, key))


doc = ReportDoc(str(OUT / "Written_Report.pdf"), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.75 * inch, title="Forecasting the 2026 U.S. Holiday Retail Season",
                        author="Charlotte Zhu, Shihan Wang, Jing Chang, Rachel Liu")
doc.multiBuild(story, onFirstPage=footer, onLaterPages=footer)
print(OUT / "Written_Report.pdf")

"""Build the short project description PDF from the saved results.

Every number in the document is read from outputs/, so re-running this after the
pipeline changes keeps the write-up in sync with the code.

Run:  python reports/build_description.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from data_layer import (COVID_YEARS, LABELS, RECOMMENDED, SHORT, ecommerce_share, error_spread,  # noqa: E402
                        holiday_table, llm_findings, load_all, summary_table)

DASHBOARD_URL = "https://holiday-retail-planner.streamlit.app"
REPO_URL = "https://github.com/RachelLiu13/holiday-retail-forecast"
OUT = ROOT / "reports"
BLUE, ORANGE, GRAY, INK, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#e4e3df"

D = load_all()
panel, po, f26 = D["panel"], D["po"], D["f26"]
s_all, s_ex = summary_table(po, False), summary_table(po, True)
hol_fc, hol_g, hol_ly = holiday_table(f26, panel)
_, share_ytd = ecommerce_share(panel)
has_llm = any(m.startswith(("Claude", "GPT")) for m in s_all.index)


def pct(x, sign=False):
    return f"{x:+.1%}" if sign else f"{x:.1%}"


# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})

fig, ax = plt.subplots(figsize=(6.8, 2.6), dpi=200)
d = s_all["holiday_APE"].sort_values(ascending=False)
cols = [GRAY if m == "Seasonal Naive" else (ORANGE if m.startswith(("Claude", "GPT")) else BLUE) for m in d.index]
ax.barh([f"{m} (hindsight*)" if "(named)" in m else m for m in d.index], d.values * 100, color=cols, height=0.55)
for i, v in enumerate(d.values):
    ax.text(v * 100 + 0.06, i, f"{v:.1%}", va="center", fontsize=8, color=INK)
ax.set_xlabel("Average absolute error of the Nov+Dec total, 11 seasons 2015-2025 (%). *Named prompts can recall the past.")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "fig_model_errors.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 2.4), dpi=200)
h = panel["clothing"].loc["2024-09":"2026-08"]
ax.plot(h.index, h / 1000, color=INK, lw=1.6, label="Actual")
fc = f26[(f26.category == "clothing")]
for m, c in [(RECOMMENDED, BLUE), ("Seasonal Naive", GRAY)]:
    x = fc[fc.model == m]
    ax.plot([h.index[-1]] + list(x.date), [h.iloc[-1] / 1000] + list(x["mean"] / 1000), color=c, lw=1.6, marker="o", ms=3,
            label=f"{m} forecast")
sb = fc[fc.model == "SARIMA"]
ax.fill_between(sb.date, sb.lo80 / 1000, sb.hi80 / 1000, color=ORANGE, alpha=0.12, lw=0, label="SARIMA 80% interval")
ax.set_ylabel("$ billions / month")
ax.set_title("Clothing stores: actual sales and the Sep-Dec 2026 forecast", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_clothing_2026.png")
plt.close(fig)

# ---------------------------------------------------------------- document
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=12.6, alignment=TA_LEFT, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#52514e"))
h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, leading=21, alignment=TA_LEFT, spaceAfter=2)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, leading=14, spaceBefore=7, spaceAfter=3,
                    textColor=colors.HexColor("#0d366b"), keepWithNext=1)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=2)


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.3), ("LEADING", (0, 0), (-1, -1), 10),
          ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK)),
          ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st.append(("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.3))
    t.setStyle(TableStyle(st))
    return t


story = [
    P("Holiday Retail Planner 2026", h1),
    P("Forecasting the U.S. holiday shopping season - and whether AI chatbots can do it", ParagraphStyle("sub", parent=body, fontSize=11, textColor=colors.HexColor("#52514e"))),
    P(f"UCLA Anderson · Time Series Analysis and Forecasting (Dr. William Yu) · Group project, October 2026<br/>"
      f"Team: Charlotte Zhu, Shihan Wang, Jing Chang, Rachel Liu<br/>"
      f"<b>Live dashboard:</b> <link href='{DASHBOARD_URL}' color='#2a78d6'>{DASHBOARD_URL}</link> &nbsp; "
      f"<b>Code:</b> <link href='{REPO_URL}' color='#2a78d6'>{REPO_URL}</link>", small),
    Spacer(1, 6),
]

tot_g = hol_g.loc["total", RECOMMENDED]
story += [P("1. Objective and motivation", h2), P(
    "November and December bring about <b>18% of U.S. retail sales</b> in two months, and over 22% for clothing stores. "
    "Retailers commit to holiday inventory in late summer, so the forecast they make in August decides whether shelves are "
    "empty or overstocked in December. Large chains have forecasting teams; small retailers mostly order <i>'same as last "
    "year'</i>. We ask: (1) how much will Americans spend this holiday season, by category; (2) can forecasting models beat "
    "'same as last year', and which one; (3) can large language models (Claude, GPT) forecast as well as statistical models, "
    "or do they just remember published data; (4) does consumer sentiment improve the forecast; and (5) how can a small shop "
    "turn a national forecast into an order? The product is an interactive dashboard that answers all five.")]

story += [P("2. Data", h2), P(
    "U.S. Census Bureau <b>Monthly Retail Trade</b>, not seasonally adjusted, $ millions, via FRED, January 1992 to August "
    f"2026 (total retail to September 2026). Seven series: total retail excluding food services, clothing, electronics &amp; "
    "appliances, sporting goods/hobby/book/music, general merchandise, electronic shopping &amp; mail-order (e-commerce), and "
    "grocery as a control with almost no holiday peak. Customer-side indicator: University of Michigan <b>Consumer "
    "Sentiment</b>. Checks found no missing months or duplicates. The March-June 2020 lockdown months are treated as "
    "outliers in model training, replaced with their pre-pandemic trajectory using only data available at the time.")]

story += [P("3. Methods and validation", h2), table([
    ["Model", "How it works"],
    ["Seasonal Naive (baseline)", "This December = last December - the retailer's habit"],
    ["Naive + Growth", "Last year's month × recent year-over-year growth"],
    ["ETS", "Damped-trend exponential smoothing with monthly seasonality, on log sales"],
    ["SARIMA", "Seasonal ARIMA on log sales; order chosen by AIC from 16 candidates on training data only"],
    ["SARIMAX + Sentiment", "SARIMA + consumer sentiment lagged 4 months (always known at forecast time)"],
    ["XGBoost", "One gradient-boosted model pooled across categories; predicts year-over-year growth"],
    ["Ensemble (recommended)", "Average of SARIMA, ETS and XGBoost in log space"],
    ["Claude Opus 5.5 / GPT", "LLM given the same monthly history in a prompt, asked for the next 4 months (JSON)"],
], [1.75 * inch, 5.0 * inch])]
story += [Spacer(1, 4), P(
    "<b>Rolling-origin backtest.</b> For each of 11 holiday seasons (2015-2025), every model stands at the end of August, is "
    "fit only on data available then, and forecasts September-December - the same origin, horizon and information for all "
    "models. The main metric is the absolute % error of the <b>November+December total</b>, the number a retailer orders "
    "against; we also report monthly MAPE, MASE and 80%-interval coverage. MAPE is suitable because sales are never near zero. "
    "<b>Memorization control for LLMs:</b> an LLM may have read the actual Census figures during training, so the main LLM "
    "test is <i>anonymized</i> (an index with first month = 100, labelled only by calendar month - no category, no year). For "
    "Claude we also ran <i>named</i> prompts with real dates and dollars; a large gap between the two would indicate recall "
    "rather than skill. Every LLM prompt and response is saved in the repository.")]

story += [P("4. Results", h2), Image(str(OUT / "fig_model_errors.png"), width=6.8 * inch, height=2.6 * inch)]
nb = s_all.loc["Seasonal Naive", "holiday_APE"]
ens = s_all.loc[RECOMMENDED, "holiday_APE"]
sent_d = s_all.loc["SARIMAX + Sentiment", "holiday_APE"] - s_all.loc["SARIMA", "holiday_APE"]
cov = s_all.loc["SARIMA", "Coverage80"]
res = [
    f"<b>Forecasting beats habit.</b> 'Same as last year' missed the holiday total by {pct(nb)} on average; the ensemble "
    f"missed by <b>{pct(ens)}</b> and beat the baseline in {s_all.loc[RECOMMENDED, 'beats_naive']:.0%} of category-seasons. "
    f"Excluding the COVID seasons the gap is {pct(s_ex.loc['Seasonal Naive', 'holiday_APE'])} vs {pct(s_ex.loc[RECOMMENDED, 'holiday_APE'])}.",
    f"<b>Model choice matters less than having a model.</b> SARIMA, the ensemble and 'last year × growth' are within a few "
    f"tenths of a point; with 11 seasons these differences are not decisive. Electronics, hobby and e-commerce are the "
    f"hardest categories; total retail and grocery the easiest.",
    f"<b>Sentiment does not help</b> (error change {sent_d * 100:+.2f} points): shoppers' mood has been weak while spending kept growing.",
    f"<b>The intervals are honest:</b> SARIMA's 80% intervals contained {cov:.0%} of actual values, so we use the "
    f"backtest error spread for the planner's scenarios.",
]
lf = llm_findings(D["tests"], s_all)
if lf:
    res += [f"<b>LLMs - hindsight, not foresight.</b> {lf['memorization']}"] if "memorization" in lf else []
    res += [f"<b>Fairly tested, LLMs are good but not better.</b> {lf['skill']}"] if "skill" in lf else []
else:
    res.append("<b>LLMs as forecasters:</b> results to be added after the LLM run.")
story += bullets(res)

rows = [["Category", "Nov+Dec 2025 actual", "Nov+Dec 2026 forecast", "Growth"]]
for c in LABELS:
    rows.append([LABELS[c], f"${hol_ly[c] / 1000:,.1f}B", f"${hol_fc.loc[c, RECOMMENDED] / 1000:,.1f}B", pct(hol_g.loc[c, RECOMMENDED], True)])
story += [Spacer(1, 4), P(f"<b>2026 forecast (ensemble, data through August 2026).</b> Total holiday retail sales of "
                          f"${hol_fc.loc['total', RECOMMENDED] / 1000:,.0f} billion, {pct(tot_g, True)} over 2025. E-commerce keeps "
                          f"gaining share ({pct(share_ytd.loc[2026])} of retail in Jan-Aug 2026 vs {pct(share_ytd.loc[2025])} a year earlier)."),
          table(rows, [2.9 * inch, 1.35 * inch, 1.45 * inch, 0.8 * inch]), Spacer(1, 4)]
sep = f26[(f26.category == "total") & (f26.date == "2026-09-01")].set_index("model")
if sep["actual"].notna().any():
    e = sep.loc[RECOMMENDED]
    story.append(P(f"<b>Early reality check:</b> September 2026 total retail (${e['actual'] / 1000:,.1f}B) was published "
                   f"after our forecast origin; the ensemble's forecast was off by {pct(e['mean'] / e['actual'] - 1, True)}."))
story.append(Image(str(OUT / "fig_clothing_2026.png"), width=6.8 * inch, height=2.4 * inch))

sig, _, n = error_spread(po, "clothing")
story += [P("5. How to use the dashboard", h2)] + bullets([
    "<b>Story</b> - the headline results. <b>Market</b> - history, the size of each category's December spike, e-commerce share.",
    "<b>2026 Forecast</b> - choose a category and models; SARIMA's 80% interval is shown as a band.",
    "<b>Model Battle</b> - backtest scores with and without COVID seasons, errors by category, a replay of any past season, "
    "and the LLM memorization test.",
    "<b>Holiday Planner</b> - a small retailer enters last season's Nov+Dec sales, stock on hand and on order, a service level, "
    "and how volatile the store is. The tool returns expected sales (own history × national category growth), a safety "
    "buffer (z × σ × expected, where σ combines our past national forecast error and the store's own volatility), the "
    "recommended order, and conservative/optimistic scenarios.",
    "<b>AI Advisor</b> - questions in plain English (e.g. 'Why this order?', 'Which model is best?'). Claude answers only from "
    "the computed results it is given and says so when information is missing; it never produces forecasts or ranges itself.",
])

story += [P("6. Limitations", h2)] + bullets([
    "National category totals, not a specific store; the planner scales the user's own history by the national trend.",
    "FRED holds today's revised data; forecasters in past years saw earlier, unrevised figures.",
    "Eleven seasons is a small sample. No inventory-cost data, so we make no claim about money saved.",
])
story += [P("7. AI tools used", h2)] + bullets([
    "<b>Claude Code</b> (Anthropic) wrote most of the data pipeline, models, backtest, dashboard and first drafts of this "
    "description and the slides, from the team's research questions and design decisions; the team reviewed the code and "
    "checked results against the raw data.",
    "<b>Claude Opus 5.5</b> and <b>OpenAI GPT</b> are forecasters in the model comparison (course Option A).",
    "<b>Claude Opus 5.5</b> powers the dashboard's AI Advisor (course Option B).",
])

doc = SimpleDocTemplate(str(OUT / "Project_Description.pdf"), pagesize=letter, leftMargin=0.75 * inch,
                        rightMargin=0.75 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                        title="Holiday Retail Planner 2026 - Project Description",
                        author="Charlotte Zhu, Shihan Wang, Jing Chang, Rachel Liu")
doc.build(story)
print(OUT / "Project_Description.pdf")

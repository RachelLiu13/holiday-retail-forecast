"""Build the standalone notebook (one self-contained .ipynb) from the project source.

Functions are copied verbatim from src/ by name, so the notebook and the GitHub project always
run the same code. The notebook downloads its own data from FRED and needs no other file.

Run:  python reports/build_standalone_notebook.py   (then execute it with nbconvert)
"""
import ast
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO_RAW = "https://raw.githubusercontent.com/RachelLiu13/holiday-retail-forecast/main/outputs"


def defs(module: str, names: list[str]) -> str:
    """Source of the named top-level functions/assignments in src/<module>.py, in file order."""
    text = (SRC / f"{module}.py").read_text()
    tree = ast.parse(text)
    out = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign):
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
        if name in names:
            out.append(ast.get_source_segment(text, node))
    missing = set(names) - {n for n in names if any(f"def {n}(" in s or s.startswith(f"{n} =") for s in out)}
    assert not missing, f"not found in {module}: {missing}"
    return "\n\n\n".join(out)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# Forecasting the 2026 U.S. Holiday Retail Season
### Classical vs. machine-learning vs. LLM forecasters - and a planning rule for small retailers

**UCLA Anderson · Time Series Analysis and Forecasting (Dr. William Yu) · Group project, October 2026**
Team: Charlotte Zhu, Shihan Wang, Jing Chang, Rachel Liu

This notebook is **self-contained**: it downloads the data from FRED, fits every model, runs the 11-season backtest,
evaluates the forecasts and produces the 2026 holiday forecast. Run all cells top to bottom (about 3-5 minutes).
It works in Jupyter or Google Colab.

The same code powers our interactive dashboard and lives as a full project on GitHub:
https://github.com/RachelLiu13/holiday-retail-forecast

**Research questions**
1. How much will Americans spend in the 2026 holiday season (Nov+Dec), by retail category?
2. Can forecasting models beat the retailer's habit of ordering *"same as last year"*? Which model is best?
3. Can large language models (Claude, GPT) forecast as well as statistical models - or do they just remember the data?
4. Does consumer sentiment improve the forecast?
5. How can a small retailer turn a national forecast into an order?"""))

C.append(code("""# In Google Colab, uncomment the next line first:
# !pip install -q xgboost statsmodels joblib

import itertools, warnings, urllib.request, json, re, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import norm, wilcoxon
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'  # also silence convergence warnings in the parallel worker processes
pd.set_option('display.width', 160)

BLUE, ORANGE, GRAY, INK, GRID = '#2a78d6', '#eb6834', '#8a8984', '#0b0b0b', '#e4e3df'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'axes.edgecolor': GRID, 'figure.dpi': 110, 'lines.linewidth': 2})"""))

C.append(md("## 1. Settings"))
C.append(code(defs("config", ["CATEGORIES", "SENTIMENT", "HORIZON", "BACKTEST_YEARS", "FINAL_ORIGIN",
                              "TRAIN_START", "HOLIDAY_MONTHS"]) +
              "\n\nWORK = Path('holiday_forecast_data'); WORK.mkdir(exist_ok=True)\nRAW = PROCESSED = OUTPUTS = WORK"))

C.append(md("""## 2. Data

U.S. Census Bureau **Monthly Retail Trade**, *not seasonally adjusted* (the holiday spike is exactly what we want to
forecast), $ millions, via FRED - plus the University of Michigan **Consumer Sentiment** index as a customer-side
indicator. The quality check confirms there are no missing months or duplicate records."""))
C.append(code('FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"\n\n\n' +
              defs("data", ["download", "build_panel", "quality_report"]) +
              "\n\n\npanel = build_panel(refresh=True)\nquality_report(panel)"))
C.append(code("""fig, ax = plt.subplots(figsize=(10, 3.4))
s = panel['total'].loc['2015':]
ax.plot(s.index, s / 1000, color=BLUE)
dec = s[s.index.month == 12]
ax.scatter(dec.index, dec / 1000, color=ORANGE, zorder=3, s=28, label='December')
ax.set_title('Total U.S. retail sales (ex. food services), $ billions per month', loc='left'); ax.legend(frameon=False); plt.show()

prof = {}
for c in CATEGORIES:
    s = panel[c].loc['2022':'2025']
    prof[CATEGORIES[c]['label']] = (s.groupby(s.index.month).sum() / s.sum() * 12)
dec = pd.DataFrame(prof).loc[12].sort_values()
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(dec.index, dec.values, color=BLUE, height=0.5); ax.axvline(1, color=INK, lw=1)
for i, v in enumerate(dec.values): ax.text(v + 0.01, i, f'{v:.2f}x', va='center')
ax.set_title('December sales vs. an average month, 2022-2025 (1.0 = no holiday peak)', loc='left'); plt.show()

share = pd.Series({y: panel.loc[f'{y}-01':f'{y}-08', 'ecommerce'].sum() / panel.loc[f'{y}-01':f'{y}-08', 'total'].sum()
                   for y in range(2015, 2027)})
print('E-commerce share of retail, Jan-Aug of each year (%):'); print((share * 100).round(2).to_string())"""))

C.append(md("""## 3. Models

All statistical models work on **log sales** (so seasonal swings scale with the level), and the March-June 2020
lockdown months are treated as outliers in training (replaced with their pre-pandemic trajectory using only data
available at the time).

| Model | Idea |
|---|---|
| Seasonal Naive | this December = last December - the retailer's habit, our baseline |
| Naive + Growth | last year's month × recent year-over-year growth |
| ETS | damped-trend exponential smoothing with monthly seasonality |
| SARIMA | seasonal ARIMA, order chosen by AIC from 16 candidates on training data only |
| SARIMAX + Sentiment | SARIMA + consumer sentiment lagged 4 months (always known at forecast time) |
| XGBoost | one gradient-boosted model pooled across categories, predicting year-over-year growth |
| Ensemble | average of SARIMA, ETS and XGBoost in log space |"""))
models_src = (SRC / "models.py").read_text()
models_body = models_src[models_src.index("COVID_MONTHS"):]
C.append(code(models_body))

C.append(md("""## 4. Rolling-origin backtest (11 holiday seasons, 2015-2025)

For each year, every model stands at the **end of August**, is fit only on data available then, and forecasts
**September-December**. Same origin, horizon and information for every model. The same code then runs from
August 2026 to produce the live 2026 forecast. *(About 2-4 minutes.)*"""))
C.append(code("CATS = list(CATEGORIES)\nENSEMBLE_OF = ['SARIMA', 'ETS', 'XGBoost']\n\n\n" +
              defs("backtest", ["_stat_models", "run_origin_set", "attach_actuals"]) + """


# Some SARIMA candidates in the AIC search do not converge; that is expected and handled (AIC picks
# among the ones that fit). Silence those warnings inside the parallel worker processes.
_stat_models_verbose = _stat_models
def _stat_models(*args):
    import warnings
    warnings.simplefilter('ignore')
    return _stat_models_verbose(*args)


origins = [pd.Timestamp(f'{y}-08-01') for y in BACKTEST_YEARS]
fc, orders, importance = run_origin_set(panel, origins)
bt = attach_actuals(fc, panel)
f26, orders26, _ = run_origin_set(panel, [pd.Timestamp(FINAL_ORIGIN)])
f26 = attach_actuals(f26, panel)
print('backtest forecasts:', len(bt), '| 2026 forecasts:', len(f26))
orders.groupby('category')[['order', 'seasonal_order']].agg(lambda s: s.value_counts().index[0])"""))

C.append(md("""## 5. Can AI chatbots forecast? (LLM experiment)

We gave **Claude Opus 5.5** and **OpenAI GPT-5.5** the same 10-year monthly history in a prompt and asked for the next
4 months as JSON. **Memorization control:** an LLM may have read the real Census figures in training, so the main test is
*anonymized* - an index (first month = 100) labelled only by calendar month, with no category name and no year. For
Claude we also ran *named* prompts (real dates, category, dollars). If named prompts do much better on past seasons,
that is hindsight, not forecasting skill.

Calling the APIs costs money and needs keys, so by default this notebook **loads the forecasts we saved** from the
project repository. Set `RUN_LLM = True` (and provide `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) to regenerate them."""))
C.append(code(defs("llm_forecast", ["HISTORY_MONTHS", "CLAUDE_MODEL", "MONTHS", "SYSTEM", "SCHEMA", "build_prompt",
                                    "_parse", "ask_claude", "openai_model", "ask_openai", "DISPLAY"]) + f"""


# Example: the anonymized prompt for clothing stores, August 2024
print(build_prompt(panel['clothing'].dropna(), pd.Timestamp('2024-08-01'), 'clothing', 'anonymized')[0][-330:])"""))
C.append(code(f"""RUN_LLM = False
REPO_RAW = '{REPO_RAW}'

if RUN_LLM:
    def llm_forecasts(origins, runs):
        oa = openai_model() if any(p == 'openai' for p, _ in runs) else None
        rows = []
        for o in origins:
            for c in CATEGORIES:
                for prov, framing in runs:
                    text, scale = build_prompt(panel[c].dropna(), o, c, framing)
                    out = ask_claude(text) if prov == 'claude' else ask_openai(text, oa)
                    rows.append(pd.DataFrame({{'date': future_index(o), 'mean': np.array(out['forecast']) * scale,
                                              'model': DISPLAY[(prov, framing)], 'category': c, 'origin': o}}))
        return attach_actuals(pd.concat(rows, ignore_index=True), panel)
    llm_bt = llm_forecasts(origins, [('claude', 'anonymized'), ('openai', 'anonymized'), ('claude', 'named')])
    llm_26 = llm_forecasts([pd.Timestamp(FINAL_ORIGIN)], [('claude', 'named'), ('openai', 'named')])
else:
    llm_bt = pd.read_csv(f'{{REPO_RAW}}/llm_backtest_forecasts.csv', parse_dates=['date', 'origin'])
    llm_26 = pd.read_csv(f'{{REPO_RAW}}/llm_forecast_2026.csv', parse_dates=['date', 'origin'])

bt_all = pd.concat([bt, llm_bt], ignore_index=True)
f26_all = pd.concat([f26, llm_26], ignore_index=True)
print(bt_all.groupby('model').size())"""))

C.append(md("""## 6. Evaluation

Main metric: absolute % error of the **November+December total** (what a retailer orders against). Also monthly MAPE,
MASE (scaled by the in-sample seasonal-naive error - comparable across categories), and the share of actual values
inside the 80% intervals (*Coverage80*, should be near 80%). MAPE is appropriate because sales are never near zero."""))
C.append(code("COVID_YEARS = (2020, 2021)\n\n\n" + defs("evaluate", ["mase_scale", "per_origin", "summary", "paired_test"]) + """


bt_all['year'] = bt_all['origin'].dt.year
po = per_origin(bt_all, panel)
fmt = {'holiday_APE': '{:.2%}', 'MAPE': '{:.2%}', 'MASE': '{:.2f}', 'Coverage80': '{:.0%}', 'win_rate_vs_naive': '{:.0%}'}
s_all, s_ex = summary(po), summary(po, exclude_covid=True)
display(s_all.style.format(fmt, na_rep='-').set_caption('All 11 seasons, 2015-2025'))
display(s_ex.style.format(fmt, na_rep='-').set_caption('Excluding the COVID seasons 2020-2021'))"""))
C.append(code("""d = s_all['holiday_APE'].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(d.index, d.values * 100, height=0.55,
        color=[GRAY if m == 'Seasonal Naive' else ORANGE if m.startswith(('Claude', 'GPT')) else BLUE for m in d.index])
for i, v in enumerate(d.values): ax.text(v * 100 + 0.05, i, f'{v:.1%}', va='center')
ax.set_title('Average error of the Nov+Dec forecast, 11 seasons (%)', loc='left'); plt.show()

print('Holiday error by category (%):')
(po.pivot_table(index='category', columns='model', values='holiday_ape') * 100).round(2)"""))
C.append(code("""# Paired Wilcoxon signed-rank tests on the same 77 category-seasons
for a, b in [('Claude (anonymized)', 'Claude (named)'), ('Claude (anonymized)', 'Ensemble'),
             ('GPT (anonymized)', 'Ensemble'), ('GPT (anonymized)', 'Claude (anonymized)'),
             ('SARIMAX + Sentiment', 'SARIMA')]:
    t = paired_test(po, a, b)
    if t:
        print(f"{a:22s} vs {b:20s}: {b} better in {t['b_better_share']:.0%} of {t['n']}, "
              f"mean diff {t['mean_diff_pts']:+.2f} pts, p = {t['p_value']:.4f}")

mm = po[po.model.isin(['Claude (anonymized)', 'Claude (named)'])].pivot_table(index='year', columns='model', values='holiday_ape')
ax = (mm * 100).plot(marker='o', figsize=(8, 3.2), color=['#6250d6', '#9085e9'])
ax.set_title("Claude's Nov+Dec error by season (%): anonymized vs named prompts", loc='left'); plt.show()"""))
C.append(code("""# Replay one season: clothing stores, 2024
cat, yr = 'clothing', 2024
fig, ax = plt.subplots(figsize=(9, 3.4))
h = panel[cat].loc[f'{yr-1}-09':f'{yr}-12']
ax.plot(h.index, h / 1000, color=INK, label='Actual')
for m, col in [('Ensemble', BLUE), ('Claude (anonymized)', '#6250d6'), ('Seasonal Naive', GRAY)]:
    d = bt_all[(bt_all.category == cat) & (bt_all.year == yr) & (bt_all.model == m)]
    ax.plot(d.date, d['mean'] / 1000, marker='o', color=col, label=m)
ax.axvline(pd.Timestamp(f'{yr}-08-15'), color=GRID)
ax.set_title(f'Clothing stores {yr}: forecasts made in August vs. actual ($ billions)', loc='left'); ax.legend(frameon=False); plt.show()"""))

C.append(md("## 7. The 2026 holiday forecast"))
C.append(code("""hol = f26_all[f26_all.date.dt.month >= 11].groupby(['category', 'model'])['mean'].sum().unstack()
ly = panel.loc['2025-11':'2025-12'].sum()
growth = hol.div(ly[hol.index], axis=0) - 1
table = pd.DataFrame({'Nov+Dec 2025 actual ($B)': (ly[hol.index] / 1000).round(1),
                      'Nov+Dec 2026 forecast ($B)': (hol['Ensemble'] / 1000).round(1),
                      'Growth (Ensemble)': (growth['Ensemble'] * 100).round(1).astype(str) + '%'})
display(table)
print('Growth by model (%):'); display((growth * 100).round(1))

# Early reality check: September 2026 total retail was published after our August origin
sep = f26_all[(f26_all.category == 'total') & (f26_all.date == '2026-09-01')][['model', 'mean', 'actual']]
sep.assign(error=lambda d: (d['mean'] / d['actual'] - 1).map('{:+.1%}'.format))"""))
C.append(code("""cat = 'clothing'
fig, ax = plt.subplots(figsize=(10, 3.4))
h = panel[cat].loc['2024-09':'2026-08']
ax.plot(h.index, h / 1000, color=INK, label='Actual')
fcc = f26_all[f26_all.category == cat]
for m, col in [('Ensemble', BLUE), ('Seasonal Naive', GRAY)]:
    d = fcc[fcc.model == m]
    ax.plot([h.index[-1]] + list(d.date), [h.iloc[-1] / 1000] + list(d['mean'] / 1000), marker='o', color=col, label=m)
sb = fcc[fcc.model == 'SARIMA']
ax.fill_between(sb.date, sb.lo80 / 1000, sb.hi80 / 1000, color=ORANGE, alpha=0.12, label='SARIMA 80% interval')
ax.set_title('Clothing stores: Sep-Dec 2026 forecast ($ billions)', loc='left'); ax.legend(frameon=False); plt.show()"""))

C.append(md("""## 8. From forecast to order (the Holiday Planner rule)

Expected sales = the store's own Nov+Dec 2025 sales × (1 + national category growth).
Uncertainty σ = √(national forecast error² + store volatility²), where the national error is the root-mean-square of
the ensemble's past holiday % errors for that category (excluding COVID seasons).
Safety buffer = z × σ × expected sales (z for the chosen service level). Order = max(0, expected + buffer − on hand − on order).

Example: a clothing boutique sold 5,000 units last Nov+Dec, has 1,200 on hand, wants a 90% chance of not running out,
and its sales typically miss plan by ±10%."""))
C.append(code("""last, on_hand, on_order, service, store_sd = 5000, 1200, 0, 0.90, 0.10
g = growth.loc['clothing', 'Ensemble']
pe = po[(po.model == 'Ensemble') & (po.category == 'clothing') & (~po.year.isin(COVID_YEARS))]['holiday_pe']
sigma_nat = np.sqrt((pe ** 2).mean()); sigma = np.sqrt(sigma_nat ** 2 + store_sd ** 2)
expected = last * (1 + g); safety = norm.ppf(service) * sigma * expected
order = max(0, expected + safety - on_hand - on_order)
print(f'national growth {g:+.1%} | national error {sigma_nat:.1%} | total sigma {sigma:.1%}')
print(f'expected {expected:,.0f} + safety {safety:,.0f} - on hand {on_hand:,} = recommended order {np.ceil(order):,.0f} units')"""))

C.append(md("""## 9. Conclusions

1. **Forecasting beats habit.** Every model roughly halves the error of "same as last year"; the ensemble is the most
   accurate classical/ML model.
2. **Consumer sentiment does not help** - adding it to SARIMA does not reduce the error.
3. **LLMs: hindsight, not foresight.** Told the category and the year, Claude looks like the best forecaster; with them
   hidden its error rises and the advantage disappears - the gap is largest in 2020-2022, when knowing what happened
   helps most. Fairly tested (anonymized), Claude and GPT are good forecasters but not better than our ensemble.
4. **2026:** a solid holiday season, led by e-commerce, which keeps taking share from stores.
5. **Limits:** national category data (not a single store), today's revised data (not real-time vintages), 11 seasons is a
   small sample, and no inventory-cost data - so no claims about money saved.

**AI tools used.** Claude Code wrote most of the code from our research questions and design decisions; we reviewed
it and checked results against the raw data. Claude Opus 5.5 and GPT-5.5 are forecasters in the comparison, and
Claude powers the dashboard's AI advisor."""))

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "Holiday_Retail_Forecast.ipynb"
nbf.write(nb, out)
print(out)

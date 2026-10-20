# Holiday Retail Planner 2026

Forecasting the 2026 U.S. holiday shopping season by retail category, comparing classical, machine-learning and
LLM forecasters on 11 past seasons, and turning the forecast into an order recommendation for small retailers.

UCLA Anderson, *Time Series Analysis and Forecasting* (Dr. William Yu), group project, fall 2026.
Team: Charlotte Zhu, Shihan Wang, Jing Chang, Rachel Liu.

- **Live dashboard:** _link added after deployment_
- **Data:** U.S. Census Bureau Monthly Retail Trade (not seasonally adjusted) and University of Michigan Consumer
  Sentiment, both via [FRED](https://fred.stlouisfed.org/). Downloaded automatically; no account needed.

## Project layout

```
src/
  config.py         series, backtest years, paths
  data.py           download FRED series, build data/processed/monthly_panel.csv, data-quality checks
  models.py         Seasonal Naive, Naive + Growth, ETS, SARIMA, SARIMAX + Sentiment, XGBoost
  backtest.py       rolling-origin backtest 2015-2025 + the 2026 forecast
  llm_forecast.py   Claude and GPT forecasts (anonymized and named prompts), cached
  evaluate.py       accuracy metrics and summary tables
app/
  streamlit_app.py  the dashboard (reads outputs/, fits nothing)
  data_layer.py     loading, tables and the AI advisor's fact sheet
  about.md          objectives, methods, validation, usage, AI tools
notebooks/
  analysis.ipynb    walk-through of the whole analysis with charts
outputs/            every forecast, metric and LLM response produced by the code
```

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.data          # download data (add --refresh for the latest release)
.venv/bin/python -m src.backtest      # ~2 minutes: all classical/ML models, 11 seasons + 2026
.venv/bin/python -m src.llm_forecast  # needs API keys in .env (see .env.example); cached in outputs/llm_cache
.venv/bin/python -m src.evaluate      # metrics tables
.venv/bin/python -m streamlit run app/streamlit_app.py
```

The LLM responses used in the results are saved in `outputs/llm_cache/`, so `src.llm_forecast` reproduces them
without calling the APIs again. The dashboard's AI Advisor needs `ANTHROPIC_API_KEY` in `.env` (locally) or in
Streamlit secrets (deployed); everything else works without a key.

## Method in one paragraph

For every year 2015-2025, each model stands at the end of August, is fit only on data available then, and
forecasts September-December. We score the error of the November+December total. Statistical models work on log
sales with the March-June 2020 lockdown months treated as outliers. XGBoost is one model pooled across categories
that predicts year-over-year growth. The ensemble averages SARIMA, ETS and XGBoost. LLMs get the same history in a
prompt; to separate forecasting skill from memorization of published figures, the main LLM test hides the
category and the year (an index labelled only by calendar month).

## AI tools used

- **Claude Code** wrote most of the pipeline, models, dashboard and first drafts of the slides and description from
  the team's research questions and design decisions; the team reviewed the code and checked the results.
- **Claude Opus 5.5** and **OpenAI GPT** are forecasters in the comparison (course Option A).
- **Claude Opus 5.5** powers the dashboard's AI Advisor, restricted to the computed results (course Option B).

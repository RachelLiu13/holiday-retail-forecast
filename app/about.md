### About this project

**Holiday Retail Planner 2026** - UCLA Anderson, *Time Series Analysis and Forecasting* (Dr. William Yu).
Team: Charlotte Zhu, Shihan Wang, Jing Chang, Rachel Liu.

#### Objective
Forecast U.S. retail sales for the 2026 holiday season (November-December) by category, test whether forecasting
beats the retailer's rule of thumb ("same as last year"), test whether AI chatbots can forecast as well as
statistical models, and turn the forecast into an order recommendation a small retailer can use.

#### Data
- **U.S. Census Bureau, Monthly Retail Trade**, via FRED, *not seasonally adjusted*, $ millions, 1992 to August 2026
  (total retail through September 2026). Seven series: total retail excluding food services, clothing, electronics &
  appliances, sporting goods/hobby/book/music, general merchandise, electronic shopping & mail-order (e-commerce),
  and grocery (a control with almost no holiday peak).
- **University of Michigan Consumer Sentiment** (FRED: UMCSENT) as a customer-side indicator.
- Data checks: no missing months or duplicates. The March-June 2020 lockdown months are treated as outliers when
  training models (replaced with their pre-pandemic trajectory using only data available at the time).

#### Methods
| Model | Idea |
|---|---|
| Seasonal Naive | This December = last December (the retailer's habit; our baseline) |
| Naive + Growth | Last year's month × recent year-over-year growth |
| ETS | Exponential smoothing, damped trend + monthly seasonality, on log sales |
| SARIMA | Seasonal ARIMA on log sales; order chosen by AIC from 16 candidates using training data only |
| SARIMAX + Sentiment | SARIMA plus consumer sentiment lagged 4 months (so it is always known at forecast time) |
| XGBoost | One gradient-boosted model across all categories, predicting year-over-year growth from recent growth, momentum, month, horizon, category and sentiment |
| Ensemble | Average of SARIMA, ETS and XGBoost (in logs) - **our recommended model** |
| Claude / GPT | Large language models given the same history in a prompt and asked for the next 4 months |

**LLM forecasting and memorization.** An LLM may have read the real Census figures during training. We therefore
test LLMs on an **anonymized** version of each series (an index, first month = 100, labelled only by calendar month,
with no category name and no year) and, for Claude, also on the **named** version. A much better named score would
indicate recall rather than forecasting skill. For the 2026 forecast nothing can be memorized, so named prompts are used.

#### Validation
Rolling-origin backtest over **11 holiday seasons (2015-2025)**. Each time, every model is fit only on data through
August and forecasts September-December - the same origin, horizon and information for all models.
Metrics: absolute % error of the **Nov+Dec total** (main metric), monthly MAPE, MAE, RMSE, MASE, and coverage of the
80% intervals. MAPE is appropriate here because sales are never close to zero. Results are shown with and without the
COVID seasons (2020-2021).

#### How to use the dashboard
- **Story** - the headline results.
- **Market** - history, the size of each category's December spike, and e-commerce's share.
- **2026 Forecast** - the forecast by category and model, with SARIMA's 80% interval.
- **Model Battle** - backtest scores; replay any past season; the LLM memorization test.
- **Holiday Planner** - enter your own last-season sales and stock to get an order recommendation with scenarios.
- **AI Advisor** - ask questions in plain English; answers are grounded in the computed results only.

#### Limitations
- National category totals, not a specific store: the planner scales *your* history by the national trend.
- FRED provides today's revised figures; forecasters in past years saw earlier, unrevised numbers.
- 11 seasons is a small sample; differences of a few tenths of a point between the top models are not decisive.
- No inventory-cost data, so we make no claim about money saved.

#### AI tools used
- **Claude Code** (Anthropic) wrote most of the data pipeline, models, backtest, dashboard and first drafts of the
  slides and description, working from our research questions and design decisions; the team reviewed the code,
  checked results against the raw data, and chose what to present.
- **Claude Opus 5.5** and **OpenAI GPT** were used as forecasters in the comparison (Option A).
- **Claude Opus 5.5** powers the AI Advisor (Option B), constrained to the computed results.

Source code: see the GitHub repository linked in our submission.

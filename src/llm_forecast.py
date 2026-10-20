"""Option A: ask large language models to forecast, under the same rules as the other models.

Two prompt framings, chosen to deal with *memorization*: an LLM may have seen the real
2015-2025 Census figures during training, so a backtest that names the series and the year
could reward recall rather than forecasting.

- "anonymized": the model gets 10 years of monthly values rescaled to an index (first
  month = 100), labelled only by calendar month (Jan..Dec), with no category name and no
  year. It cannot look the answer up, so this is the fair test of forecasting skill.
- "named": the same history with real dates, the real series name and dollar units. If the
  named version is much more accurate than the anonymized one on past years, that gap is
  evidence of memorization rather than skill.

For the 2026 forecast there is nothing to memorize, so the named framing is a fair test.

Run:  python -m src.llm_forecast            (backtest + 2026, both providers)
      python -m src.llm_forecast --dry-run  (print one prompt, no API calls)

Keys are read from .env (ANTHROPIC_API_KEY, OPENAI_API_KEY). Responses are cached in
outputs/llm_cache/ so re-runs cost nothing and results are reproducible.
"""
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .config import BACKTEST_YEARS, CATEGORIES, FINAL_ORIGIN, HORIZON, OUTPUTS, ROOT
from .data import load_panel
from .models import future_index

load_dotenv(ROOT / ".env")

CACHE = OUTPUTS / "llm_cache"
HISTORY_MONTHS = 120
CLAUDE_MODEL = "claude-opus-5-5"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SYSTEM = (
    "You are a careful time-series forecaster. You are given a monthly history and must "
    "forecast the next 4 months. Use only the numbers provided. Think about level, trend, "
    "and month-of-year seasonality, then return your forecast in the requested JSON format."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "2-4 sentences on the pattern you used"},
        "forecast": {"type": "array", "items": {"type": "number"}, "description": "exactly 4 values, in order"},
    },
    "required": ["reasoning", "forecast"],
    "additionalProperties": False,
}


def build_prompt(y: pd.Series, origin: pd.Timestamp, category: str, framing: str):
    """Return (prompt text, scale factor to convert the answer back to $ millions)."""
    hist = y.loc[:origin].iloc[-HISTORY_MONTHS:]
    future = future_index(origin)
    if framing == "anonymized":
        scale = hist.iloc[0] / 100.0
        vals = hist / scale
        lines = [f"{MONTHS[d.month - 1]}: {v:.1f}" for d, v in zip(hist.index, vals)]
        targets = ", ".join(MONTHS[d.month - 1] for d in future)
        text = (
            f"Below are {len(hist)} consecutive months of an anonymous monthly sales index "
            f"(first month = 100). Only the calendar month is shown; the series runs oldest "
            f"to newest.\n\n" + "\n".join(lines) +
            f"\n\nForecast the next 4 months ({targets}) on the same index scale."
        )
        return text, scale
    label = CATEGORIES[category]["label"]
    lines = [f"{d:%Y-%m}: {v:,.0f}" for d, v in zip(hist.index, hist)]
    targets = ", ".join(f"{d:%Y-%m}" for d in future)
    text = (
        f"Below are monthly U.S. retail sales for '{label}' (U.S. Census Bureau Monthly "
        f"Retail Trade, not seasonally adjusted, millions of dollars), oldest to newest.\n\n"
        + "\n".join(lines) +
        f"\n\nForecast the next 4 months ({targets}) in millions of dollars."
    )
    return text, 1.0


def _cache_path(provider, model, text):
    key = hashlib.sha256(f"{provider}|{model}|{SYSTEM}|{text}".encode()).hexdigest()[:24]
    return CACHE / f"{provider}_{key}.json"


def _parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0) if m else raw)
    fc = [float(v) for v in data["forecast"]]
    if len(fc) != HORIZON or not all(np.isfinite(fc)) or min(fc) <= 0:
        raise ValueError(f"bad forecast: {fc}")
    return {"reasoning": data.get("reasoning", ""), "forecast": fc}


def ask_claude(text: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": text}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refused: {resp.stop_details}")
    raw = next(b.text for b in resp.content if b.type == "text")
    return {**_parse(raw), "model": resp.model,
            "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}}


def openai_model() -> str:
    """Use OPENAI_MODEL if set, else the newest flagship 'gpt-N' / 'gpt-N.M' model on the account."""
    if os.getenv("OPENAI_MODEL"):
        return os.environ["OPENAI_MODEL"]
    from openai import OpenAI
    models = [m for m in OpenAI().models.list() if re.fullmatch(r"gpt-\d+(\.\d+)?", m.id)]
    return max(models, key=lambda m: m.created).id


def ask_openai(text: str, model: str) -> dict:
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": text + "\n\nReturn JSON: {\"reasoning\": str, \"forecast\": [4 numbers]}"}],
        response_format={"type": "json_object"},
    )
    return {**_parse(resp.choices[0].message.content), "model": resp.model,
            "usage": {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}}


def ask(provider: str, text: str, oa_model: str | None) -> dict:
    model = CLAUDE_MODEL if provider == "claude" else oa_model
    path = _cache_path(provider, model, text)
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(8):
        try:
            out = ask_claude(text) if provider == "claude" else ask_openai(text, model)
            break
        except Exception as e:  # retries cover rate limits and the rare malformed answer
            if attempt == 7:
                raise
            print(f"  retry {provider} ({type(e).__name__}: {str(e)[:120]})", flush=True)
            time.sleep(min(90, 10 * 2 ** attempt))  # back off: 10s, 20s, 40s, then 90s
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**out, "prompt": text}, indent=1))
    return out


# Which runs to make. Anonymized is the main, fair comparison for both providers; the
# named backtest (Claude only, to keep cost down) measures how much memorization helps.
RUNS = [("claude", "anonymized"), ("openai", "anonymized"), ("claude", "named")]
DISPLAY = {("claude", "anonymized"): "Claude (anonymized)",
           ("openai", "anonymized"): "GPT (anonymized)",
           ("claude", "named"): "Claude (named)",
           ("openai", "named"): "GPT (named)"}


def run(origins, runs, panel, oa_model, workers=None):
    # GPT-5.5 reasons at length, so fewer parallel requests keep it under the tokens-per-minute limit
    workers = workers or (3 if any(p == "openai" for p, _ in runs) else 8)
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(p, f, c, o) for o in origins for c in CATEGORIES for p, f in runs]
    prompts = [build_prompt(panel[c].dropna(), o, c, f) for p, f, c, o in jobs]
    done = [0]

    def one(i):
        out = ask(jobs[i][0], prompts[i][0], oa_model)
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(jobs):
            print(f"  {done[0]}/{len(jobs)} prompts done", flush=True)
        return out

    # requests are independent, so run them concurrently; results keep the job order
    with ThreadPoolExecutor(workers) as pool:
        outs = list(pool.map(one, range(len(jobs))))

    rows, notes = [], []
    for (prov, framing, cat, origin), (_, scale), out in zip(jobs, prompts, outs):
        vals = np.array(out["forecast"]) * scale
        idx = future_index(origin)
        rows.append(pd.DataFrame({"date": idx, "mean": vals, "model": DISPLAY[(prov, framing)],
                                  "category": cat, "origin": origin}))
        notes.append({"model": DISPLAY[(prov, framing)], "category": cat, "origin": origin.date(),
                      "llm_version": out.get("model"), "reasoning": out["reasoning"]})
    fc = pd.concat(rows, ignore_index=True)
    fc["h"] = (fc["date"].dt.year - fc["origin"].dt.year) * 12 + fc["date"].dt.month - fc["origin"].dt.month
    return fc, pd.DataFrame(notes)


def main():
    panel = load_panel()
    if "--dry-run" in sys.argv:
        o = pd.Timestamp("2024-08-01")
        for fr in ("anonymized", "named"):
            print(f"===== {fr} =====\n{build_prompt(panel['clothing'].dropna(), o, 'clothing', fr)[0][-600:]}\n")
        return
    # --only claude / --only openai runs one provider (e.g. while the other account is unfunded);
    # cached responses make it safe to run the providers separately and then together.
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    runs = [r for r in RUNS if only in (None, r[0])]
    final_runs = [r for r in [("claude", "named"), ("openai", "named")] if only in (None, r[0])]
    oa_model = openai_model() if any(p == "openai" for p, _ in runs) else None
    print("Claude model:", CLAUDE_MODEL, "| OpenAI model:", oa_model)
    long = panel[list(CATEGORIES)].stack().rename("actual").reset_index().rename(columns={"level_1": "category"})

    origins = [pd.Timestamp(f"{y}-08-01") for y in BACKTEST_YEARS]
    bt, bt_notes = run(origins, runs, panel, oa_model)
    bt.merge(long, on=["date", "category"], how="left").to_csv(OUTPUTS / "llm_backtest_forecasts.csv", index=False)

    f26, f26_notes = run([pd.Timestamp(FINAL_ORIGIN)], final_runs, panel, oa_model)
    f26.merge(long, on=["date", "category"], how="left").to_csv(OUTPUTS / "llm_forecast_2026.csv", index=False)
    pd.concat([bt_notes, f26_notes]).to_csv(OUTPUTS / "llm_reasoning.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()

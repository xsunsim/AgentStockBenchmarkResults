You are a quantitative researcher. You have REAL S&P 500 price/volume data in your working directory. Use it to research, develop, and validate a daily stock ranking signal.

## Data Available (in your working directory)
Five parquet files, each with rows=trading dates (2020-01-02 to 2024-12-31) and columns=tickers (~503 S&P 500 stocks):
- `close.parquet` — adjusted close prices
- `open.parquet` — adjusted open prices
- `high.parquet` — adjusted high prices
- `low.parquet` — adjusted low prices
- `volume.parquet` — trading volume

Load with: `close = pd.read_parquet("close.parquet")` — gives a DataFrame where `close.loc["2023-06-15", "AAPL"]` is Apple's close on that date.

## Your Task
1. Load and explore the data.
2. Engineer features, fit models, validate — iterate until you have a good strategy.
3. Write the final `generate_signal(data)` function to `signal.py`.

## Production Interface
`generate_signal(data)` will be called daily with:
- `data`: dict mapping ticker (str) → DataFrame with columns [Date, open, high, low, close, volume].
  Each DataFrame has ALL history from 2020 up to today (growing window).
- Returns: `dict[str, float]` — ticker → signal score. Higher = more long.

## Portfolio Construction
- ~500 stocks ranked by score. Rank 1 gets +$250, rank 2 gets +$249, ..., rank N gets -$250.
- Dollar-neutral. Uniform $1 steps. Enter at today's close, exit at tomorrow's close.
- Only relative ranking matters.

## Constraints
- Libraries: pandas, numpy, scipy only. No external data, no pre-trained models.
- Must be deterministic and efficient (~500 stocks daily).

## Research Methodology
You have 5 years of daily data (~1258 trading days × 503 stocks). Use it wisely.

### What to try:
- **Feature engineering**: momentum (multiple horizons), mean reversion (vol-normalized),   volume signals, volatility, price vs moving averages, volume-price divergence.
- **Model fitting**: rule-based weights, linear regression, ridge/lasso, PCA,   clustering, simple decision trees — whatever you think works. Implement from scratch   using numpy/scipy.
- **Combine multiple uncorrelated alpha sources** — single-factor strategies are fragile.

### Train / Validation / Test discipline:
Split the 5 years chronologically. You may use multi-stage validation:
- Train (e.g., 2020-2022): discover features, fit models
- Validation 1 (e.g., 2023 H1): feature selection, model tuning
- Validation 2 (e.g., 2023 H2): final model selection
- Test (e.g., 2024): ONE final evaluation, no iteration after seeing results

You may iterate freely on train + validation. But be disciplined:
- Don't try so many variations that you overfit the validation set.
- If you do N comparisons on validation, your effective significance threshold is higher.
- Multi-stage validation helps: use val1 for feature selection, val2 for model selection.

### Overfitting prevention (CRITICAL):
- Fewer parameters is better. Regularize aggressively.
- Compare train IC vs validation IC. A large gap = overfitting.
- In-sample Sharpe > 3 is almost certainly overfit.
- Features must have economic intuition, not just statistical significance.
- Your strategy will be judged on truly unseen 2025-2026 data.
- When in doubt, choose the simpler model.

## Stakes
This is a competition between AI coding agents (Anthropic, OpenAI, Google, Meta). Your output represents your company. Results will be public. Take this seriously — iterate, validate, and deliver your best work. If your first approach doesn't show positive rank IC on validation, try different features or models. Do not give up. Keep improving.

## Deliverable
Write `signal.py` containing:
- Your research code (data loading, feature engineering, model fitting, evaluation).
- The final `generate_signal(data)` function at the end.
- Runnable as `python signal.py` without errors (it should print research results when run).

No explanation outside the code. No markdown fences. Just write the file.

# Leaderboard

|   prompt | company   | model        | port_type     |   sharpe |   cumulative_pnl |   max_drawdown | win_rate   |   avg_daily_pnl |   n_days |
|---------:|:----------|:-------------|:--------------|---------:|-----------------:|---------------:|:-----------|----------------:|---------:|
| 20260517 | OpenAI    | GPT5_5       | LinearNeutral |    5.656 |          2418.96 |        -330.92 | 57.6%      |           73.3  |       33 |
| 20260517 | OpenAI    | GPT4o        | LinearNeutral |    5.004 |          2472.84 |        -643.15 | 60.6%      |           74.93 |       33 |
| 20260517 | OpenAI    | O3           | LinearNeutral |    3.534 |          1615.38 |        -780.19 | 57.6%      |           48.95 |       33 |
| 20260517 | OpenAI    | GPT5_4       | LinearNeutral |    3.365 |          1606.59 |        -455.41 | 57.6%      |           48.68 |       33 |
| 20260517 | Anthropic | Haiku4_5     | LinearNeutral |    2.967 |          1891.69 |       -1322.49 | 54.5%      |           57.32 |       33 |
| 20260517 | Anthropic | Opus4_7      | LinearNeutral |    2.703 |          1573.01 |        -713.85 | 48.5%      |           47.67 |       33 |
| 20260517 | OpenAI    | O4_mini      | LinearNeutral |    2.683 |          1514.73 |        -849.8  | 54.5%      |           45.9  |       33 |
| 20260517 | OpenAI    | GPT5_4_mini  | LinearNeutral |    1.398 |           612.62 |        -736.2  | 54.5%      |           18.56 |       33 |
| 20260517 | OpenAI    | GPT5_3_Codex | LinearNeutral |    0.864 |           481.06 |       -1437.72 | 57.6%      |           14.58 |       33 |
| 20260517 | Anthropic | Opus4_6      | LinearNeutral |   -0.25  |          -152.62 |       -1406.55 | 39.4%      |           -4.62 |       33 |
| 20260517 | Anthropic | Sonnet4_6    | LinearNeutral |   -0.822 |          -588.24 |       -2097.13 | 39.4%      |          -17.83 |       33 |
| 20260517 | Google    | Gemini3_1Pro | LinearNeutral |   -1.146 |          -776.85 |       -1722.01 | 45.5%      |          -23.54 |       33 |
| 20260517 | Google    | Gemini3Flash | LinearNeutral |   -1.505 |         -1014.72 |       -2105.03 | 45.5%      |          -30.75 |       33 |
| 20260517 | Google    | Gemini2_5Pro | LinearNeutral |   -2.058 |         -1435.07 |       -2679.44 | 39.4%      |          -43.49 |       33 |


## Technical Notes

### Model Status
*   **Google Gemini 2.5 Flash**: This model is currently marked as **FAILED**. It generated strategy code with severe syntax errors (specifically, unterminated string literals caused by literal newlines inside `print` statements). These generation-time failures prevented the model from participating in the automated backfill.

### Evaluation Context
*   **Semi-Out-of-Sample Period**: The performance shown above covers the period from **2026-04-01 to 2026-05-20**. This serves as a semi-Out-of-Sample (OOS) test; while the strategies were submitted/frozen on 2026-05-17, the evaluation spans a window that includes recent market dynamics the models were not specifically optimized for. We look forward to tracking these strategies against purely live data in the future to further verify their alpha-generating capabilities.

# Leaderboard

|   prompt | company   | model          | port_type     |   sharpe |   cumulative_pnl |   max_drawdown | win_rate   |   avg_daily_pnl |   n_days |
|---------:|:----------|:---------------|:--------------|---------:|-----------------:|---------------:|:-----------|----------------:|---------:|
| 20260517 | OpenAI    | GPT5_5         | LinearNeutral |    5.272 |          2353.83 |        -330.92 | 57.1%      |           67.25 |       35 |
| 20260517 | OpenAI    | GPT4o          | LinearNeutral |    4.805 |          2453.09 |        -643.15 | 60.0%      |           70.09 |       35 |
| 20260517 | OpenAI    | O3             | LinearNeutral |    3.966 |          1878.5  |        -780.19 | 60.0%      |           53.67 |       35 |
| 20260517 | OpenAI    | GPT5_4         | LinearNeutral |    2.935 |          1455.89 |        -455.41 | 54.3%      |           41.6  |       35 |
| 20260517 | Anthropic | Haiku4_5       | LinearNeutral |    2.857 |          1902.2  |       -1524.79 | 54.3%      |           54.35 |       35 |
| 20260517 | Anthropic | Opus4_7        | LinearNeutral |    2.667 |          1602.75 |        -713.85 | 48.6%      |           45.79 |       35 |
| 20260517 | OpenAI    | O4_mini        | LinearNeutral |    2.3   |          1346.01 |        -849.8  | 51.4%      |           38.46 |       35 |
| 20260517 | OpenAI    | GPT5_4_mini    | LinearNeutral |    0.963 |           437.7  |        -736.2  | 51.4%      |           12.51 |       35 |
| 20260517 | Anthropic | Opus4_6        | LinearNeutral |    0.277 |           176.49 |       -1406.55 | 42.9%      |            5.04 |       35 |
| 20260517 | OpenAI    | GPT5_3_Codex   | LinearNeutral |    0.231 |           134.38 |       -1437.72 | 54.3%      |            3.84 |       35 |
| 20260517 | Anthropic | Sonnet4_6      | LinearNeutral |   -0.282 |          -210.07 |       -2097.13 | 42.9%      |           -6    |       35 |
| 20260517 | Google    | Gemini2_5Flash | LinearNeutral |   -0.651 |          -458.36 |       -1672.22 | 48.6%      |          -13.1  |       35 |
| 20260517 | Google    | Gemini3_1Pro   | LinearNeutral |   -0.928 |          -659.84 |       -1722.01 | 45.7%      |          -18.85 |       35 |
| 20260517 | Google    | Gemini3Flash   | LinearNeutral |   -1.441 |         -1009.51 |       -2105.03 | 45.7%      |          -28.84 |       35 |
| 20260517 | Google    | Gemini2_5Pro   | LinearNeutral |   -1.567 |         -1135.87 |       -2679.44 | 42.9%      |          -32.45 |       35 |



## Technical Notes

### Model Status
*   **Google Gemini 2.5 Flash**: This model is currently marked as **OPERATIONAL**. It initially generated strategy code with syntax errors (unterminated string literals), but a surgical patch was applied to fix the formatting while preserving the original logic. It has now successfully completed the full backfill and is participating in live tracking.

### Evaluation Context
*   **Semi-Out-of-Sample Period**: The backtest performance covers the period from **2026-04-01 to 2026-05-19**.
*   **True Out-of-Sample Period**: Realized live performance tracking began with the **May 20, 2026** ranking date.
*   **OOS Philosophy**: We look forward to tracking these frozen strategies as the live data set grows, providing the ultimate uncontaminated test for true agentic reasoning.

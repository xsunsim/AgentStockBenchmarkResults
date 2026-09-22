# Recent Strong Models Daily PnL Report

Scored through ranking date 2026-09-17.

## Summary

| strategy_id                                   | model               |   sharpe |   cumulative_pnl |   max_drawdown |   win_rate |   avg_daily_pnl |   n_days |
|:----------------------------------------------|:--------------------|---------:|-----------------:|---------------:|-----------:|----------------:|---------:|
| 20260517__OpenAI__O3__LinearNeutral           | OpenAI O3           |    3.44  |          7621.47 |       -1325.56 |      0.598 |           65.14 |      117 |
| 20260517__Anthropic__Sonnet4_6__LinearNeutral | Anthropic Sonnet4.6 |    3.322 |         10708.3  |       -2097.13 |      0.573 |           91.52 |      117 |
| 20260517__Anthropic__Opus4_6__LinearNeutral   | Anthropic Opus4.6   |    3.014 |          8058.9  |       -1406.55 |      0.573 |           68.88 |      117 |
| 20260517__OpenAI__GPT5_5__LinearNeutral       | OpenAI GPT5.5       |    2.558 |          5475.87 |       -1705.99 |      0.624 |           46.8  |      117 |
| 20260517__OpenAI__GPT5_4__LinearNeutral       | OpenAI GPT5.4       |    2.554 |          5378.46 |       -1262.96 |      0.573 |           45.97 |      117 |
| 20260517__OpenAI__GPT5_4_mini__LinearNeutral  | OpenAI GPT5.4.mini  |    2.047 |          3563.25 |       -1033.53 |      0.581 |           30.46 |      117 |
| 20260517__Anthropic__Opus4_8__LinearNeutral   | Anthropic Opus4.8   |    1.963 |          4826.5  |       -1654.33 |      0.564 |           41.25 |      117 |
| 20260517__Google__Gemini3_1Pro__LinearNeutral | Google Gemini3.1Pro |    1.902 |          6019.48 |       -1908.14 |      0.521 |           51.45 |      117 |

## Pending New 20260921 Models

These models are integrated as strategy artifacts but have no daily PnL file yet:
- `20260921__Anthropic__Opus4_8__LinearNeutral`
- `20260921__Anthropic__Opus5__LinearNeutral`
- `20260921__Anthropic__Sonnet5__LinearNeutral`
- `20260921__OpenAI__GPT5_6_Sol__LinearNeutral`

## Latest Daily PnL

| strategy_label      | strategy_id                                   |   total_pnl |   cumulative_pnl |   long_pnl |   short_pnl |
|:--------------------|:----------------------------------------------|------------:|-----------------:|-----------:|------------:|
| OpenAI O3           | 20260517__OpenAI__O3__LinearNeutral           |      221.92 |          7621.47 |     291.84 |      -69.92 |
| Anthropic Sonnet4.6 | 20260517__Anthropic__Sonnet4_6__LinearNeutral |     -130.51 |         10708.3  |     120.8  |     -251.31 |
| Anthropic Opus4.6   | 20260517__Anthropic__Opus4_6__LinearNeutral   |       44.98 |          8058.9  |     193.48 |     -148.49 |
| OpenAI GPT5.5       | 20260517__OpenAI__GPT5_5__LinearNeutral       |       62.15 |          5475.87 |     198.97 |     -136.81 |
| OpenAI GPT5.4       | 20260517__OpenAI__GPT5_4__LinearNeutral       |      -93.26 |          5378.46 |      95.99 |     -189.26 |
| OpenAI GPT5.4.mini  | 20260517__OpenAI__GPT5_4_mini__LinearNeutral  |       24.14 |          3563.25 |     149.67 |     -125.53 |
| Anthropic Opus4.8   | 20260517__Anthropic__Opus4_8__LinearNeutral   |     -189.97 |          4826.5  |      97.33 |     -287.3  |
| Google Gemini3.1Pro | 20260517__Google__Gemini3_1Pro__LinearNeutral |     -297.57 |          6019.48 |     113.95 |     -411.52 |

Detailed daily rows are in `recent_strong_models_daily_pnl.csv`; wide daily/cumulative columns are in `recent_strong_models_daily_pnl_wide.csv`.

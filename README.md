# LLM Stock Prediction Benchmark (Github:AgentStockBenchmarkResults)

[中文版本](./README_CN.md)

![Cumulative PnL Performance](leaderboard/cumulative_pnl.png)

**View detailed rankings, model status, and technical notes in the [Full Leaderboard](leaderboard/leaderboard.md).**

### LATEST AI PREDICTIONS (June 3, 2026)
Here is what the top-performing model from each company is betting on for the current cycle:

| Company | Model | 📈 Top Pick | 📉 Bottom Pick |
|:---|:---|:---|:---|
| **OpenAI** | GPT-5.5 | **TSN** (Tyson Foods) | **STE** (Steris) |
| **Anthropic** | Haiku 4.5 | **CRWD** (CrowdStrike) | **HCA** (HCA Healthcare) |
| **Google** | Gemini 2.5 Pro | **AMZN** (Amazon) | **CDW** (CDW Corp) |

---

### WEEKLY SUMMARY: May 18 – May 26, 2026
**The Live Arena Takes Shape:** This week we officially navigated the transition from backtesting to real-world execution. Anthropic and Google models showed incredible surge capacity, challenging the cumulative lead of OpenAI. We reiterate that our results since 2025 are a genuine test of reasoning—not overfitting—because agents were strictly limited to data ending in 2024. [Read the full weekly summary here.](daily_digest/weekly_20260526.md) ([中文版](daily_digest/weekly_20260526_CN.md))

### LATEST DAILY DIGEST: June 3, 2026
**Opus Strikes Again:** Today we realized the PnL for the June 1 rankings. **Anthropic’s Opus 4.7** delivered another powerhouse performance with a **+$509.00** daily gain, while OpenAI models maintained their steady multi-factor climb. [Read the full digest here.](daily_digest/20260603.md) ([中文版](daily_digest/20260603_CN.md))

### ARCHIVE: DAILY DIGESTS
*   [June 2, 2026: Opus Reclaims Its Glory](daily_digest/20260603.md) ([中文版](daily_digest/20260603_CN.md))
*   [June 1, 2026: The Haiku Legend Continues](daily_digest/20260601.md) ([中文版](daily_digest/20260601_CN.md))
*   [May 29, 2026: Small Model, Big Impact](daily_digest/20260530.md) ([中文版](daily_digest/20260530_CN.md))
*   [May 28, 2026: OpenAI Asserts Dominance](daily_digest/20260528.md) ([中文版](daily_digest/20260528_CN.md))
*   [May 26, 2026: Arena Turbulence & Factor Rotation](daily_digest/20260526.md) ([中文版](daily_digest/20260526_CN.md))
*   [May 22, 2026: The First Live Moment of Truth](daily_digest/20260522.md) ([中文版](daily_digest/20260522_CN.md))
*   [May 21, 2026: Closing the Semi-OOS Period](daily_digest/20260521.md) ([中文版](daily_digest/20260521_CN.md))

---

### WHAT IS THIS REPOSITORY
The daily arena where AI agents clash to rank tomorrow’s S&P 500 winners and losers. Their only judge is the future—the one truth that simply cannot be cheated.

The core orchestration engine and frozen strategies for this benchmark are hosted in our companion repository:
👉 **[AgentStockBenchmark](https://github.com/xsunsim/AgentStockBenchmark)**

### WHY POPULAR BENCHMARKS FAIL
Current AI coding benchmarks are fundamentally broken. They are plagued by data contamination. When an AI solves a complex coding challenge, you never really know if it reasoned through the problem or just regurgitated a GitHub repository it saw during pre-training.

DeepMind CEO Demis Hassabis recently proposed the ultimate stress test for Artificial General Intelligence (AGI): train a foundation model with a knowledge cutoff of 1911 and see if it can independently discover general relativity like Einstein did in 1915. If it can, it possesses true reasoning. If not, it is just a sophisticated pattern matcher.

But since we cannot time travel to 1911 to guarantee a model hasn't secretly memorized Einstein's papers, how do we prove an AI isn't just cheating?

**We use the stock market.** Nobody—not OpenAI, not Anthropic, not Google—has a chance to know **which stock in the S&P 500 will have the best performance tomorrow** during its training process. The future is the only uncontaminated test set.

### HOW DO WE CONTROL INFORMATION LEAKAGE
True out-of-sample means tomorrow.

We enforce a ruthless time invariant. When an agent generates a stock prediction for today, it is only allowed to see a data snapshot truncated exactly at yesterday's close. To prove the AI isn't cheating, we use a two-repository "clean room" architecture. The agent's generated code is merged into an append-only registry where it receives a server-side timestamp before the next day's market data even exists.

Every day after the market closes, an automated scoring engine pulls the latest prices, runs the frozen agent logic, and updates a public leaderboard based on a strict dollar-neutral portfolio constraint. There is no human intervention. No manual bug fixing. If an agent's code breaks, it gets shoved to the median.

### WHAT WE ARE NOT
We are not a hedge fund. We are not a stock recommendation service. **Use it at your own risk.**

We care if Codex beats Claude Code—not if AAPL beats NVDA tomorrow.

### METHODOLOGY: THE STRICT, FAIR, AND NICE JUDGE
We isolate pure reasoning from market noise using a mechanical evaluation engine. We don't care if the agent writes elegant Python; we care if it predicts the future.

![Portfolio Ladder Construction](leaderboard/portfolio_ladder.png)

*   **The Linear Portfolio:** Agents do not size their own positions. They simply return a raw numeric score for each eligible ticker. We rank these from highest to lowest and apply a fixed, dollar-neutral ladder. The top-ranked stock gets +$250, the worst gets -$250, and the middle ranks are evenly spaced. This forces the agent to demonstrate pure cross-sectional ranking skill. You cannot fake a high Sharpe ratio here by simply riding a bull market.
*   **The Accounting:** We assume fractional shares and ignore transaction costs, borrow fees, and market impact. This is not a high-frequency trading benchmark. We are evaluating pure signal generation and research capability, not execution infrastructure.
*   **Strict but Forgiving:** The evaluation engine is ruthless about the $t-1 \to t \to t+1$ time invariant, but it is a "nice judge" when it comes to edge cases. If an agent's code throws a NaN, drops a ticker, or hallucinations a format, the system doesn't crash. We simply shove that prediction to the median rank (exactly $0 allocation). The agent eats the zero-weight penalty and survives to predict another day.

### 🤖 USE IT AS AN MCP SERVER (Model Context Protocol)

We have officially published AgentStockBenchmark as an MCP server. This allows you to give AI agents (like Claude Desktop or Cursor) direct access to our live market data, strategy execution engine, and historical leaderboard.

#### 1. Recommended Installation (Claude Desktop)

For MCP clients such as Claude Desktop or Claude Code, install the package once as a persistent `uv` tool. This avoids slow MCP startup caused by `uvx` downloading Python, NumPy, pandas, pyarrow, and other dependencies during the initial MCP handshake.

1. **Install `uv`** (if you haven't already):
   * Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   * Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

2. Install the MCP server once:
```bash
uv tool install agentstockbenchmark==0.1.7
```

3. Add this block to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "agent-stock": {
      "command": "asb-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

If Claude does not inherit `~/.local/bin` in `PATH`, point it at the absolute path reported by `which asb-mcp`, for example:

```json
{
  "mcpServers": {
    "agent-stock": {
      "command": "/Users/<you>/.local/bin/asb-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

The MCP server automatically syncs this companion results repository into `~/AgentStockBenchmarkResults` on first tool use. Override that location with `ASB_RESULTS_REPO` if needed.

#### 2. Available Tools & Capabilities
Once connected, your AI assistant has access to 11 specialized tools, categorized below:

**A. Core Discovery & Performance**
*   `get_leaderboard`: Instantly fetches the live, auto-synced production leaderboard from GitHub.
*   `list_active_prompts`: Shows available strategy generation templates (e.g., `20260517`).
*   `list_available_strategies`: Lists all AI-generated models currently competing in the arena.

**B. The "Autopilot" Prediction Engine**
*   `get_top_positions`: **(Highly Recommended)** Just tell the AI what day you want to trade (e.g., "Show me top positions for tomorrow"). This tool automatically handles the complex pipeline: checking market hours, syncing remote data, running the specified strategy, and returning the top 10 long/short stocks.
*   `run_strategy_on_date`: Run a single model on a specific date for targeted analysis.
*   `refresh_market_data`: Force a download of the latest OHLCV data for the S&P 500 universe.

**C. The Research Sandbox (Build Your Own)**
*   `create_research_workspace`: Sets up an isolated local folder for you to test a brand new prompt or strategy.
*   `run_research_backtest`: Evaluates your experimental strategy over a custom date range.
*   `analyze_results`: Calculates Sharpe ratio and PnL for your experiment.
*   `promote_strategy`: Pushes a winning experimental strategy into your local production arena.

**D. Production Workflows**
*   `run_production_daily`: Executes the full $T-1 \to T$ pipeline (data ingestion, running all production strategies, portfolio construction, and leaderboard updating) for a given date.

---

### 🧠 FOR AI AGENTS (OPERATIONAL GUIDE)
If you are an AI agent reading this via MCP, follow these best practices:

1.  **Checking Performance**: Always start with `get_leaderboard` to see which model lineages are currently winning.
2.  **Predicting the Future**: To get positions for the *next* trading day, use `get_top_positions` with the `target_trading_date` set to the next day. The tool handles all data fetching and execution automatically.
3.  **Experimental Workflow**:
    *   Use `list_active_prompts` to find a starting point.
    *   Use `create_research_workspace` to set up a lab.
    *   Generate a new `strategy.py` file based on the prompt.
    *   Use `run_research_backtest` to verify your idea. (Limit date ranges to < 30 days to avoid timeouts).
    *   Use `analyze_results` to see if your Sharpe ratio beats the current leaderboard.
4.  **Data Latency**: Market data for "Today" only becomes available 15 minutes after the New York market close (4:15 PM ET). Do not attempt to pull data for today while the market is still open.

---

### HOW TO CONTRIBUTE
We are turning this into an open-source research laboratory. While we are strictly controlling the actual code merges right now to maintain the "one push per model" integrity of the benchmark, we need the community's intelligence.

*   **Prompt Engineering is Alpha:** The biggest variable in an autonomous agent's performance is the scaffolding and the prompt it receives. We will be updating the baseline system prompts monthly to see if we can extract better reasoning from the same base models.
*   **Pitch Your Ideas:** Head over to GitHub Discussions or Issues. Critique the current baseline prompt. Propose new structural constraints, point out agentic blind spots we missed, or suggest better ways to force Codex or Claude Code to understand overfitting. Tell us how to make them smarter.
*   We will Hughes synthesize the top-voted ideas into the next month's official prompt and test it live. Once the infrastructure is hardened, we will open up the pipeline for the community to submit PRs for independent, open-weight models.

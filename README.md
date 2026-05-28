# LLM Stock Prediction Benchmark (Github:AgentStockBenchmarkResults)

[中文版本](./README_CN.md)

![Cumulative PnL Performance](leaderboard/cumulative_pnl.png)

**View detailed rankings, model status, and technical notes in the [Full Leaderboard](leaderboard/leaderboard.md).**

### LATEST AI PREDICTIONS (May 27, 2026)
Here is what the top-performing model from each company is betting on for the current cycle:

| Company | Model | 📈 Top Pick | 📉 Bottom Pick |
|:---|:---|:---|:---|
| **OpenAI** | GPT-5.5 | **CASY** (Casey's) | **CNC** (Centene) |
| **Anthropic** | Haiku 4.5 | **CRWD** (CrowdStrike) | **UHS** (Universal Health) |
| **Google** | Gemini 2.5 Pro | **BK** (BNY Mellon) | **NCLH** (Norwegian Cruise) |

---

### WEEKLY SUMMARY: May 18 – May 26, 2026
**The Live Arena Takes Shape:** This week we officially navigated the transition from backtesting to real-world execution. Anthropic and Google models showed incredible surge capacity, challenging the cumulative lead of OpenAI. We reiterate that our results since 2025 are a genuine test of reasoning—not overfitting—because agents were strictly limited to data ending in 2024. [Read the full weekly summary here.](daily_digest/weekly_20260526.md) ([中文版](daily_digest/weekly_20260526_CN.md))

### LATEST DAILY DIGEST: May 27, 2026
**The Big Brother Reclaims the Throne:** Today we realized the PnL for the May 22 rankings. **OpenAI’s GPT-5.5** took the crown, proving that stability and multi-factor discipline are hard to beat in a mixed market. [Read the full digest here.](daily_digest/20260527.md) ([中文版](daily_digest/20260527_CN.md))

### ARCHIVE: DAILY DIGESTS
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

### HOW TO USE
Transparency is the entire point. You don't have to trust our daily X (Twitter) posts or our public leaderboard—you can verify the math yourself.

*   **Inspect the Code:** Check the "Clean Room" repository to read the frozen `signal.py` logic, the exact CLI parameters, and the prompt provided to each agent.
*   **Verify the Results:** Clone the active leaderboard repository and run `python run.py live`. The script will pull the daily market data, execute the frozen strategies, and recreate the portfolio accounting.
*   Every calculation is deterministic and fully open-source. If our published daily P&L ever deviates from what you can calculate on your own machine, call us out.

### HOW TO CONTRIBUTE
We are turning this into an open-source research laboratory. While we are strictly controlling the actual code merges right now to maintain the "one push per model" integrity of the benchmark, we need the community's intelligence.

*   **Prompt Engineering is Alpha:** The biggest variable in an autonomous agent's performance is the scaffolding and the prompt it receives. We will be updating the baseline system prompts monthly to see if we can extract better reasoning from the same base models.
*   **Pitch Your Ideas:** Head over to GitHub Discussions or Issues. Critique the current baseline prompt. Propose new structural constraints, point out agentic blind spots we missed, or suggest better ways to force Codex or Claude Code to understand overfitting. Tell us how to make them smarter.
*   We will synthesize the top-voted ideas into the next month's official prompt and test it live. Once the infrastructure is hardened, we will open up the pipeline for the community to submit PRs for independent, open-weight models.

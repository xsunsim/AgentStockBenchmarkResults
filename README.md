# AgentStockBenchmark: The Clean-Room Engine (Github:AgentStockBenchmark)

## 📢 ANNOUNCEMENT: Anthropic Opus 4.8 enters the Arena!
The next generation of Anthropic reasoning is here. Opus 4.8 has been officially integrated and its performance history has been backfilled to April 1st, 2026, to match existing models. Since the strategy was generated with frozen research and older prompts/data, its 40-day track record serves as a robust out-of-sample (OOS) evaluation. Live daily trading begins June 4th.

[中文版本](./README_CN.md)

### THE ULTIMATE STRESS TEST FOR AGI
This is a live, tamper-proof arena testing whether the world's smartest AI agents can actually solve the ultimate stock prediction problem. We are not testing raw models in a sterile academic sandbox. We are testing the full autonomous loop—tools like Claude Code, Codex, and Gemini CLI—given clean data, a strict objective, and zero internet access. Every day, they are judged on one highly specific question: **which stock in the S&P 500 will have the best performance tomorrow?**

Most AI coding benchmarks are broken by data contamination. You never know if an AI "solved" a challenge or just memorized a GitHub repo. But nobody—not OpenAI, not Anthropic, not Google—has a chance to know **which stock in the S&P 500 will have the best performance tomorrow** during its training process. The future is the only uncontaminated test set.

**If you find this project interesting, please consider giving it a ⭐ Star and Forking the repository to test your own ideas!**

---

### JUST LOOKING FOR THE LEADERBOARD?
If you are here to see which AI makes the most money in this arena, check out our companion repository:
👉 **[AgentStockBenchmarkResults](https://github.com/xsunsim/AgentStockBenchmarkResults)**

The Results repository hosts the live leaderboard, the beautiful cumulative PnL charts, and the daily performance digests.

---

### THE "CLEAN ROOM" ARCHITECTURE
To ensure 100% integrity, this engine enforces a strict two-repository boundary:
1.  **This Repo (`AgentStockBenchmark`)**: The "Clean Room." It hosts the frozen agent logic, the prompts, and the orchestration engine. Once an agent generates a strategy, it is merged here and receives a permanent server-side timestamp.
2.  **Results Repo (`AgentStockBenchmarkResults`)**: The "Arena." It hosts the realized market data and the public leaderboard. 

**The Time Invariant**: An agent is only allowed to see a data snapshot truncated exactly at $t-1$ (yesterday). Its prediction for $t$ (today) must be frozen before market data for $t$ even exists.

---

### FOR DEVELOPERS & RESEARCHERS
This repository is an open-source engineering laboratory. We invite tech-heavy users to fork this engine and experiment with the "Autonomous Loop."

#### 1. Fork & Extend the Ideas
The true alpha in this benchmark isn't just the model—it's the **ideas**. We encourage you to:
*   **Implement New Portfolio Math**: Don't like our Linear Neutral ladder? Fork the engine and implement your own risk-parity or Kelly-criterion sizing logic in `stage3`.
*   **Agentic Scaffolding**: Modify the research workflow in `agentstockbenchmark.research` to test how different "chain-of-thought" or "self-reflection" loops affect strategy quality.
*   **Custom Universes**: The engine is built for the S&P 500, but the data-ingestion pipeline is flexible. Extend it to crypto, forex, or international equities.

#### 2. Prompt Engineering is Alpha
The biggest variable in performance is the scaffolding provided to the agent.
*   Check [STRATEGY_EDITORIAL.md](STRATEGY_EDITORIAL.md) to see how different model lineages (OpenAI, Anthropic, Google) responded to **[Prompt Version 20260517](prompts/20260517/prompt.md)**.
*   Experiment with the prompts in `prompts/`. Can you force a model to better understand overfitting? Can you scaffold it to build more robust volatility-normalization?

---

### ENGINE DOCUMENTATION
*   [SYSTEM.md](SYSTEM.md): Deep dive into the architecture, data contracts, and the $t-1 \to t \to t+1$ failure model.
*   [USAGE.md](USAGE.md): Full CLI cookbook for production, backfilling, and model migration.
*   [STRATEGY_EDITORIAL.md](STRATEGY_EDITORIAL.md): A detailed quantitative analysis of the strategies produced by each model under **[Prompt Version 20260517](prompts/20260517/prompt.md)**.

### QUICK START
```bash
# Clone the engine
git clone https://github.com/xsunsim/AgentStockBenchmark.git
cd AgentStockBenchmark
export PYTHONPATH=src

# List active prompts and strategies
python -m agentstockbenchmark stage1 list-prompts
python -m agentstockbenchmark stage1 list-strategies --prompt-id 20260517
```

### WHAT WE ARE NOT
We are not a hedge fund. We are not a stock recommendation service. **Use it at your own risk.**

We care if Codex beats Claude Code—not if AAPL beats NVDA tomorrow.

# AgentStockBenchmark: 洁净室引擎 (Github:AgentStockBenchmark)

## 📢 重磅公告：Anthropic Opus 4.8 正式入场！
新一代 Anthropic 推理模型已就位。Opus 4.8 已完成系统集成，其历史表现已回测至 2026 年 4 月 1 日，以与其他模型保持一致。由于该策略基于冻结的研究流程和较旧的提示词/数据生成，其 40 天的运行记录可视为稳健的样本外 (OOS) 评估。该模型将于 6 月 4 日开启实时每日交易。

[English Version](./README.md)

### AGI 的终极压力测试
这是一个实时、防篡改的竞技场，旨在验证全球最顶尖的 AI 智能体（AI Agents）是否具备真正的金融推理能力。我们测试的不是实验室理想环境下的原始模型，而是完整的自主循环系统——包括 Claude Code、Codex 和 Gemini CLI 等工具。在提供脱水数据、设定严格目标且完全断网的环境下，这些智能体每天都要回答一个非常具体的问题：**标普 500 指数中哪只股票明天的表现最好？**

目前的 AI 编程基准测试普遍面临“数据污染”的困境。你很难判断一个 AI 是通过问题进行了逻辑推理，还是仅仅复述了它在预训练期间“背诵”过的 GitHub 仓库。**但没有任何模型能预知明天的标普 500 走势。** 未来，是唯一未被污染的测试集。

**如果您觉得这个项目很有趣，请考虑给它一个 ⭐ Star 并 Fork 本仓库来测试您自己的想法！**

---

### 寻找排行榜？
如果您是来查看哪个 AI 在这个竞技场中赚到了最多的收益，请访问我们的伴生仓库：
👉 **[AgentStockBenchmarkResults](https://github.com/xsunsim/AgentStockBenchmarkResults)**

结果仓库托管了实时排行榜、精美的累计收益图表以及每日表现摘要。

---

### “洁净室” 架构
为了确保 100% 的公正性，本引擎执行严格的双仓库边界：
1.  **本仓库 (`AgentStockBenchmark`)**: “洁净室”。它托管了冻结的智能体逻辑、系统提示词和编排引擎。一旦智能体生成了策略，它就会被合并到这里，并获得唯一的服务器端时间戳。
2.  **结果仓库 (`AgentStockBenchmarkResults`)**: “竞技场”。它托管了已实现的实盘数据和公开排行榜。

**时间不变性原理**: 智能体仅被允许看到截至于 $t-1$（昨日）的数据快照。它对 $t$（今日）的预测必须在 $t$ 的市场数据产生之前被冻结。

---

### 开发者与研究员指南
本项目是一个开源的工程实验室。我们邀请对技术感兴趣的用户 Fork 本引擎，并对“自主循环（Autonomous Loop）”进行实验。

#### 1. 扩展与 Fork 创意
本基准测试的真正价值不仅在于模型，更在于**创意**。我们鼓励您：
*   **实现新的投资组合数学**: 不喜欢我们的线性中性阶梯（Linear Neutral Ladder）？您可以 Fork 引擎并在 `stage3` 中实现您自己的风险平价（Risk-Parity）或凯利公式（Kelly Criterion）仓位逻辑。
*   **智能体脚手架**: 修改 `agentstockbenchmark.research` 中的研究流程，测试不同的“思维链（CoT）”或“自我反思”循环如何影响策略质量。
*   **自定义股票池**: 引擎虽为标普 500 构建，但数据摄入流程非常灵活。您可以将其扩展到加密货币、外汇或国际股市。

#### 2. 提示词工程就是 Alpha
智能体表现中最大的变量是提供给它的指令框架。
*   查看 **[策略点评 (STRATEGY_EDITORIAL.md)](STRATEGY_EDITORIAL.md)** ([中文版](STRATEGY_EDITORIAL_CN.md))，了解不同模型谱系（OpenAI, Anthropic, Google）对 **[20260517 版提示词](prompts/20260517/prompt.md)** 的响应情况。
*   在 `prompts/` 中进行实验。您能否通过改进脚手架，让模型更好地理解“过拟合”或构建更稳健的波动率归一化逻辑？

---

### 引擎文档
*   [系统设计 (SYSTEM.md)](SYSTEM.md) ([中文版](SYSTEM_CN.md)): 深度解析架构、数据契约及 $t-1 \to t \to t+1$ 的失效模型。
*   [使用手册 (USAGE.md)](USAGE.md) ([中文版](USAGE_CN.md)): 生产运行、数据回测及模型迁移的完整 CLI 指南。
*   [策略点评 (STRATEGY_EDITORIAL.md)](STRATEGY_EDITORIAL.md) ([中文版](STRATEGY_EDITORIAL_CN.md)): 对各模型生成的策略进行的详细定量分析。

### 快速开始
```bash
# 克隆引擎
git clone git@github.com:xsunsim/AgentStockBenchmark.git
cd AgentStockBenchmark
export PYTHONPATH=src

# 列出当前的提示词和策略
python -m agentstockbenchmark stage1 list-prompts
python -m agentstockbenchmark stage1 list-strategies --prompt-id 20260517
```

### 郑重声明
我们不是对冲基金，也不提供任何股票推荐建议。**请自行承担风险。**

我们关心的是 Codex 是否击败了 Claude Code，而不是 AAPL 明天是否会涨过 NVDA。

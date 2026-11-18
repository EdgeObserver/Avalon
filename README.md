# 阿瓦隆 · AI 对局

用 [AgentScope](https://github.com/modelscope/agentscope) 搭的《阿瓦隆》多智能体桌游：6 个 AI 玩家 + 你自己的座位，浏览器里开局。

![Python](https://img.shields.io/badge/python-3.12-blue)
![AgentScope](https://img.shields.io/badge/AgentScope-2.0.12-green)
![Gradio](https://img.shields.io/badge/UI-Gradio%206-orange)
![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen)

## 这是什么

一个可玩的阿瓦隆实现，同时也是一个用来观察「模型会不会撒谎、会不会推理」的实验台。

- **能玩**：完整的 5~10 人流程，组队、投票、出任务牌、刺杀，加了一个自定的**沉船追责**环节。
- **能看**：圆桌视图标出队长与出局者，局末揭晓身份，每个 agent 还会写一段**心路历程**。
- **能测**：每个动作都有结构化 schema 与语义校验，非法动作会被回问修复，兜底不会崩对局。
- **能复盘**：整局事件落盘成 JSONL，附带一个分析脚本，可统计梅林的投票精准度、刺客的指认排名等指标。

## 快速开始

```bash
conda env create -f environment.yml
conda activate avalon

cp .env.example .env      # 填入你的 API key
python ui.py --players 7 --seat 1
```

浏览器打开 `http://127.0.0.1:7860`，点「开始游戏」抽身份。

### 模型配置

默认走 **OpenCode Go**（OpenAI 兼容端点），模型 `deepseek-v4.1-flash`：

```env
OPENCODE_API_KEY=sk-...
# OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1
# AVALON_MODEL=deepseek-v4.1-flash
```

也支持任何 OpenAI 兼容端点。本地 Ollama 无需 key：

```python
model = build_model(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3.5:9b",
)
```

> OpenCode Go 的 `/chat/completions` 要求 `x-opencode-session` 头，缺了会 400。
> `build_model()` 已经自动加上，不用自己处理。

### 命令行跑全 AI 对局

```python
import asyncio
from agents import build_model
from batch import run_batch, summarize, print_summary, save_jsonl

async def main():
    results = await run_batch(12, players=7, seed_base=1000, model=build_model())
    print_summary(summarize(results))
    save_jsonl(results, "runs/7p_12games.jsonl")

asyncio.run(main())
```

分析已落盘的日志：

```python
from analysis import analyze, load_jsonl, print_analysis
print_analysis(analyze(load_jsonl("runs/7p_12games.jsonl")))
```

## 界面

- **顶栏**：人数、座位、种子（同种子可复现同一份身份）
- **身份卡**：你的角色与私有信息
- **左栏**：对局日志，实时滚动
- **右栏**：圆桌视图（👑 队长 / 蓝环 你的座位 / 灰 出局 / 红色角标 票数）、当前该你做什么、输入框
- **局末**：获胜方、刺客的刺杀理由、身份表、每个人的复盘
- **📖 规则**：随时打开完整规则弹窗

输入格式：`2,4` 组队、`同意/反对` 投票、`成功/失败` 出牌、座位号刺杀、`弃权` 跳过淘汰票。回车提交，Shift+回车换行。

## 项目结构

| 文件 | 职责 |
|---|---|
| `prompts.py` | 角色定义、规则简报、七个角色的策略与进阶技巧、表演要点 |
| `roles.py` | 人数 → 角色配置 → 私有信息（谁能看到谁） |
| `agents.py` | 构造模型与 Agent（OpenAI 兼容 / 多人 formatter） |
| `actions.py` | **动作层**：结构化输出 + 语义校验 + 回问 + 兜底 |
| `rules.py` | 对局状态与所有动作的合法性规则 |
| `engine.py` | 主循环：组队 → 投票 → 出牌 → 追责 → 刺杀，含事件日志 |
| `human.py` | 人类座位：把 UI 输入包装成 Agent 接口 |
| `channels.py` | 阵营私密频道（已实现，默认未接入对局） |
| `batch.py` | 批量对局、指标汇总、JSONL 落盘 |
| `analysis.py` | 从日志挖行为指标，不复跑对局 |
| `ui.py` | Gradio 界面 |

## 几个设计决定

**动作层的三层保证。** 模型的输出要过三关：pydantic schema 保证形状，`legal()` 校验保证语义（座位存在、人数正确、好人不能出失败牌），失败则带着**合法集合**回问，仍不行才降级。实测报错里不写「可选值有哪些」时，小模型会连续三次撞同一堵墙。

**把跨轮信息聚合好再喂给模型。** 投票记录散在自然语言里时，模型数不出来。所以追责前会广播一份「公开记录」：每轮组队与结果、每人反对票的命中率、历次追责票型。刺客刺杀前也拿同一份。

**信息隔离靠引擎，不靠提示词。** 私有信息由 `roles.py` 生成，投递给谁由引擎决定；`PrivateChannel` 的越权读写直接抛 `PermissionError`。

**淘汰会公开阵营。** 否则「误杀好人 ⇒ 投他的人可疑」这条推理链根本没有前提，场上玩家不知道出局者是好是坏。

**没有淘汰机制的官方规则 + 一个自定环节。** 沉船追责是变体规则，可以在 `engine.py::_accountability` 里关掉。

## 关于 AgentScope 2.x

网上大部分 AgentScope 教程还停留在 v0.x（`DialogAgent`、`MsgHub`、`fanout_pipeline`），在 2.0 里全部不存在。本项目里踩到的坑：

| 现象 | 原因 |
|---|---|
| `MsgHub` / `fanout_pipeline` 导入失败 | v2 已删除，`pipeline` 只剩 `GoalPipeline` / `TeamPipeline` |
| `OpenAIChatModel(api_key=...)` 报错 | 现在要传 `OpenAICredential` 对象 |
| `Msg(content="字符串")` 报错 | content 必须是内容块列表；用 `UserMsg(name=..., content="...")` |
| `agent.observe(...)` 没生效 | 它是 async 的，必须 `await` |
| 模型只回一句 `The maximum reasoning-acting iterations are exceeded.` | 开了 thinking，推理块不产生正文，ReAct 循环空转到上限。关掉 `thinking_enable` |
| 多 agent 分不清谁在说话 | 必须用 `OpenAIMultiAgentFormatter`，普通 formatter 会把所有发言揉成一条 user 消息 |

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

30 个用例，全部纯逻辑、不调用模型，约 1.3 秒跑完：

- 角色分配的不变量（梅林看不到莫德雷德、奥伯伦对所有人隐身、派西维尔看到两人）
- 动作合法性（人数不符、重复队员、好人出失败牌、指认自己、对出局者行动）
- 人类输入解析（含 `-1` 不能被静默解析成 `1` 这种坑）
- 用假 agent 跑完整对局，覆盖三条胜负路径

## Roadmap

- [ ] 红方密会阶段（`channels.py` 已实现，未接入主循环）
- [ ] 明暗主题切换
- [ ] 把批量结果做成对比报告（提示词 A/B）
- [ ] 支持扩展角色（湖中仙女、暗影等）

## 许可

MIT，见 [LICENSE](LICENSE)。

本项目只做技术演示与研究用途，和任何商业发行的《阿瓦隆》产品无关。

"""结构化动作层：把「模型说了什么」变成「引擎可以信任的输入」。

为什么要这一层：`reply(..., structured_schema=...)` 只能保证 **格式** 合法
（类型、枚举、必填、pydantic validator）。它不知道游戏状态，也不知道
"玩家7" 在 5 人局里根本不存在。实测失败时它还是**静默**的——
`structured_output` 直接是 None，不抛异常。

所以这里补上另外两段：

    调用 → 判空 → 语义合法性校验 → 带错误回问 → 降级 / 抛错

用法::

    def legality(v: dict) -> list[str]:
        if v["target"] not in alive_seats:
            return [f"target 必须是存活玩家：{sorted(alive_seats)}"]
        if v["target"] == self_seat:
            return ["不能选自己"]
        return []

    res = await act(agent, "请选择本轮的击杀目标。", KillVote, legal=legality)
    if not res.ok:
        log.warning("降级了：%s", res.errors)
    target = res.value["target"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel

from agentscope.agent import Agent
from agentscope.message import Msg, UserMsg

logger = logging.getLogger(__name__)

#: 语义校验器：拿到模型输出的 dict，返回违规说明列表；空列表 = 合法。
LegalityChecker = Callable[[dict[str, Any]], list[str]]


class ActError(RuntimeError):
    """所有尝试都失败，且没有提供 fallback。"""


@dataclass
class ActResult:
    """一次动作的结果。引擎应该先看 ok，再决定用不用 value。"""

    ok: bool
    value: dict[str, Any] | None
    source: Literal["model", "fallback"]
    attempts: int
    #: 每一轮累积的违规说明（含框架层面的结构化失败）
    errors: list[str] = field(default_factory=list)
    #: 每次尝试模型返回的原始 structured_output，失败时便于审计
    raw: list[dict[str, Any] | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.ok and self.value is None:
            raise ValueError("ActResult 标了 ok 但 value 是 None，逻辑有问题")


def _fmt_reason(reply: Msg) -> str:
    reason = getattr(reply, "finished_reason", None)
    return str(getattr(reason, "value", reason))


async def act(
    agent: Agent,
    instruction: str | Msg,
    schema: type[BaseModel],
    *,
    legal: LegalityChecker | None = None,
    max_attempts: int = 3,
    fallback: Callable[[], dict[str, Any]] | None = None,
    speaker: str = "主持人",
) -> ActResult:
    """让 agent 产出一个合法动作。

    Args:
        agent: 要行动的 Agent。
        instruction: 第一次的指令；字符串会被包成 UserMsg。
        schema: 动作的 pydantic 模型，决定输出形状。
        legal: 语义校验器。**强烈建议传**——不传的话这一层就只剩判空，
            等于放弃了它对游戏规则的把关。
            **报错文案里必须写清合法集合**。实测：只说「不能刺杀你自己」
            时，4B 模型能连撞三次同一堵墙；改成「不能刺杀你自己。合法目标
            只有：1、2、3、4、5、7」后一次就改对了。这条错误信息就是模型
            唯一能看到的纠错线索，别省。
        max_attempts: 外层重试次数。框架内部已经会为"格式不对"重试
            （max_iters + grace），这里的重试是给"格式对但语义错"用的，
            2~3 次足够。
        fallback: 全部失败时的确定性兜底。**它必须是合法的**，本函数会
            用同一个 legal 校验一遍；兜底本身就是脏的，那不如直接抛。
        speaker: 回问消息的发送者名字，通常就是主持人。

    Returns:
        ActResult。`ok=False` 表示走的是 fallback（此时 value 依然可用，
        但引擎应当记一笔降级）。

    Raises:
        ActError: 所有尝试失败且没有 fallback；或者 fallback 本身不合法。
    """
    if max_attempts < 1:
        raise ValueError("max_attempts 至少为 1")

    errors: list[str] = []
    raws: list[dict[str, Any] | None] = []
    # 注意：Agent.reply() 只吃 Msg，传裸字符串会被当成字符列表逐字解析，
    # 报 "Invalid message in the input: 请"。这里必须自己包成 UserMsg。
    msg: Msg = (
        instruction
        if isinstance(instruction, Msg)
        else UserMsg(name=speaker, content=instruction)
    )

    for attempt in range(1, max_attempts + 1):
        # ---- 1. 让模型产出（框架内部已对格式问题重试过） ----
        try:
            reply = await agent.reply(msg, structured_schema=schema)
        except Exception as e:  # noqa: BLE001 - 网络/限流不该炸掉整局游戏
            problem = f"调用失败：{type(e).__name__}: {e}"
            logger.warning("act 第 %d 次调用异常: %s", attempt, problem)
            errors.append(problem)
            msg = UserMsg(name=speaker, content=problem + "\n请重新作答。")
            continue

        raw = reply.structured_output
        raws.append(raw)

        # ---- 2. 判空：框架会静默返回 None ----
        if raw is None:
            problem = (
                f"没有拿到符合格式的答案（finished_reason={_fmt_reason(reply)}）。"
            )
            errors.append(problem)
            logger.warning("act 第 %d 次结构化输出为空", attempt)
            msg = UserMsg(
                name=speaker,
                content=problem + f"\n请严格按照 {schema.__name__} 的字段重新给出。",
            )
            continue

        # ---- 3. 语义校验 ----
        problems = list(legal(raw)) if legal else []
        if not problems:
            return ActResult(
                ok=True,
                value=raw,
                source="model",
                attempts=attempt,
                errors=errors,
                raw=raws,
            )

        errors.extend(problems)
        logger.info("act 第 %d 次语义不合法: %s", attempt, problems)
        msg = UserMsg(
            name=speaker,
            content="你刚才的选择不合法：" + "；".join(problems) + "。请重新给出。",
        )

    # ---- 4. 降级 ----
    if fallback is None:
        raise ActError(
            f"{agent.name} 连续 {max_attempts} 次没有给出合法动作：{' | '.join(errors)}"
        )

    value = fallback()
    problems = list(legal(value)) if legal else []
    if problems:
        raise ActError(
            "fallback 本身就不合法：" + "；".join(problems) + "（兜底逻辑有 bug）"
        )
    logger.warning(
        "%s 降级到 fallback，%d 次尝试均失败：%s",
        agent.name,
        max_attempts,
        " | ".join(errors),
    )
    return ActResult(
        ok=False,
        value=value,
        source="fallback",
        attempts=max_attempts,
        errors=errors,
        raw=raws,
    )

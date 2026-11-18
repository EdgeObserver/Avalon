"""人类玩家：把 UI 输入接成 Agent 的接口。

设计上只做一件事——**让人类座位伪装成一个 Agent**。引擎、`act()`、`legal` 校验
全都不需要知道这是人还是模型：

- `reply(msg, structured_schema)` 挂起等 UI 交答案，返回一个带 `structured_output` 的 Msg
- `observe(msgs)` 把公告记下来给 UI 显示

好处是合法性校验完全复用：人类输入不合法时，`act()` 会像对待模型一样带着错误
信息回来再问一次，UI 那边看到的就是「请重新输入」。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from agentscope.message import Msg, TextBlock
from pydantic import BaseModel

from rules import (
    ApprovalVote,
    Assassination,
    EliminationVote,
    QuestCard,
    Speech,
    TeamProposal,
)


def _text_of(msg) -> str:
    return "".join(
        b.text for b in msg.content if getattr(b, "type", "") == "text"
    ).strip()


@dataclass
class HumanRequest:
    """当前等待人类处理的一次决策。"""

    schema: type[BaseModel]
    prompt: str
    hint: str

    @property
    def kind(self) -> str:
        return {
            TeamProposal: "组队",
            ApprovalVote: "投票",
            QuestCard: "出任务牌",
            Assassination: "刺杀",
            Speech: "追责发言",
            EliminationVote: "淘汰投票",
        }.get(self.schema, self.schema.__name__)


HINTS: dict[type[BaseModel], str] = {
    TeamProposal: "输入座位号，用逗号分隔，例如 `2,4`。想加理由用竖线隔开：`2,4 | 想看看他们配合`",
    ApprovalVote: "输入 `同意` 或 `反对`",
    QuestCard: "输入 `成功` 或 `失败`（好人只能出成功牌）",
    Assassination: "输入你要指认的座位号，例如 `4`",
    Speech: "直接说你想说的话，2~4 句就够",
    EliminationVote: "输入要淘汰的座位号，例如 `3`；不想淘汰任何人就输入 `弃权`",
}


_SEAT_RE = re.compile(r"-?\d+")


def _seats(text: str, what: str = "座位号") -> list[int]:
    """抽出座位号，并挡住 0 / 负数。

    注意不能只匹配 \\d+：那样 "-1" 会被静默解析成 1，把用户的笔误变成
    另一个座位的指令——比报错危险得多。
    """
    nums = [int(n) for n in _SEAT_RE.findall(text)]
    bad = [n for n in nums if n < 1]
    if bad:
        raise ValueError(f"{what}必须是 1 及以上的正整数，不能是 {bad[0]}")
    return nums


def parse_action(schema: type[BaseModel], text: str) -> dict:
    """把一行文本解析成 schema 要的 dict。宽进严出——格式错直接报错，语义错交给引擎。"""
    raw = text.strip()
    if not raw:
        raise ValueError("输入不能为空")

    if schema is TeamProposal:
        head, _, reason = raw.partition("|")
        seats = _seats(head, "组队")
        if not seats:
            raise ValueError("没找到座位号，例如输入 `2,4`")
        return {"team": seats, "reason": reason.strip()}

    if schema is ApprovalVote:
        if raw in ("同意", "是", "yes", "y", "1"):
            return {"approve": "同意"}
        if raw in ("反对", "否", "no", "n", "0"):
            return {"approve": "反对"}
        raise ValueError("只能输入 `同意` 或 `反对`")

    if schema is QuestCard:
        if raw.startswith("成功"):
            return {"card": "成功"}
        if raw.startswith("失败"):
            return {"card": "失败"}
        raise ValueError("只能输入 `成功` 或 `失败`")

    if schema is Assassination:
        nums = _seats(raw, "刺杀目标")
        if len(nums) != 1:
            raise ValueError("请只输入一个座位号")
        return {"target": nums[0], "reasoning": ""}

    if schema is Speech:
        return {"text": raw}

    if schema is EliminationVote:
        if raw in ("弃权", "跳过", "pass", "none", "无"):
            return {"target": None, "reason": ""}
        nums = _seats(raw, "淘汰对象")
        if len(nums) != 1:
            raise ValueError("请输入一个座位号，或输入 `弃权`")
        return {"target": nums[0], "reason": ""}

    raise ValueError(f"不认识的动作类型：{schema.__name__}")


class HumanPlayer:
    """一个座位的真人玩家，接口与 `agentscope.agent.Agent` 对齐。

    `validator` 是可选的前置校验器 ``(schema, payload) -> list[str]``，
    通常直接复用 `rules.check_*`。有它的话，非法输入会在**人类这一层**就被
    打回重填，不会消耗 `act()` 的重试预算——否则连输三次非法输入会把对局
    直接抛 `ActError` 打断。
    """

    def __init__(self, seat: int, validator=None) -> None:
        self.seat = seat
        self.name = f"玩家{seat}"
        self.validator = validator
        self.pending: HumanRequest | None = None
        self.seen: list[str] = []
        self._future: asyncio.Future | None = None

    # ---- Agent 接口 -------------------------------------------------- #

    async def observe(self, msgs) -> None:
        items = msgs if isinstance(msgs, list) else [msgs]
        for m in items:
            t = _text_of(m)
            if t:
                self.seen.append(t)

    async def reply(self, msg, structured_schema=None):
        base = _text_of(msg)
        hint = HINTS.get(structured_schema, "按提示输入")
        problem = ""

        while True:
            prompt = base + (f"\n\n⚠️ {problem}" if problem else "")
            loop = asyncio.get_running_loop()
            self._future = loop.create_future()
            self.pending = HumanRequest(
                schema=structured_schema, prompt=prompt, hint=hint
            )
            try:
                payload = await self._future
            finally:
                self.pending = None
                self._future = None

            problems = (
                list(self.validator(structured_schema, payload))
                if self.validator else []
            )
            if not problems:
                break
            problem = "；".join(problems)

        return Msg(
            name=self.name,
            role="assistant",
            content=[TextBlock(type="text", text=base)],
            structured_output=payload,
        )

    # ---- 给 UI 用 ---------------------------------------------------- #

    def submit(self, text: str) -> str:
        """UI 提交一次输入。返回错误文案，空串表示已受理。"""
        if self.pending is None or self._future is None:
            return "现在不是你的回合"
        if self._future.done():
            return "上一次输入还在处理中"
        try:
            payload = parse_action(self.pending.schema, text)
        except ValueError as e:
            return str(e)
        self._future.set_result(payload)
        return ""

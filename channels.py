"""阵营私密频道。

> **状态：已实现，但当前未接入对局主循环。** `engine.py` 不会自动启用它；
> 想用的话在 `_accountability` 之类的位置调 `post()` / `deliver()` 即可。
> 官方阿瓦隆并没有局内私聊，所以默认关闭是故意的。

先分清三件容易混在一起的事：

1. **身份互认**：开局一次性告诉红方「你的同伙是谁」。这是一次性信息注入，
   走 prompts 的 private_info，不需要任何通信设施。
2. **私密频道**：局内同阵营之间的对话，只有成员看得见。本文件负责这个。
3. **公开发言**：全员可见，走正常的 reply()。

设计上的三个决定：

- 频道用「伪参与者」的身份注入，name 就是频道名（如「红方频道」）。
  这样 formatter 渲染出来是 `红方频道: 玩家2：建议塞玩家5`，
  模型一眼看出这是频道而不是某人当众说的，不会把私房话当成公开发言。
- 增量投递：每个座位记一个游标，只投递它还没看过的消息，
  避免每轮把整段历史重复塞进上下文。
- 自己的发言不回灌给自己：它已经作为 assistant 回合留在自己的记忆里了。

本模块是纯同步逻辑，不碰 asyncio；投递由引擎 `await agent.observe(msg)` 完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from agentscope.message import UserMsg

#: 频道在对话里的代号。会被当作消息的 name 渲染。
CHANNEL_SPEAKER = "红方频道"


@dataclass
class PrivateChannel:
    """一个阵营的私密频道。

    Example:
        >>> ch = PrivateChannel(members=[2, 4, 6])
        >>> ch.post(2, "第一轮把玩家5塞进队伍，我来投破坏。")
        >>> msg = ch.deliver(4)          # 玩家4 拿到一条待 observe 的消息
        >>> ch.deliver(1)                # 好人看不到 → PermissionError
    """

    members: frozenset[int]
    label: str = CHANNEL_SPEAKER
    _log: list[tuple[int, str]] = field(default_factory=list, init=False)
    _cursor: dict[int, int] = field(default_factory=dict, init=False)

    def __init__(self, members: Iterable[int], label: str = CHANNEL_SPEAKER):
        self.members = frozenset(members)
        self.label = label
        self._log = []
        self._cursor = {seat: 0 for seat in self.members}

    # ------------------------------------------------------------------ #
    # 成员判断
    # ------------------------------------------------------------------ #

    def can_see(self, seat: int) -> bool:
        return seat in self.members

    def _require_member(self, seat: int) -> None:
        if not self.can_see(seat):
            raise PermissionError(
                f"玩家{seat} 不是「{self.label}」的成员，不能读写这个频道"
            )

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def post(self, sender: int, text: str) -> None:
        """频道成员发言。text 会原样保留，不做任何裁剪。"""
        self._require_member(sender)
        text = text.strip()
        if not text:
            return
        self._log.append((sender, text))

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #

    def pending(self, seat: int) -> list[tuple[int, str]]:
        """该座位还没看过的消息（不含自己说的）。"""
        self._require_member(seat)
        cursor = self._cursor.get(seat, 0)
        items = [
            (sender, text)
            for sender, text in self._log[cursor:]
            if sender != seat
        ]
        self._cursor[seat] = len(self._log)
        return items

    def deliver(self, seat: int) -> UserMsg | None:
        """把未读消息打包成一条 UserMsg，交给引擎去 observe。

        没有新消息时返回 None。
        """
        items = self.pending(seat)
        if not items:
            return None
        body = "\n".join(f"玩家{sender}：{text}" for sender, text in items)
        return UserMsg(name=self.label, content=body)

    def transcript(self, seat: int) -> str:
        """给日志/回放用的完整转录（不移动游标）。"""
        self._require_member(seat)
        return "\n".join(
            f"玩家{sender}：{text}" for sender, text in self._log
        )

    def __len__(self) -> int:
        return len(self._log)

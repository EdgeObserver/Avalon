"""对局状态 + 动作合法性规则。

这个模块只做两件事：**记住场上谁还在场**，以及**把规则翻译成校验函数**，
供 actions.act(legal=...) 使用。

为什么要把「谁还在场」单独拎出来：标准阿瓦隆没有淘汰，但你这套是人类 + AI
混桌，人类可能中途退出、某个 AI 可能崩掉。这类玩家必须同时从**两个方向**上
被排除掉，少了一个就是 bug：

1. **不能投**：不在场的人不参与投票，也不该被询问。
2. **不能被投**：组队候选、刺杀目标里不能出现不在场的人。

两端共用同一个 `active` 集合，就不会出现「提示词里还能选他、校验器却说不行」
这种自相矛盾——那种情况会让模型在重试里死循环。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from prompts import QUEST_SIZES

#: 同一轮任务连续被否决多少次，坏人直接获胜
MAX_REJECTIONS = 5


# --------------------------------------------------------------------------- #
# 动作的 schema：形状由 pydantic 保证，语义由下面的 check_* 保证
# --------------------------------------------------------------------------- #


class TeamProposal(BaseModel):
    """队长组队。"""

    team: list[int] = Field(description="本轮任务队伍，人数必须等于本轮规定人数")
    reason: str = Field(default="", description="一句话理由")


class ApprovalVote(BaseModel):
    """全员投票：同意 / 反对。"""

    approve: Literal["同意", "反对"] = Field(description="只能二选一，不要写别的")
    reason: str = Field(default="", description="一句话理由")


class QuestCard(BaseModel):
    """任务牌。好人只能出成功。"""

    card: Literal["成功", "失败"] = Field(description="只能二选一")
    bluff: str = Field(default="", description="你打算怎么解释这张牌（可留空）")


class Assassination(BaseModel):
    """刺客指认梅林。"""

    target: int = Field(description="你认为是梅林的玩家座位号")
    reasoning: str = Field(default="", description="一句话说明为什么是他")


class Speech(BaseModel):
    """追责环节的一次公开发言。"""

    text: str = Field(description="你要说的话，2~4 句，150 字以内")


class EliminationVote(BaseModel):
    """追责环节的淘汰票。"""

    target: int | None = Field(
        default=None,
        description="要淘汰的座位号；填 null 表示弃权",
    )
    reason: str = Field(default="", description="一句话理由")


class Reflection(BaseModel):
    """局后复盘。"""

    text: str = Field(description="150 字以内的复盘，直接说，不要分点")


# --------------------------------------------------------------------------- #
# 对局状态
# --------------------------------------------------------------------------- #


@dataclass
class GameState:
    """最小可用的对局状态。

    只放「所有动作都依赖」的事实。任务历史、投票记录等留给引擎自己维护，
    不要往这里堆——这个类越干净，校验函数越好写。
    """

    seats: tuple[int, ...]
    #: 还在场的座位。人类退出 / AI 崩溃时从里面移除。
    active: set[int] = field(default_factory=set)
    #: 第几轮任务，0 起
    round_index: int = 0
    #: 当前轮连续被否决的次数
    rejections: int = 0
    #: 当前队长
    leader: int = 1

    def __post_init__(self) -> None:
        if len(set(self.seats)) != len(self.seats):
            raise ValueError("seats 里有重复座位号")
        bad = [s for s in self.seats if not isinstance(s, int) or s < 1]
        if bad:
            raise ValueError(f"座位号必须是 1 及以上的正整数，收到：{bad}")
        if not self.active:
            self.active = set(self.seats)
        unknown = self.active - set(self.seats)
        if unknown:
            raise ValueError(f"active 里出现了不存在的座位：{sorted(unknown)}")
        if self.leader not in self.active:
            raise ValueError(f"队长 {self.leader} 不在场")

    # ---- 查询 -------------------------------------------------------- #

    @property
    def total_players(self) -> int:
        return len(self.seats)

    @property
    def team_size(self) -> int:
        """本轮队伍人数。玩家被淘汰后可能凑不够，按在场人数封顶。"""
        base = QUEST_SIZES[self.total_players][self.round_index]
        return min(base, len(self.active))

    @property
    def fails_needed(self) -> int:
        """本轮需要几张失败牌才算任务失败（第 4 轮、7 人以上是 2 张）。"""
        if self.round_index == 3 and self.total_players >= 7:
            return 2
        return 1

    def can_vote(self, seat: int) -> bool:
        """这个座位此刻该不该被问投票。"""
        return seat in self.active

    def voters(self) -> list[int]:
        return sorted(self.active)

    def legal_targets(self) -> list[int]:
        """可以被指认 / 被选进队伍的座位（刺杀目标、组队候选共用）。"""
        return sorted(self.active)

    # ---- 推进 -------------------------------------------------------- #

    def retire(self, seat: int) -> None:
        """玩家退出 / 掉线。"""
        if seat not in self.active:
            return
        if len(self.active) == 1:
            raise ValueError("最后一名玩家不能退出")
        self.active.discard(seat)
        if self.leader == seat:
            self.rotate_leader()

    def rotate_leader(self) -> int:
        """队长交给下一个还在场的人。"""
        order = sorted(self.active)
        idx = order.index(self.leader) if self.leader in order else -1
        self.leader = order[(idx + 1) % len(order)]
        return self.leader

    def record_rejection(self) -> bool:
        """记一次否决。返回 True 表示触发 5 次否决、坏人直接获胜。"""
        self.rejections += 1
        return self.rejections >= MAX_REJECTIONS

    def next_round(self) -> None:
        self.round_index += 1
        self.rejections = 0
        self.rotate_leader()


# --------------------------------------------------------------------------- #
# 校验函数：全部返回「违规说明列表」，空列表 = 合法
# --------------------------------------------------------------------------- #


def check_team_proposal(v: dict, state: GameState) -> list[str]:
    """组队合法性。"""
    team = v.get("team") or []
    problems: list[str] = []

    if len(set(team)) != len(team):
        problems.append(f"队员不能重复，你给了 {team}")

    if len(team) != state.team_size:
        problems.append(f"本轮必须选 {state.team_size} 人，你选了 {len(team)} 人")

    absent = [s for s in team if s not in state.active]
    if absent:
        problems.append(
            f"玩家 {absent} 已不在场，不能进队伍。可选：{state.legal_targets()}"
        )

    return problems


def check_approval(v: dict, state: GameState, voter: int) -> list[str]:
    """投票合法性。不在场的人不该出现在这里——这一层是兜底。"""
    if not state.can_vote(voter):
        return [f"玩家 {voter} 已不在场，不能投票"]
    if v.get("approve") not in ("同意", "反对"):
        return ["只能投「同意」或「反对」"]
    return []


def check_quest_card(v: dict, state: GameState, actor: int, is_evil: bool) -> list[str]:
    """任务牌合法性。好人投「失败」不是策略问题，是规则违规。"""
    if actor not in state.active:
        return [f"玩家 {actor} 已不在场，不能出任务牌"]
    card = v.get("card")
    if card not in ("成功", "失败"):
        return ["只能出「成功」或「失败」"]
    if card == "失败" and not is_evil:
        return ["你是好人阵营，好人只能出「成功」牌"]
    return []


def check_assassination(v: dict, state: GameState, actor: int) -> list[str]:
    """刺杀合法性。"""
    target = v.get("target")
    if target not in state.active:
        return [
            f"玩家 {target} 不在场，不能指认。可选：{state.legal_targets()}"
        ]
    if target == actor:
        # 刺客指认自己毫无意义，直接挡掉，避免模型钻空子
        return ["不能指认你自己，请指向你认为是梅林的玩家"]
    return []


def check_speech(v: dict, state: GameState, actor: int) -> list[str]:
    """发言合法性：只要不空、不离谱地长就行。"""
    if actor not in state.active:
        return [f"玩家 {actor} 已不在场，不能发言"]
    text = (v.get("text") or "").strip()
    if not text:
        return ["发言不能为空，说一两句你的判断"]
    if len(text) > 300:
        return [f"发言太长了（{len(text)} 字），请压到 150 字以内"]
    return []


def check_elimination_vote(v: dict, state: GameState, voter: int) -> list[str]:
    """淘汰票合法性。允许弃权（target 为 null）。"""
    if not state.can_vote(voter):
        return [f"玩家 {voter} 已不在场，不能投票"]
    target = v.get("target")
    if target is None:
        return []
    if target not in state.active:
        return [f"玩家 {target} 已在场外。可选：{state.legal_targets()}"]
    if target == voter:
        return ["不能投自己，可以投别人或弃权"]
    return []

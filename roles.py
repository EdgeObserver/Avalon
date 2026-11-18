"""角色分配 + 私有信息生成。纯规则，不调用模型，可以单独跑单测。

输入人数，输出「几号位是什么角色」，以及每个座位该看到的私有信息。

私有信息的口径按官方规则来（你贴的那份也一样）：
- **坏人互相确认的是"同伴是谁"，不是"同伴是什么角色"**。所以这里默认只报座位号，
  不报「玩家3是莫甘娜」——想要那种便利版本可以开 reveal_teammate_roles=True。
- 奥伯伦对所有人隐身：他看不到同伙，同伙也看不到他。
- 莫德雷德对梅林隐身。
- 派西维尔看到梅林和莫甘娜两个人，但分不清谁是谁。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from prompts import (
    ALL_ROLES,
    ASSASSIN,
    EVIL_ROLES,
    LOYAL,
    MERLIN,
    MORDRED,
    MORGANA,
    OBERON,
    PERCIVAL,
    QUEST_SIZES,
    is_evil,
)

#: 各人数下的标准角色配置（官方建议）
ROLE_TABLE: dict[int, tuple[str, ...]] = {
    5: (MERLIN, PERCIVAL, LOYAL, MORGANA, ASSASSIN),
    6: (MERLIN, PERCIVAL, LOYAL, LOYAL, MORGANA, ASSASSIN),
    7: (MERLIN, PERCIVAL, LOYAL, LOYAL, MORGANA, ASSASSIN, MORDRED),
    8: (MERLIN, PERCIVAL, LOYAL, LOYAL, LOYAL, MORGANA, ASSASSIN, MORDRED),
    9: (MERLIN, PERCIVAL, LOYAL, LOYAL, LOYAL, LOYAL, MORGANA, ASSASSIN, MORDRED),
    10: (MERLIN, PERCIVAL, LOYAL, LOYAL, LOYAL, LOYAL, MORGANA, ASSASSIN, MORDRED, OBERON),
}

#: 必须有且仅有一个的角色
_UNIQUE_ROLES = (MERLIN, PERCIVAL, MORGANA, ASSASSIN, MORDRED, OBERON)


def fmt_seats(seats: Sequence[int]) -> str:
    """[2, 4, 6] -> 「玩家2、玩家4、玩家6」。"""
    if not seats:
        return "无"
    return "、".join(f"玩家{s}" for s in sorted(seats))


def validate_roles(roles: Sequence[str], total_players: int) -> tuple[str, ...]:
    """检查一套角色配置是否合法，返回规范化的 tuple。"""
    roles = tuple(roles)

    if len(roles) != total_players:
        raise ValueError(f"角色数 {len(roles)} 与人数 {total_players} 不符")
    if total_players not in QUEST_SIZES:
        raise ValueError(f"不支持 {total_players} 人局，可选：{sorted(QUEST_SIZES)}")

    unknown = [r for r in roles if r not in ALL_ROLES]
    if unknown:
        raise ValueError(f"未知角色：{unknown}，可选：{ALL_ROLES}")

    for role in _UNIQUE_ROLES:
        n = roles.count(role)
        if n > 1:
            raise ValueError(f"{role} 只能有一个，你给了 {n} 个")

    for required in (MERLIN, ASSASSIN):
        if required not in roles:
            raise ValueError(f"必须有 {required}，否则这局没法正常结算")

    n_evil = sum(1 for r in roles if is_evil(r))
    if n_evil == 0 or n_evil == total_players:
        raise ValueError(f"必须同时有红蓝双方，当前红方 {n_evil} 人")

    return roles


@dataclass(frozen=True)
class Assignment:
    """一局的角色分配结果。"""

    #: 座位号 -> 角色
    roles: dict[int, str]
    #: 随机种子，便于复现这一局
    seed: int | None = None
    #: 私有信息里是否点名同伙的角色
    reveal_teammate_roles: bool = False

    def __post_init__(self) -> None:
        bad = [s for s in self.roles if not isinstance(s, int) or s < 1]
        if bad:
            raise ValueError(f"座位号必须是 1 及以上的正整数，收到：{bad}")

    # ---- 基本查询 ---------------------------------------------------- #

    @property
    def seats(self) -> tuple[int, ...]:
        return tuple(sorted(self.roles))

    @property
    def total_players(self) -> int:
        return len(self.roles)

    def role_of(self, seat: int) -> str:
        try:
            return self.roles[seat]
        except KeyError:
            raise KeyError(f"没有 {seat} 号位") from None

    def seats_with(self, role: str) -> list[int]:
        return sorted(s for s, r in self.roles.items() if r == role)

    def one_seat_with(self, role: str) -> int:
        seats = self.seats_with(role)
        if len(seats) != 1:
            raise ValueError(f"{role} 有 {len(seats)} 个座位，无法唯一确定")
        return seats[0]

    # ---- 阵营 -------------------------------------------------------- #

    def evil_seats(self) -> list[int]:
        """所有红方，含奥伯伦。"""
        return sorted(s for s, r in self.roles.items() if is_evil(r))

    def good_seats(self) -> list[int]:
        return sorted(s for s, r in self.roles.items() if not is_evil(r))

    def evil_network(self) -> list[int]:
        """互相认识的红方——**不含奥伯伦**。

        私密频道的成员就是这批人。
        """
        return sorted(
            s for s, r in self.roles.items() if is_evil(r) and r != OBERON
        )

    # ---- 私有信息 ---------------------------------------------------- #

    def private_info(self, seat: int) -> str:
        """该座位开局应该看到的那段文字，直接喂给 create_agent(private_info=...)。"""
        role = self.role_of(seat)

        if role == MERLIN:
            visible = [s for s in self.evil_seats() if self.role_of(s) != MORDRED]
            text = f"你看到这些玩家是红方：{fmt_seats(visible)}。"
            if len(visible) < len(self.evil_seats()):
                text += "（还有一名红方对你隐身，你不会看到他。）"
            return text

        if role == PERCIVAL:
            pair = sorted(self.seats_with(MERLIN) + self.seats_with(MORGANA))
            if len(pair) != 2:
                return "你没有额外信息。"
            a, b = pair
            return (
                f"玩家{a} 和 玩家{b} 这两个人里，一个是梅林，一个是莫甘娜。"
                f"你知道这件事，但分不清谁是谁。"
            )

        if role == LOYAL:
            return "你没有额外信息。"

        if role == OBERON:
            return (
                "你是孤狼奥伯伦：你看不到其他红方，他们也看不到你。"
                "没有人会掩护你，你也拿不到任何人的掩护。"
            )

        # 其余红方：莫甘娜 / 刺客 / 莫德雷德
        mates = [s for s in self.evil_network() if s != seat]
        if not self.reveal_teammate_roles:
            return f"你的同伙：{fmt_seats(mates)}。"
        detail = "、".join(
            f"玩家{s}（{self.role_of(s)}）" for s in mates
        )
        return f"你的同伙：{detail or '无'}。"

    # ---- 裁判视角（只用于日志，绝不能进 prompt） ---------------------- #

    def referee_sheet(self) -> str:
        lines = [f"{self.total_players} 人局（seed={self.seed}）"]
        for seat in self.seats:
            role = self.role_of(seat)
            side = "红" if is_evil(role) else "蓝"
            lines.append(f"  玩家{seat:<2} [{side}] {role}")
        return "\n".join(lines)


def assign_roles(
    total_players: int,
    *,
    roles: Sequence[str] | None = None,
    seed: int | None = None,
    reveal_teammate_roles: bool = False,
) -> Assignment:
    """把角色随机发到座位上。

    Args:
        total_players: 人数，5~10。
        roles: 自定义角色配置；不传就用 ROLE_TABLE 里的标准配置。
        seed: 随机种子。**建议每局都记下来**，否则出了问题无法复现。
        reveal_teammate_roles: 私有信息里是否点明同伙的角色。默认按官方规则
            只说座位。

    Returns:
        Assignment。
    """
    pool = list(roles) if roles is not None else list(ROLE_TABLE[total_players])
    pool = list(validate_roles(pool, total_players))

    rng = random.Random(seed)
    rng.shuffle(pool)

    seats = list(range(1, total_players + 1))
    return Assignment(
        roles=dict(zip(seats, pool)),
        seed=seed,
        reveal_teammate_roles=reveal_teammate_roles,
    )

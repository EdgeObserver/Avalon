"""纯逻辑测试：不调用任何模型。

覆盖四层：
1. roles  —— 角色分配与私有信息的口径
2. rules  —— 状态校验与动作合法性
3. human  —— 人类输入解析
4. engine —— 三条胜负路径（用假 agent 跑完整对局）
"""

from __future__ import annotations

import re

import pytest
from agentscope.message import Msg, TextBlock

from engine import AvalonGame
from human import parse_action
from prompts import (
    ASSASSIN,
    LOYAL,
    MERLIN,
    MORDRED,
    MORGANA,
    OBERON,
    PERCIVAL,
    is_evil,
)
from roles import ROLE_TABLE, Assignment, assign_roles, validate_roles
from rules import (
    ApprovalVote,
    Assassination,
    EliminationVote,
    GameState,
    QuestCard,
    Speech,
    TeamProposal,
    check_approval,
    check_assassination,
    check_elimination_vote,
    check_quest_card,
    check_speech,
    check_team_proposal,
)

# --------------------------------------------------------------------------- #
# roles
# --------------------------------------------------------------------------- #


def seen_seats(text: str) -> set[int]:
    """从私有信息里解析座位号。

    不能用 `"玩家1" in text` —— 「玩家10」也包含「玩家1」。
    """
    return {int(m) for m in re.findall(r"玩家(\d+)", text)}


@pytest.mark.parametrize("n", sorted(ROLE_TABLE))
def test_assignment_invariants(n: int) -> None:
    a = assign_roles(n, seed=1234)

    assert a.seats == tuple(range(1, n + 1))
    assert len(a.roles) == n

    evil = set(a.evil_seats())
    network = set(a.evil_network())

    # 奥伯伦是红方，但不在互认网络里
    if a.seats_with(OBERON):
        oberon = a.one_seat_with(OBERON)
        assert oberon in evil and oberon not in network
        assert seen_seats(a.private_info(oberon)) == set()

    # 梅林看到「红方 - 莫德雷德」
    merlin = a.one_seat_with(MERLIN)
    expect = {s for s in evil if a.role_of(s) != MORDRED}
    assert seen_seats(a.private_info(merlin)) == expect

    # 派西维尔看到梅林和莫甘娜
    percival = a.one_seat_with(PERCIVAL)
    assert seen_seats(a.private_info(percival)) == {
        a.one_seat_with(MERLIN),
        a.one_seat_with(MORGANA),
    }

    # 红方互认：同伙 = 网络 - 自己
    for seat in network:
        assert seen_seats(a.private_info(seat)) == network - {seat}


def test_assignment_is_reproducible() -> None:
    assert assign_roles(7, seed=42).roles == assign_roles(7, seed=42).roles
    assert assign_roles(7, seed=42).roles != assign_roles(7, seed=43).roles


@pytest.mark.parametrize(
    "roles, why",
    [
        ((MERLIN, PERCIVAL, MORGANA), "人数不符"),
        ((MERLIN, MERLIN, MORGANA, ASSASSIN, LOYAL), "两个梅林"),
        ((PERCIVAL, LOYAL, MORGANA, ASSASSIN, LOYAL), "没有梅林"),
        ((MERLIN, LOYAL, LOYAL, LOYAL, LOYAL), "没有红方"),
    ],
)
def test_invalid_role_configs_rejected(roles, why) -> None:
    with pytest.raises(ValueError):
        validate_roles(roles, len(roles))


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #


def test_game_state_rejects_bad_seats() -> None:
    for bad in ((0, 1, 2), (-1, 1, 2), ("a", 1)):
        with pytest.raises(ValueError):
            GameState(seats=bad)
    with pytest.raises(ValueError):
        Assignment(roles={0: MERLIN})


def test_quest_table_and_double_fail_rule() -> None:
    g7 = GameState(seats=(1, 2, 3, 4, 5, 6, 7))
    assert [GameState(seats=g7.seats, round_index=i).team_size for i in range(5)] == [
        2, 3, 3, 4, 4,
    ]
    assert GameState(seats=g7.seats, round_index=3).fails_needed == 2
    assert GameState(seats=(1, 2, 3, 4, 5, 6), round_index=3).fails_needed == 1


def test_team_proposal_legality() -> None:
    st = GameState(seats=(1, 2, 3, 4, 5, 6, 7))
    assert check_team_proposal({"team": [1, 2]}, st) == []
    assert check_team_proposal({"team": [1, 2, 3]}, st)  # 人数不对
    assert check_team_proposal({"team": [1, 1]}, st)  # 重复

    st.retire(3)
    problems = check_team_proposal({"team": [1, 3]}, st)
    assert any("不在场" in p for p in problems)
    assert "[3]" not in str(st.legal_targets())


def test_approval_vote_legality() -> None:
    st = GameState(seats=(1, 2, 3, 4, 5))
    assert check_approval({"approve": "同意"}, st, voter=1) == []
    st.retire(4)
    assert check_approval({"approve": "同意"}, st, voter=4)


def test_quest_card_legality() -> None:
    st = GameState(seats=(1, 2, 3, 4, 5))
    assert check_quest_card({"card": "成功"}, st, actor=1, is_evil=False) == []
    # 好人不能出失败牌——这是规则违规，不是策略选择
    assert check_quest_card({"card": "失败"}, st, actor=1, is_evil=False)
    assert check_quest_card({"card": "失败"}, st, actor=2, is_evil=True) == []


def test_assassination_and_elimination_legality() -> None:
    st = GameState(seats=(1, 2, 3, 4, 5))
    assert check_assassination({"target": 1}, st, actor=3) == []
    assert check_assassination({"target": 3}, st, actor=3)  # 不能指认自己

    st.retire(4)
    assert check_assassination({"target": 4}, st, actor=3)
    assert check_elimination_vote({"target": None}, st, voter=1) == []  # 弃权合法
    assert check_elimination_vote({"target": 4}, st, voter=1)  # 已出局
    assert check_elimination_vote({"target": 1}, st, voter=1)  # 不能投自己


def test_speech_legality() -> None:
    st = GameState(seats=(1, 2, 3, 4, 5))
    assert check_speech({"text": "我觉得玩家2可疑"}, st, actor=1) == []
    assert check_speech({"text": "   "}, st, actor=1)
    assert check_speech({"text": "长" * 400}, st, actor=1)


def test_five_rejections_end_the_game() -> None:
    st = GameState(seats=(1, 2, 3, 4, 5))
    assert [st.record_rejection() for _ in range(4)] == [False] * 4
    assert st.record_rejection() is True


# --------------------------------------------------------------------------- #
# human 输入解析
# --------------------------------------------------------------------------- #


def test_parse_actions() -> None:
    assert parse_action(TeamProposal, "2,4 | 试试") == {
        "team": [2, 4], "reason": "试试"
    }
    assert parse_action(ApprovalVote, "同意") == {"approve": "同意"}
    assert parse_action(ApprovalVote, "反对") == {"approve": "反对"}
    assert parse_action(QuestCard, "成功") == {"card": "成功"}
    assert parse_action(Assassination, "5") == {"target": 5, "reasoning": ""}
    assert parse_action(EliminationVote, "弃权") == {"target": None, "reason": ""}
    assert parse_action(Speech, "随便说点什么") == {"text": "随便说点什么"}


@pytest.mark.parametrize("text", ["0", "-1", "1,-2", "1, 0"])
def test_parse_rejects_non_positive_seats(text: str) -> None:
    # 关键："-1" 不能被静默解析成 1
    with pytest.raises(ValueError):
        parse_action(TeamProposal, text)


def test_parse_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_action(Speech, "   ")


# --------------------------------------------------------------------------- #
# engine：用假 agent 跑完整对局
# --------------------------------------------------------------------------- #


class FakeAgent:
    """按固定策略行动的假 agent，接口与 Agent 对齐。"""

    def __init__(self, seat, assignment, state, *, sabotage=False, approve=True,
                 prefer_evil=False, assassin_hits_merlin=False):
        self.name = f"玩家{seat}"
        self.seat = seat
        self.a = assignment
        self.state = state
        self.sabotage = sabotage
        self.approve = approve
        self.prefer_evil = prefer_evil
        self.assassin_hits_merlin = assassin_hits_merlin

    async def observe(self, msgs) -> None:
        return None

    def _team(self) -> list[int]:
        active = sorted(self.state.active)
        evil = [s for s in active if is_evil(self.a.role_of(s))]
        good = [s for s in active if not is_evil(self.a.role_of(s))]
        order = (evil + good) if self.prefer_evil else (good + evil)
        return order[: self.state.team_size]

    async def reply(self, msg, structured_schema=None):
        if structured_schema is TeamProposal:
            out = {"team": self._team(), "reason": ""}
        elif structured_schema is ApprovalVote:
            out = {"approve": "同意" if self.approve else "反对"}
        elif structured_schema is QuestCard:
            i_am_evil = is_evil(self.a.role_of(self.seat))
            out = {"card": "失败" if (self.sabotage and i_am_evil) else "成功"}
        elif structured_schema is Speech:
            out = {"text": "先听大家的。"}
        elif structured_schema is EliminationVote:
            out = {"target": None, "reason": ""}
        elif structured_schema is Assassination:
            if self.assassin_hits_merlin:
                target = self.a.one_seat_with(MERLIN)
            else:
                target = next(
                    s for s in self.a.good_seats() if self.a.role_of(s) != MERLIN
                )
            out = {"target": target, "reasoning": "测试"}
        else:  # pragma: no cover
            raise AssertionError(structured_schema)

        return Msg(
            name=self.name,
            role="assistant",
            content=[TextBlock(type="text", text="")],
            structured_output=out,
        )


async def play(seed: int = 1, **cfg):
    a = assign_roles(5, seed=seed)
    state = GameState(seats=a.seats)
    agents = {s: FakeAgent(s, a, state, **cfg) for s in a.seats}
    game = AvalonGame(a, agents, state=state, verbose=False)
    return a, game, await game.play()


@pytest.mark.asyncio
async def test_blue_wins_when_merlin_survives() -> None:
    a, game, result = await play(seed=1)
    assert result is not None  # 曾经因为少写 return 变成 None
    assert result.winner == "蓝方"
    assert result.successes == 3
    assert result.assassinated != a.one_seat_with(MERLIN)


@pytest.mark.asyncio
async def test_red_wins_by_three_failed_quests() -> None:
    _, _, result = await play(seed=1, sabotage=True, prefer_evil=True)
    assert result.winner == "红方"
    assert result.fails == 3


@pytest.mark.asyncio
async def test_red_wins_by_five_rejections() -> None:
    _, _, result = await play(seed=1, approve=False)
    assert result.winner == "红方"
    assert "否决" in result.reason


@pytest.mark.asyncio
async def test_assassination_hit_flips_to_red() -> None:
    a, _, result = await play(seed=3, assassin_hits_merlin=True)
    assert result.winner == "红方"
    assert result.assassinated == a.one_seat_with(MERLIN)


@pytest.mark.asyncio
async def test_events_and_ledger_are_recorded() -> None:
    _, game, _ = await play(seed=5, sabotage=True, prefer_evil=True)
    kinds = {e["type"] for e in game.events}
    assert {"round_start", "proposal", "quest_result"} <= kinds
    # 沉船后必须进入追责
    assert "accountability_start" in kinds
    ledger = game._public_ledger()
    assert "公开记录" in ledger

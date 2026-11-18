"""阿瓦隆主循环：把 state / roles / agents / actions 串成一局游戏。

    组队 → 全员投票 → (否决: 换队长 / 5 次判负) → 出任务牌 → 公布结果 → 下一轮
                                              └→ 蓝方 3 胜 → 刺杀 → 终局

三个实现要点：

1. **投票是公开的**，所以投票结束后一次性广播汇总，而不是逐个 observe
   （逐个会把 5 条消息塞进每个人的上下文）。
2. **任务牌是保密的**，收集期间谁都不 observe 别人的牌，只公布「本轮成功/失败」。
   但日志里会单独记一份 `secret`，方便事后做研究——那份数据绝不回灌给 agent。
3. 只广播**主持人公告**。玩家自己的发言已经作为 assistant 回合留在自己记忆里。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Literal

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock, UserMsg

from actions import act
from prompts import MERLIN, is_evil
from roles import Assignment, fmt_seats
from rules import (
    ApprovalVote,
    Assassination,
    EliminationVote,
    GameState,
    QuestCard,
    Reflection,
    Speech,
    TeamProposal,
    check_approval,
    check_assassination,
    check_elimination_vote,
    check_quest_card,
    check_speech,
    check_team_proposal,
)

logger = logging.getLogger(__name__)

Winner = Literal["蓝方", "红方"]


@dataclass
class VoteRecord:
    """一次公开投票。outcome 在队伍真的出发后才回填。"""

    round: int
    leader: int
    team: list[int]
    ballot: dict[int, str]
    outcome: str | None = None  # "成功" / "失败" / None=被否决、未出发

    @property
    def approved(self) -> bool:
        yes = sum(1 for v in self.ballot.values() if v == "同意")
        return yes * 2 > len(self.ballot)


@dataclass
class GameResult:
    winner: Winner
    reason: str
    successes: int
    fails: int
    assassinated: int | None = None
    rounds_played: int = 0
    seed: int | None = None
    assignment: dict[int, str] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    elapsed: float = 0.0
    error: str | None = None

    def __str__(self) -> str:
        return (
            f"{self.winner}获胜（{self.reason}）"
            f" 任务 {self.successes} 成功 / {self.fails} 失败"
            f"，共 {self.rounds_played} 轮"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON 的 key 只能是字符串
        d["assignment"] = {str(k): v for k, v in self.assignment.items()}
        return d


class _GameOver(Exception):
    def __init__(self, winner: Winner, reason: str) -> None:
        super().__init__(reason)
        self.winner = winner
        self.reason = reason


class AvalonGame:
    def __init__(
        self,
        assignment: Assignment,
        agents: dict[int, Agent],
        *,
        state: GameState | None = None,
        verbose: bool = True,
        reveal_eliminated_role: bool = True,
    ) -> None:
        missing = set(assignment.seats) - set(agents)
        if missing:
            raise ValueError(f"缺少这些座位的 agent：{sorted(missing)}")

        self.a = assignment
        self.agents = agents
        self.state = state or GameState(seats=assignment.seats)
        self.verbose = verbose
        #: 淘汰后是否公开阵营。关掉的话，别人无法判断投出去的是好人还是坏人，
        #: 「误杀好人 → 投他的人可疑」这条推理链就断了。
        self.reveal_eliminated_role = reveal_eliminated_role

        self.history: list[str] = []
        self.events: list[dict] = []
        self.vote_history: list[VoteRecord] = []
        self.successes = 0
        self.fails = 0
        self.rounds_played = 0
        self._t0 = time.time()
        #: 最近一次追责投票的票数，给 UI 画圆桌用
        self.last_tally: dict[int, int] | None = None
        #: 最近一次被淘汰的座位
        self.last_eliminated: int | None = None
        #: 历次追责记录（含票型与出局者）
        self.elimination_history: list[dict] = []
        #: play() 的最终结果，复盘时要用
        self.result: GameResult | None = None

    # ------------------------------------------------------------------ #
    # 对外
    # ------------------------------------------------------------------ #

    async def play(self) -> GameResult:
        try:
            while self.successes < 3 and self.fails < 3:
                if self.state.round_index >= 5:
                    raise _GameOver("红方", "5 轮打完蓝方仍未拿到 3 胜")
                await self._play_round()
            return await self._assassination()
        except _GameOver as e:
            return self._result(e.winner, e.reason)

    # ------------------------------------------------------------------ #
    # 一轮
    # ------------------------------------------------------------------ #

    async def _play_round(self) -> None:
        r = self.state.round_index + 1
        self.rounds_played = r
        size = self.state.team_size
        self._say(f"\n===== 第 {r} 轮（需 {size} 人，{self.state.fails_needed} 张失败牌判负）=====")
        self._event(
            "round_start", round=r, team_size=size,
            fails_needed=self.state.fails_needed,
            successes=self.successes, fails=self.fails,
        )

        # ---- 组队（可能反复被否决） ----
        rejections_this_round = 0
        while True:
            leader = self.state.leader
            team, reason = await self._propose(leader, r, size)
            approved, tally, ballot = await self._vote(leader, team, r)
            self._event(
                "vote_rejected" if not approved else "team_approved",
                round=r, leader=leader, team=team, ballot=ballot,
                tally=tally, rejections_this_round=rejections_this_round + (0 if approved else 1),
            )

            if approved:
                self._say(f"  队伍通过：{fmt_seats(team)}")
                break

            self._say(f"  队伍被否决（{tally}），队长移交")
            rejections_this_round += 1
            if self.state.record_rejection():
                raise _GameOver("红方", "连续 5 次组队被否决")
            self.state.rotate_leader()

        # ---- 出任务牌（保密收集） ----
        fails, cards = await self._collect_cards(team, r)
        needed = self.state.fails_needed
        outcome = "失败" if fails >= needed else "成功"
        if outcome == "失败":
            self.fails += 1
        else:
            self.successes += 1

        detail = f"{fails} 张失败牌，需 {needed} 张"
        # 回填这一轮「出发了的那次投票」的事后结果
        if self.vote_history:
            self.vote_history[-1].outcome = outcome
        self._say(f"  任务{outcome}（{detail}）｜累计 {self.successes} 成功 / {self.fails} 失败")
        self._event(
            "quest_result", round=r, team=team, fails=fails,
            fails_needed=needed, outcome=outcome,
            successes=self.successes, fails_total=self.fails,
            secret={str(k): v for k, v in cards.items()},
        )
        await self._announce(
            f"第 {r} 轮结果：任务{outcome}（{detail}）。"
            f"目前累计 {self.successes} 轮成功、{self.fails} 轮失败。"
        )

        # ---- 沉船追责：失败但还没到第 3 次（否则就终局了） ----
        if outcome == "失败" and self.fails < 3:
            await self._accountability(r)

        if self.successes >= 3:
            return
        if self.fails >= 3:
            raise _GameOver("红方", "3 轮任务失败")

        self.state.next_round()

    # ------------------------------------------------------------------ #
    # 各阶段
    # ------------------------------------------------------------------ #

    async def _propose(self, leader: int, r: int, size: int) -> tuple[list[int], str]:
        prompt = (
            f"第 {r} 轮，你是队长「玩家{leader}」。"
            f"请从在场玩家中挑选 {size} 人执行任务（可以选自己）。"
            f"在场玩家：{fmt_seats(sorted(self.state.active))}"
        )
        res = await act(
            self.agents[leader], prompt, TeamProposal,
            legal=lambda v: check_team_proposal(v, self.state),
            fallback=lambda: {
                "team": sorted(self.state.active)[: self.state.team_size],
                "reason": "（脚本补位）",
            },
        )
        team = list(res.value["team"])
        reason = res.value.get("reason") or ""
        self._say(
            f"  [第{r}轮] 队长 玩家{leader} 提议 {fmt_seats(team)}"
            + (f"（{reason}）" if reason else "")
        )
        self._event(
            "proposal", round=r, leader=leader, team=team,
            reason=reason, source=res.source, attempts=res.attempts,
        )
        await self._announce(f"队长 玩家{leader} 提议的队伍是：{fmt_seats(team)}。")
        return team, reason

    async def _vote(
        self, leader: int, team: list[int], r: int
    ) -> tuple[bool, str, dict[str, str]]:
        prompt = (
            f"第 {r} 轮，队长 玩家{leader} 提议的队伍是：{fmt_seats(team)}。"
            f"请投票，只能选「同意」或「反对」。"
        )

        async def one(seat: int) -> tuple[int, str]:
            res = await act(
                self.agents[seat], prompt, ApprovalVote,
                legal=lambda v, s=seat: check_approval(v, self.state, voter=s),
                fallback=lambda: {"approve": "反对"},
            )
            return seat, res.value["approve"]

        # 投票本来就是同时进行的，并发收集既符合规则也省时间
        results = await asyncio.gather(*(one(s) for s in self.state.voters()))
        votes = dict(results)

        yes = sum(1 for v in votes.values() if v == "同意")
        approved = yes * 2 > len(votes)
        detail = "、".join(
            f"玩家{s}{'同意' if v == '同意' else '反对'}" for s, v in sorted(votes.items())
        )
        tally = f"{yes} 同意 / {len(votes) - yes} 反对"
        self._say(f"  投票：{detail} → {tally}")
        await self._announce(f"投票结果：{detail}。共 {tally}。")

        # 记进公开投票档案，刺杀阶段会汇总给刺客
        self.vote_history.append(
            VoteRecord(round=r, leader=leader, team=list(team), ballot=dict(votes))
        )
        return approved, tally, {str(k): v for k, v in votes.items()}

    async def _collect_cards(self, team: list[int], r: int) -> tuple[int, dict[int, str]]:
        """秘密收集任务牌。收集期间不向任何人广播。"""

        async def one(seat: int) -> str:
            evil = is_evil(self.a.role_of(seat))
            res = await act(
                self.agents[seat],
                f"你被选入第 {r} 轮的任务队伍。请出任务牌。",
                QuestCard,
                legal=lambda v, s=seat, e=evil: check_quest_card(
                    v, self.state, actor=s, is_evil=e
                ),
                fallback=lambda: {"card": "成功"},
            )
            return res.value["card"]

        # 同时出牌；结果只在内部统计，绝不 observe 给别人
        cards_list = await asyncio.gather(*(one(s) for s in team))
        cards = dict(zip(team, cards_list))
        return sum(1 for c in cards_list if c == "失败"), cards

    async def _accountability(self, r: int) -> None:
        """沉船追责：从队长开始轮流发言，然后投票淘汰一人。

        梅林被淘汰 = 红方立即获胜。这条让「淘汰」变成一把双刃剑：
        好人想清掉可疑的人，坏人想借票把梅林做掉。
        """
        self._say("\n----- 追责环节 -----")
        self._event("accountability_start", round=r, active=sorted(self.state.active))

        # 把跨轮才能统计出来的公开信息聚合好再给所有人——这是「查票型」的前提
        ledger = self._public_ledger()
        await self._broadcast(UserMsg(name="主持人", content=ledger))
        self._event("ledger_sent", where="accountability", round=r, chars=len(ledger))

        # ---- 发言：队长起，按座位顺序 ----
        for seat in self._speak_order():
            if seat not in self.state.active:
                continue
            res = await act(
                self.agents[seat],
                f"第 {r} 轮任务失败，现在进入追责环节。"
                f"请发言：你怎么看这次失败，建议淘汰谁？",
                Speech,
                legal=lambda v, s=seat: check_speech(v, self.state, actor=s),
                fallback=lambda: {"text": "我没有更多线索，先听大家的判断。"},
            )
            text = res.value["text"].strip()
            self._say(f"  玩家{seat}：{text}")
            self._event("speech", round=r, seat=seat, text=text)
            await self._broadcast(
                Msg(
                    name=f"玩家{seat}",
                    role="assistant",
                    content=[TextBlock(type="text", text=text)],
                ),
                exclude={seat},
            )

        # ---- 投票（同时进行，互相看不到） ----
        async def one(seat: int) -> tuple[int, int | None]:
            res = await act(
                self.agents[seat],
                "追责投票：请选择要淘汰的玩家。没有把握就弃权（target 填 null）。",
                EliminationVote,
                legal=lambda v, s=seat: check_elimination_vote(
                    v, self.state, voter=s
                ),
                fallback=lambda: {"target": None, "reason": "（脚本弃权）"},
            )
            return seat, res.value.get("target")

        results = await asyncio.gather(*(one(s) for s in self.state.voters()))
        votes = dict(results)

        counts = Counter(t for t in votes.values() if t is not None)
        tally = dict(counts)
        self.last_tally = tally
        self.last_eliminated = None

        detail = "、".join(
            f"玩家{s}→{'弃权' if t is None else f'玩家{t}'}" for s, t in sorted(votes.items())
        )
        self._say(f"  投票：{detail}")
        self._event(
            "elimination_vote", round=r,
            ballot={str(k): v for k, v in votes.items()},
            tally={str(k): v for k, v in tally.items()},
        )

        eliminated: int | None = None
        if counts:
            top = max(counts.values())
            leaders = [s for s, c in counts.items() if c == top]
            if len(leaders) == 1:
                eliminated = leaders[0]

        role = self.a.role_of(eliminated) if eliminated is not None else None
        self.elimination_history.append(
            {
                "round": r,
                "ballot": dict(votes),
                "eliminated": eliminated,
                "role": role if self.reveal_eliminated_role else None,
            }
        )

        if eliminated is None:
            self._say("  平票或全员弃权，无人出局")
            await self._announce("追责投票没有形成多数，本轮无人出局。")
            return

        self.state.retire(eliminated)
        self.last_eliminated = eliminated
        self._say(f"  玩家{eliminated} 被淘汰（{role}）")
        self._event(
            "eliminated", round=r, seat=eliminated, role=role,
            tally={str(k): v for k, v in tally.items()},
        )

        if self.reveal_eliminated_role:
            camp = "红方（坏人）" if is_evil(role) else "蓝方（好人）"
            await self._announce(
                f"玩家{eliminated} 被投票淘汰，退出本局。公开其阵营：{camp}。"
            )
        else:
            await self._announce(f"玩家{eliminated} 被投票淘汰，退出本局。（阵营不公开）")

        if role == MERLIN:
            raise _GameOver("红方", "梅林被投票淘汰")

    def _speak_order(self) -> list[int]:
        """从当前队长开始，按座位顺时针。"""
        order = sorted(self.state.active)
        if not order:
            return []
        idx = order.index(self.state.leader) if self.state.leader in order else 0
        return order[idx:] + order[:idx]

    async def _assassination(self) -> GameResult:
        assassin = self.a.one_seat_with("刺客")
        self._say("\n===== 刺杀阶段 =====")
        self._event("assassination_phase")
        await self._announce("好人已经完成 3 轮任务。现在是刺杀阶段，请刺客指认梅林。")

        # 把「跨轮才能统计出来」的信息聚合好再给模型。这些全是公开信息，
        # 提前算好不算作弊——否则模型得自己在自然语言里数票，小模型数不动。
        matrix = self._public_ledger()
        await self.agents[assassin].observe(UserMsg(name="主持人", content=matrix))
        self._event("ledger_sent", where="assassination", chars=len(matrix))

        res = await act(
            self.agents[assassin],
            "请指认你认为是梅林的玩家。可以参考上面这份投票档案。",
            Assassination,
            legal=lambda v: check_assassination(v, self.state, actor=assassin),
            fallback=lambda: {
                "target": next(
                    s for s in sorted(self.state.active) if s != assassin
                ),
                "reasoning": "（脚本兜底）",
            },
        )
        target = int(res.value["target"])
        hit = self.a.role_of(target) == "梅林"
        self._say(
            f"  刺客 玩家{assassin} 指认 玩家{target}"
            f"（{'命中梅林' if hit else '指认错误'}）"
        )
        self._event(
            "assassination", assassin=assassin, target=target, hit=hit,
            reasoning=res.value.get("reasoning", ""), source=res.source,
        )
        return self._result(
            "红方" if hit else "蓝方",
            "刺客刺杀梅林命中" if hit else "梅林幸存",
            assassinated=target,
        )

    # ------------------------------------------------------------------ #
    # 局后复盘（不影响对局，仅供观战）
    # ------------------------------------------------------------------ #

    async def reflect(self, *, skip: set[int] | None = None) -> dict[int, str]:
        """让每个 agent 写一段赛后心路历程。

        只在 UI 里调用，不写进 events、不影响胜负。真相已经公开，所以这里
        会把完整身份表告诉它们——要的是「诚实复盘」，不是马后炮。
        """
        if self.result is None:
            return {}

        skip = skip or set()
        reveal = "、".join(f"玩家{s}={self.a.role_of(s)}" for s in self.a.seats)
        prompt = (
            f"对局已经结束。结果：{self.result}。\n"
            f"所有人的真实身份：{reveal}。\n\n"
            "请诚实复盘，150 字以内，像跟朋友聊天那样直接说：\n"
            "你的真实目标是什么、你做过哪些关键动作（组队/投票/破坏/指控）、"
            "哪一步判断对了、哪一步错了、如果重来会怎么改。\n"
            "不要分点罗列，不要客套。"
        )

        async def one(seat: int) -> tuple[int, str]:
            try:
                msg = await self.agents[seat].reply(
                    UserMsg(name="主持人", content=prompt),
                    structured_schema=Reflection,
                )
                return seat, ((msg.structured_output or {}).get("text") or "").strip()
            except Exception as e:  # noqa: BLE001 - 复盘失败不该影响结果展示
                logger.warning("玩家%d 复盘失败：%s", seat, e)
                return seat, ""

        seats = [s for s in self.a.seats if s not in skip]
        pairs = await asyncio.gather(*(one(s) for s in seats))
        return {s: t for s, t in pairs if t}

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #

    def _public_ledger(self) -> str:
        """把散落在对话里的公开信息聚合成一张表。

        模型没法在自然语言里自己数票（实测过：投票记录都在上下文里，它还是
        统计不出来）。所以每到需要「查历史」的节点——追责、刺杀——就把统计
        结果直接递给它。全是公开信息，不算给谁开挂。
        """
        lines = ["# 公开记录（所有玩家都看得到，这里是汇总版）"]

        if self.vote_history:
            lines += ["", "## 每轮的组队与结果"]
            for rec in self.vote_history:
                passed = "通过" if rec.approved else "被否决"
                tail = f"任务{rec.outcome}" if rec.outcome else "队伍未出发"
                ballot = "、".join(
                    f"玩家{s}{'同意' if v == '同意' else '反对'}"
                    for s, v in sorted(rec.ballot.items())
                )
                lines.append(
                    f"- 第{rec.round}轮 队长玩家{rec.leader} 提议{fmt_seats(rec.team)}"
                    f" → {passed} → {tail}"
                )
                lines.append(f"    {ballot}")

            lines += ["", "## 每人投票统计"]
            for seat in sorted({s for rec in self.vote_history for s in rec.ballot}):
                total = oppose = oppose_fail = approve = approve_pass = 0
                for rec in self.vote_history:
                    v = rec.ballot.get(seat)
                    if v is None:
                        continue
                    total += 1
                    if v == "反对":
                        oppose += 1
                        oppose_fail += rec.outcome == "失败"
                    else:
                        approve += 1
                        approve_pass += rec.outcome == "成功"
                opp_txt = (
                    f"反对 {oppose} 次（其中 {oppose_fail} 次队伍事后失败）"
                    if oppose else "反对 0 次"
                )
                app_txt = (
                    f"赞成 {approve} 次（其中 {approve_pass} 次队伍成功）"
                    if approve else "赞成 0 次"
                )
                lines.append(f"- 玩家{seat}：投 {total} 票｜{opp_txt}｜{app_txt}")

        if self.elimination_history:
            lines += ["", "## 追责投票记录"]
            for e in self.elimination_history:
                ballot = "、".join(
                    f"玩家{k}→{'弃权' if v is None else f'玩家{v}'}"
                    for k, v in sorted(e["ballot"].items())
                )
                if e["eliminated"] is None:
                    tail = "→ 无人出局"
                elif e.get("role"):
                    camp = "红方" if is_evil(e["role"]) else "蓝方"
                    tail = f"→ 玩家{e['eliminated']} 出局（{camp}）"
                else:
                    tail = f"→ 玩家{e['eliminated']} 出局（阵营未公开）"
                lines.append(f"- 第{e['round']}轮：{ballot} {tail}")

        lines += [
            "",
            "读法：「反对精准度」= 反对票里队伍事后失败的比例，越高说明他越可能提前知情。",
            "注意：好人也会集体投错人，一次误杀只能当加权证据，不能当铁证。",
        ]
        return "\n".join(lines)

    async def _announce(self, text: str) -> None:
        await self._broadcast(UserMsg(name="主持人", content=text))

    async def _broadcast(self, msg: Msg, exclude: set[int] | None = None) -> None:
        """投递给所有还在场的玩家（可排除发送者自己）。"""
        exclude = exclude or set()
        for seat, agent in self.agents.items():
            if seat in exclude or seat not in self.state.active:
                continue
            await agent.observe(msg)

    def _say(self, text: str) -> None:
        self.history.append(text)
        if self.verbose:
            print(text)

    def _event(self, kind: str, **payload) -> None:
        self.events.append({"type": kind, "t": round(time.time() - self._t0, 3), **payload})

    def _result(
        self, winner: Winner, reason: str, *, assassinated: int | None = None
    ) -> GameResult:
        self.result = GameResult(
            winner=winner,
            reason=reason,
            successes=self.successes,
            fails=self.fails,
            assassinated=assassinated,
            rounds_played=self.rounds_played,
            seed=self.a.seed,
            assignment=dict(self.a.roles),
            events=self.events,
            elapsed=round(time.time() - self._t0, 2),
        )
        return self.result

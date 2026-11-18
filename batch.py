"""批量对局 + 指标统计 + JSONL 落盘。

用法::

    results = await run_batch(50, players=5, seed_base=1000)
    save_jsonl(results, "runs/5p_50games.jsonl")
    print_summary(summarize(results))

设计上的两点：

- **模型实例跨对局复用**，但 agent 每局重建。agent 带着上一局的记忆进新局，
  会让「红方知道上局谁骗过自己」这种脏数据混进来。
- 单局失败不会中断批次，会以 `error` 字段记录在结果里。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from agentscope.model import OpenAIChatModel

from agents import create_agent
from channels import PrivateChannel
from engine import AvalonGame, GameResult
from roles import assign_roles
from rules import GameState


async def play_one(
    seed: int,
    total_players: int,
    model: OpenAIChatModel,
    *,
    verbose: bool = False,
) -> GameResult:
    """跑一局。任何异常都转成带 error 字段的 GameResult。"""
    try:
        a = assign_roles(total_players, seed=seed)
        state = GameState(seats=a.seats)
        channel = PrivateChannel(a.evil_network())
        agents = {
            seat: create_agent(
                a.role_of(seat),
                seat,
                total_players=a.total_players,
                private_info=a.private_info(seat),
                private_channel=channel.label if seat in a.evil_network() else None,
                model=model,
            )
            for seat in a.seats
        }
        game = AvalonGame(a, agents, state=state, verbose=verbose)
        return await game.play()
    except Exception as e:  # noqa: BLE001 - 一局崩了不该毁掉整批
        return GameResult(
            winner="红方",
            reason=f"对局异常：{type(e).__name__}",
            successes=0,
            fails=0,
            seed=seed,
            error=f"{type(e).__name__}: {e}",
        )


async def run_batch(
    n_games: int,
    *,
    players: int = 5,
    seed_base: int = 0,
    model: OpenAIChatModel,
    max_concurrency: int = 3,
    verbose: bool = False,
) -> list[GameResult]:
    """并发跑 n 局，返回按 seed 排序的结果。"""
    sem = asyncio.Semaphore(max_concurrency)

    async def guarded(seed: int) -> GameResult:
        async with sem:
            if verbose:
                print(f"  -> 开始 seed={seed}")
            return await play_one(seed, players, model, verbose=verbose)

    seeds = [seed_base + i for i in range(n_games)]
    results = await asyncio.gather(*(guarded(s) for s in seeds))
    return sorted(results, key=lambda r: r.seed or 0)


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #


def summarize(results: list[GameResult]) -> dict:
    ok = [r for r in results if not r.error]
    n = len(ok)
    if n == 0:
        return {"games": 0, "error_games": len(results)}

    blue = sum(1 for r in ok if r.winner == "蓝方")
    red = n - blue
    assassinations = [r for r in ok if r.assassinated is not None]

    events = [e for r in ok for e in r.events]
    by_type = Counter(e["type"] for e in events)

    # 质量指标：降级次数、重试次数
    fallbacks = sum(1 for e in events if e.get("source") == "fallback")
    retried = sum(1 for e in events if (e.get("attempts") or 1) > 1)
    proposals = by_type["proposal"]

    return {
        "games": n,
        "error_games": len(results) - n,
        "blue_wins": blue,
        "red_wins": red,
        "blue_win_rate": round(blue / n, 3),
        "red_win_rate": round(red / n, 3),
        "avg_rounds": round(mean(r.rounds_played for r in ok), 2),
        "avg_elapsed_s": round(mean(r.elapsed for r in ok), 1),
        "reached_assassination": len(assassinations),
        "assassination_hit_rate": (
            round(sum(1 for r in assassinations if r.assassinated is not None
                      and r.assassinated in _merlin_seats(r)) / len(assassinations), 3)
            if assassinations else None
        ),
        "reasons": dict(Counter(r.reason for r in ok)),
        "proposals": proposals,
        "rejections": by_type["vote_rejected"],
        "rejection_rate": round(by_type["vote_rejected"] / proposals, 3) if proposals else 0,
        "fallback_events": fallbacks,
        "retried_events": retried,
        "retry_rate": round(retried / proposals, 3) if proposals else 0,
    }


def _merlin_seats(r: GameResult) -> set[int]:
    return {int(s) for s, role in r.assignment.items() if role == "梅林"}


def print_summary(s: dict) -> None:
    print("\n" + "=" * 46)
    print(f"对局数        {s['games']}（异常 {s['error_games']}）")
    print(f"蓝方胜率      {s['blue_win_rate']:.1%}  ({s['blue_wins']})")
    print(f"红方胜率      {s['red_win_rate']:.1%}  ({s['red_wins']})")
    print(f"平均轮数      {s['avg_rounds']}")
    print(f"平均耗时      {s['avg_elapsed_s']}s / 局")
    print(f"进入刺杀      {s['reached_assassination']} 局"
          + (f"，命中率 {s['assassination_hit_rate']:.1%}"
             if s["assassination_hit_rate"] is not None else ""))
    print(f"组队否决率    {s['rejection_rate']:.1%}"
          f"（{s['rejections']}/{s['proposals']}）")
    print(f"降级次数      {s['fallback_events']}")
    print(f"重试率        {s['retry_rate']:.1%}")
    print("\n获胜原因分布：")
    for reason, cnt in sorted(s["reasons"].items(), key=lambda kv: -kv[1]):
        print(f"  {cnt:>3}  {reason}")


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #


def save_jsonl(results: list[GameResult], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return p

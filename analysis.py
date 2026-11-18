"""从对局日志里挖行为指标。

读取 `save_jsonl` 产出的文件，不复跑对局。

关注三类问题：
1. **信息隔离有没有生效**——好人是否被塞进了只有坏人掌握的信息
2. **推理质量**——梅林的投票是否精准，刺客是否找对了人
3. **坏人是否克制**——破坏票的时机分布
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evil_seats(record: dict) -> set[int]:
    from prompts import is_evil

    return {int(s) for s, role in record["assignment"].items() if is_evil(role)}


def merlin_seat(record: dict) -> int | None:
    for s, role in record["assignment"].items():
        if role == "梅林":
            return int(s)
    return None


def analyze(records: list[dict]) -> dict:
    ok = [r for r in records if not r.get("error")]

    # ---- 队伍构成与任务结果 ----
    clean_teams = dirty_teams = 0
    clean_fails = dirty_fails = 0
    rounds_to_first_fail: list[int] = []

    # ---- 梅林投票倾向 ----
    merlin_approve_dirty = merlin_approve_clean = 0
    merlin_oppose_dirty = merlin_oppose_clean = 0

    # ---- 坏人破坏票 ----
    sab_events: list[tuple[int, int, bool]] = []  # (round, 队伍人数, 是否有人破坏)

    for r in ok:
        ev = evil_seats(r)
        merlin = merlin_seat(r)
        first_fail_round: int | None = None

        for e in r["events"]:
            kind = e["type"]

            if kind in ("team_approved", "vote_rejected"):
                team = {int(s) for s in e["team"]}
                dirty = bool(team & ev)
                ballot = e.get("ballot") or {}
                if merlin is not None and str(merlin) in ballot:
                    said = ballot[str(merlin)] == "同意"
                    if dirty:
                        merlin_approve_dirty += said
                        merlin_oppose_dirty += not said
                    else:
                        merlin_approve_clean += said
                        merlin_oppose_clean += not said
                if kind == "team_approved":
                    if dirty:
                        dirty_teams += 1
                    else:
                        clean_teams += 1

            elif kind == "quest_result":
                team = {int(s) for s in e["team"]}
                dirty = bool(team & ev)
                if dirty:
                    dirty_fails += e["outcome"] == "失败"
                    sab_events.append((e["round"], len(team), bool(e["fails"])))
                else:
                    clean_fails += e["outcome"] == "失败"
                if e["outcome"] == "失败" and first_fail_round is None:
                    first_fail_round = e["round"]

        if first_fail_round is not None:
            rounds_to_first_fail.append(first_fail_round)

    def rate(a: int, b: int) -> float | None:
        return round(a / b, 3) if b else None

    blue = sum(1 for r in ok if r["winner"] == "蓝方")
    assassinated = [r for r in ok if r.get("assassinated") is not None]
    hits = sum(1 for r in assassinated if r["assassinated"] == merlin_seat(r))

    return {
        "games": len(ok),
        "blue_win_rate": rate(blue, len(ok)),
        # 干净队伍 vs 含坏人队伍
        "clean_teams": clean_teams,
        "dirty_teams": dirty_teams,
        "clean_team_fail_rate": rate(clean_fails, clean_teams),
        "dirty_team_fail_rate": rate(dirty_fails, dirty_teams),
        "avg_round_to_first_fail": (
            round(mean(rounds_to_first_fail), 2) if rounds_to_first_fail else None
        ),
        # 梅林投票
        "merlin_oppose_rate_on_dirty": rate(merlin_oppose_dirty, merlin_oppose_dirty + merlin_approve_dirty),
        "merlin_oppose_rate_on_clean": rate(merlin_oppose_clean, merlin_oppose_clean + merlin_approve_clean),
        # 刺杀
        "assassination_games": len(assassinated),
        "assassination_hit_rate": rate(hits, len(assassinated)),
        # 破坏票时机
        "sabotage_by_round": {
            int(k): round(mean(v), 3)
            for k, v in _group(sab_events).items()
        },
        "winner_reasons": dict(Counter(r["reason"] for r in ok)),
    }


def _group(items):
    buckets: dict[int, list[int]] = defaultdict(list)
    for rnd, _size, sabotaged in items:
        buckets[rnd].append(int(sabotaged))
    return buckets


def print_analysis(m: dict) -> None:
    print("=" * 52)
    print(f"对局数                  {m['games']}")
    print(f"蓝方胜率                {m['blue_win_rate']:.1%}")
    print()
    print(f"干净队伍数/失败率        {m['clean_teams']} / {m['clean_team_fail_rate']:.1%}"
          if m["clean_team_fail_rate"] is not None else "干净队伍 0 次")
    print(f"含坏人队伍数/失败率      {m['dirty_teams']} / {m['dirty_team_fail_rate']:.1%}"
          if m["dirty_team_fail_rate"] is not None else "含坏人队伍 0 次")
    print(f"首次失败平均发生在第     {m['avg_round_to_first_fail']} 轮")
    print()
    print(f"梅林对含坏人队伍投反对   {_pct(m['merlin_oppose_rate_on_dirty'])}")
    print(f"梅林对干净队伍投反对     {_pct(m['merlin_oppose_rate_on_clean'])}")
    print("  （两者差距越大，说明梅林越会用投票暗示，也越容易被刺客锁定）")
    print()
    print(f"进入刺杀                {m['assassination_games']} 局，命中率 {_pct(m['assassination_hit_rate'])}")
    print("\n含坏人队伍里出现破坏票的比例（按轮次）：")
    for rnd, v in sorted(m["sabotage_by_round"].items()):
        print(f"  第 {rnd} 轮  {v:.0%}")


def _pct(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "—"

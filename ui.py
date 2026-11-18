"""阿瓦隆网页界面（Gradio）。

结构很简单：对局跑在后台的 asyncio 任务里，UI 每秒轮询一次状态。

- 轮到人类时，`HumanPlayer.pending` 会变成非空，界面右侧出现输入框
- 提交后通过 `HumanPlayer.submit()` 把值塞回协程，对局继续
- 人类输入不合法时，引擎会带着错误信息再来问一次，界面自动显示新的提示

启动：

    conda activate avalon
    python ui.py
"""

from __future__ import annotations

import asyncio
import math
import random
import warnings
from dataclasses import dataclass

# Gradio 6 在队列里刷一堆 pandas 弃用警告，跟本项目无关，静音
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="gradio")
warnings.filterwarnings("ignore", message=".*copy keyword is deprecated.*")
warnings.filterwarnings("ignore", message=".*no_silent_downcasting.*")

import gradio as gr

from agents import build_model, create_agent
from channels import PrivateChannel
from engine import AvalonGame, GameResult
from human import HumanPlayer
from prompts import is_evil
from roles import assign_roles
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


def make_validator(state: GameState, seat: int, role: str):
    """复用 rules 里的校验函数，让人类输入在本地就被打回，不消耗引擎重试。"""

    def validate(schema, payload: dict) -> list[str]:
        if schema is TeamProposal:
            return check_team_proposal(payload, state)
        if schema is ApprovalVote:
            return check_approval(payload, state, seat)
        if schema is QuestCard:
            return check_quest_card(payload, state, seat, is_evil(role))
        if schema is Assassination:
            return check_assassination(payload, state, seat)
        if schema is Speech:
            return check_speech(payload, state, seat)
        if schema is EliminationVote:
            return check_elimination_vote(payload, state, seat)
        return []

    return validate


@dataclass
class Session:
    """一局游戏的运行时状态。"""

    assignment = None
    game: AvalonGame | None = None
    human: HumanPlayer | None = None
    task: asyncio.Task | None = None
    result: GameResult | None = None
    #: 局后复盘：座位 -> 一段话
    reflections: dict[int, str] | None = None
    reflecting: bool = False

    def reset(self) -> None:
        self.assignment = None
        self.game = None
        self.human = None
        self.task = None
        self.result = None
        self.reflections = None
        self.reflecting = False


SESS = Session()


async def start_game(total: int, seat: int, seed: int) -> None:
    SESS.reset()

    total, seat, seed = int(total), int(seat), int(seed)
    if not 5 <= total <= 10:
        raise gr.Error("人数只能是 5~10")
    if seat < 1 or seat > total:
        raise gr.Error(f"座位号必须是 1~{total} 之间的正整数（不能是 0 或负数）")

    a = assign_roles(total, seed=seed)
    state = GameState(seats=a.seats)
    channel = PrivateChannel(a.evil_network())
    model = build_model(
        temperature=0.85, max_tokens=200, session_id=f"avalon-ui-{seed}"
    )

    human = HumanPlayer(seat, validator=make_validator(state, seat, a.role_of(seat)))
    agents: dict = {}
    for s in a.seats:
        if s == seat:
            agents[s] = human
        else:
            agents[s] = create_agent(
                a.role_of(s), s,
                total_players=a.total_players,
                private_info=a.private_info(s),
                private_channel=channel.label if s in a.evil_network() else None,
                model=model,
            )

    SESS.assignment = a
    SESS.human = human
    SESS.game = AvalonGame(a, agents, state=state, verbose=False)
    SESS.task = asyncio.create_task(SESS.game.play())


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #


def render_identity() -> str:
    if SESS.assignment is None:
        return "### 你的身份\n点击「开始游戏」抽取身份。"
    if SESS.human is None:
        return "### 你的身份\n—"
    seat = SESS.human.seat
    role = SESS.assignment.role_of(seat)
    side = "红方（坏人）" if is_evil(role) else "蓝方（好人）"
    return (
        f"### 你是 **{role}**（{side}）· 玩家{seat}\n\n"
        f"> {SESS.assignment.private_info(seat)}\n\n"
        f"<sub>种子 {SESS.assignment.seed}｜"
        f"人数 {SESS.assignment.total_players}</sub>"
    )


def render_log() -> str:
    if SESS.game is None:
        return ""
    return "\n".join(SESS.game.history)


def render_table() -> str:
    """圆桌 + 人物：标出队长、你、出局者、最新票数。"""
    if SESS.assignment is None or SESS.game is None:
        return ""

    a, st = SESS.assignment, SESS.game.state
    n = len(a.seats)
    R, C = 118, 170
    done = SESS.task is not None and SESS.task.done()
    tally = SESS.game.last_tally or {}

    p = [f'<svg viewBox="0 0 340 340" width="330" height="330" '
         f'xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    # 桌面
    p.append(f'<circle cx="{C}" cy="{C}" r="{R + 44}" fill="#fbf8f1" stroke="#e3dbca" stroke-width="2"/>')
    p.append(f'<circle cx="{C}" cy="{C}" r="{R - 46}" fill="#f3ecdd" stroke="#e7ddc9"/>')
    round_no = min(st.round_index + 1, 5)
    p.append(f'<text x="{C}" y="{C - 6}" text-anchor="middle" font-size="14" fill="#8a7a60">第 {round_no} 轮</text>')
    p.append(f'<text x="{C}" y="{C + 16}" text-anchor="middle" font-size="12" fill="#a89a80">队长 玩家{st.leader}</text>')

    for i, seat in enumerate(a.seats):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        x, y = C + R * math.cos(ang), C + R * math.sin(ang)
        active = seat in st.active
        is_leader = active and seat == st.leader
        is_me = SESS.human is not None and seat == SESS.human.seat

        fill = "#e3ecff" if active else "#dcdcdc"
        stroke = "#c9a227" if is_leader else ("#3b6fd4" if is_me else "#c2cad6")
        swidth = 4 if is_leader else (3 if is_me else 1.5)

        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="25" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{swidth}"/>')
        p.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="15" '
                 f'font-weight="600" fill="{"#1f3b73" if active else "#9a9a9a"}">{seat}</text>')

        if is_leader:
            p.append(f'<text x="{x:.1f}" y="{y - 30:.1f}" text-anchor="middle" font-size="16">👑</text>')
        if is_me:
            p.append(f'<text x="{x:.1f}" y="{y + 42:.1f}" text-anchor="middle" font-size="11" '
                     f'fill="#3b6fd4" font-weight="600">你</text>')
        if not active:
            p.append(f'<text x="{x:.1f}" y="{y + 42:.1f}" text-anchor="middle" font-size="11" '
                     f'fill="#999">出局</text>')

        votes = tally.get(seat, 0)
        if votes:
            bx, by = x + 20, y - 20
            p.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="10" fill="#c0392b"/>')
            p.append(f'<text x="{bx:.1f}" y="{by + 4:.1f}" text-anchor="middle" font-size="11" '
                     f'fill="#fff" font-weight="700">{votes}</text>')

        if done:
            role = a.role_of(seat)
            color = "#b03a3a" if is_evil(role) else "#2f6f3e"
            p.append(f'<text x="{x:.1f}" y="{y + 42:.1f}" text-anchor="middle" font-size="10.5" '
                     f'fill="{color}">{role}</text>')

    p.append("</svg>")
    return "".join(p)


def render_action() -> str:
    if SESS.task is None:
        return "尚未开始。"
    if SESS.task.done():
        return "**对局结束。**"
    req = SESS.human.pending if SESS.human else None
    if req is None:
        return "⏳ 等待其他玩家行动……"
    return f"### 轮到你了：{req.kind}\n\n{req.prompt}\n\n**{req.hint}**"


def render_result() -> str:
    if SESS.task is None or not SESS.task.done():
        return ""
    if SESS.result is None:
        SESS.result = SESS.task.result()
    r = SESS.result

    lines = [
        f"## {r.winner}获胜",
        "",
        f"{r.reason}｜任务 {r.successes} 成功 / {r.fails} 失败｜共 {r.rounds_played} 轮",
    ]

    # 刺杀理由：刺客在这个时刻到底在想什么，比结果更有意思
    ev = next(
        (e for e in SESS.game.events if e["type"] == "assassination"), None
    )
    if ev:
        verdict = "命中梅林" if ev.get("hit") else "指认错误"
        reasoning = (ev.get("reasoning") or "").strip() or "（他没给出理由）"
        lines += [
            "",
            f"> **刺客 玩家{ev['assassin']} 指认 玩家{ev['target']}**（{verdict}）",
            f"> {reasoning}",
        ]

    table = ["", "| 座位 | 角色 | 阵营 |", "| --- | --- | --- |"]
    for seat in SESS.assignment.seats:
        role = SESS.assignment.role_of(seat)
        side = "🔴 红" if is_evil(role) else "🔵 蓝"
        me = " ← 你" if seat == SESS.human.seat else ""
        table.append(f"| {seat}{me} | {role} | {side} |")

    return "\n".join(lines + table)


def render_reflections() -> str:
    if SESS.game is None:
        return ""
    if SESS.reflections is None:
        return "### 心路历程\n正在让每个 agent 复盘…" if SESS.reflecting else ""

    lines = ["### 心路历程（局后复盘）", ""]
    for seat in SESS.assignment.seats:
        text = SESS.reflections.get(seat)
        if not text:
            continue
        role = SESS.assignment.role_of(seat)
        mark = "🔴" if is_evil(role) else "🔵"
        me = " · 你" if SESS.human and seat == SESS.human.seat else ""
        lines += [f"**{mark} 玩家{seat}（{role}）{me}**：{text}", ""]
    return "\n".join(lines) if len(lines) > 3 else ""


# --------------------------------------------------------------------------- #
# 事件
# --------------------------------------------------------------------------- #


async def on_start(total, seat, seed):
    await start_game(total, seat, seed)
    return (
        render_identity(), render_table(), render_log(), render_action(),
        "", "", render_reflections(),
    )


async def on_start_random(total, seat):
    return await on_start(total, seat, random.randint(1, 10**6))


def on_submit(text: str):
    if SESS.human is None:
        return "还没开始游戏", ""
    err = SESS.human.submit(text or "")
    return (err or "✅ 已提交，等待判定…", "") + (
        render_log(), render_action(), render_table()
    )


async def _run_reflection() -> None:
    try:
        SESS.reflections = await SESS.game.reflect(
            skip={SESS.human.seat} if SESS.human else set()
        )
    except Exception as e:  # noqa: BLE001 - 复盘失败不该影响结果展示
        print("复盘失败:", e)
        SESS.reflections = {}
    finally:
        SESS.reflecting = False


async def on_tick():
    # 对局一结束就让所有 agent 各自复盘一次（只跑一次）
    if (
        SESS.task is not None
        and SESS.task.done()
        and SESS.reflections is None
        and not SESS.reflecting
        and SESS.game is not None
    ):
        SESS.reflecting = True
        asyncio.create_task(_run_reflection())

    return (
        render_log(), render_action(), render_result(),
        render_table(), render_reflections(),
    )


# --------------------------------------------------------------------------- #
# 界面
# --------------------------------------------------------------------------- #

#: 让多行输入框按回车就提交（Gradio 默认多行框要 Ctrl+Enter）。
#: Shift+Enter 仍然换行；中文输入法选词时的回车不会误触发。
_ENTER_JS = """() => {
  const bind = () => {
    document.querySelectorAll('#cmd-input textarea').forEach((ta) => {
      if (ta.dataset.enterBound === '1') return;
      ta.dataset.enterBound = '1';
      ta.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' || e.shiftKey || e.isComposing || e.keyCode === 229) return;
        e.preventDefault();
        const btn = document.getElementById('cmd-submit');
        if (btn) btn.click();
      });
    });
  };
  bind();
  new MutationObserver(bind).observe(document.body, { childList: true, subtree: true });
}
"""


RULES_HTML = """
<div id="rules-modal" style="display:none;position:fixed;inset:0;z-index:99999;
     background:rgba(20,20,25,.55);align-items:center;justify-content:center;padding:20px;">
  <div style="width:min(780px,100%);max-height:84vh;overflow:auto;background:#fff;
       border-radius:14px;padding:24px 30px;box-shadow:0 18px 60px rgba(0,0,0,.28);
       line-height:1.75;color:#000;font-size:14px;">
    <style>
      /* 弹窗内一律黑字，避免被主题的灰字覆盖 */
      #rules-modal, #rules-modal * { color: #000 !important; }
    </style>
    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
      <h2 style="margin:0;font-size:20px;color:#000;">🗡️ 阿瓦隆规则</h2>
      <button id="rules-close" style="border:none;background:#eef0f3;border-radius:8px;
              padding:6px 14px;cursor:pointer;font-size:13px;">关闭 ✕</button>
    </div>

    <h3 style="font-size:15px;margin:18px 0 6px;color:#000;">阵营与胜负</h3>
    <ul style="margin:0;padding-left:20px;">
      <li><b>蓝方（好人）</b>：完成 3 轮任务，并且终局时梅林没有被刺客指认。</li>
      <li><b>红方（坏人）</b>：让 3 轮任务失败；或者好人 3 胜后由刺客指认梅林命中。</li>
      <li>同一轮连续 <b>5 次</b> 组队被否决，红方直接获胜。</li>
      <li>本作追加：<b>梅林被追责投票淘汰，红方立即获胜</b>。</li>
    </ul>

    <h3 style="font-size:15px;margin:18px 0 6px;color:#000;">每轮流程</h3>
    <ol style="margin:0;padding-left:20px;">
      <li><b>组队</b>：队长挑选本轮队伍，可以选自己。</li>
      <li><b>投票</b>：全员同时投「同意 / 反对」，赞成过半则队伍出发，否则队长顺移。</li>
      <li><b>出任务牌</b>：队员秘密出牌。好人只能出成功，坏人可以出成功伪装，也可以出失败破坏。</li>
      <li><b>公布结果</b>：出现规定数量的失败牌即任务失败（7 人以上第 4 轮需要 2 张）。</li>
      <li><b>移交队长</b>：交给下一位还在场的玩家。</li>
    </ol>

    <h3 style="font-size:15px;margin:18px 0 6px;color:#000;">追责环节（本作追加）</h3>
    <ul style="margin:0;padding-left:20px;">
      <li>任务失败、且不是第 3 次失败时触发。</li>
      <li>从本轮队长开始，按座位顺序每人公开发言一次。</li>
      <li>之后全体在场玩家投票淘汰一人；<b>平票或全弃权则无人出局</b>。</li>
      <li>出局者不能再发言、投票，也不能被选入队伍；出局后<b>公开阵营</b>。</li>
    </ul>

    <h3 style="font-size:15px;margin:18px 0 6px;color:#000;">刺杀</h3>
    <p style="margin:0;">好人拿到 3 胜后进入刺杀阶段，刺客指认一名玩家。命中梅林则红方翻盘，否则好人获胜。</p>

    <h3 style="font-size:15px;margin:18px 0 6px;color:#000;">任务人数</h3>
    <table style="border-collapse:collapse;font-size:13px;">
      <tr style="background:#f5f6f8;">
        <th style="border:1px solid #c6ccd4;padding:4px 12px;">人数</th>
        <th style="border:1px solid #c6ccd4;padding:4px 12px;">第1轮</th>
        <th style="border:1px solid #c6ccd4;padding:4px 12px;">第2轮</th>
        <th style="border:1px solid #c6ccd4;padding:4px 12px;">第3轮</th>
        <th style="border:1px solid #c6ccd4;padding:4px 12px;">第4轮</th>
        <th style="border:1px solid #c6ccd4;padding:4px 12px;">第5轮</th>
      </tr>
      <tr><td style="border:1px solid #c6ccd4;padding:3px 12px;">5</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">2</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">2</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td></tr>
      <tr><td style="border:1px solid #c6ccd4;padding:3px 12px;">6</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">2</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4</td></tr>
      <tr><td style="border:1px solid #c6ccd4;padding:3px 12px;">7</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">2</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4<b>*</b></td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4</td></tr>
      <tr><td style="border:1px solid #c6ccd4;padding:3px 12px;">8</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">5<b>*</b></td><td style="border:1px solid #c6ccd4;padding:3px 12px;">5</td></tr>
      <tr><td style="border:1px solid #c6ccd4;padding:3px 12px;">9</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">5<b>*</b></td><td style="border:1px solid #c6ccd4;padding:3px 12px;">5</td></tr>
      <tr><td style="border:1px solid #c6ccd4;padding:3px 12px;">10</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">3</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">4</td><td style="border:1px solid #c6ccd4;padding:3px 12px;">5<b>*</b></td><td style="border:1px solid #c6ccd4;padding:3px 12px;">5</td></tr>
    </table>
    <p style="margin:6px 0 0;color:#000;font-size:12.5px;">带 * 的轮次需要 <b>2 张失败牌</b> 才算失败。</p>

    <h3 style="font-size:15px;margin:18px 0 6px;color:#000;">你的输入</h3>
    <ul style="margin:0;padding-left:20px;">
      <li>组队：<code>2,4</code>，想加理由用竖线：<code>2,4 | 试试他们</code></li>
      <li>投票：<code>同意</code> / <code>反对</code></li>
      <li>任务牌：<code>成功</code> / <code>失败</code>（好人只能成功）</li>
      <li>追责发言：直接说话</li>
      <li>淘汰票：座位号，或 <code>弃权</code></li>
      <li>回车提交，Shift+回车换行</li>
    </ul>
  </div>
</div>
"""

#: 规则弹窗的开关。用事件委托，避免 Gradio 重渲染后监听器丢失。
_RULES_JS = """
() => {
  if (document.body.dataset.rulesBound === '1') return;
  document.body.dataset.rulesBound = '1';
  const modal = () => document.getElementById('rules-modal');
  document.addEventListener('click', (e) => {
    const m = modal();
    if (!m) return;
    if (e.target.closest('#rules-btn')) { m.style.display = 'flex'; return; }
    if (e.target.closest('#rules-close') || e.target === m) { m.style.display = 'none'; }
  });
  document.addEventListener('keydown', (e) => {
    const m = modal();
    if (m && e.key === 'Escape') m.style.display = 'none';
  });
}
"""


def build_ui(
    default_players: int = 5,
    default_seat: int = 1,
    default_seed: int = 20241117,
) -> gr.Blocks:
    with gr.Blocks(title="阿瓦隆 · AI 对局") as demo:
        gr.Markdown("# 🗡️ 阿瓦隆\n人类 + AI 混桌，5~10 人。")

        with gr.Row():
            players = gr.Dropdown(
                [5, 6, 7, 8, 9, 10], value=default_players, label="人数", scale=1
            )
            seat = gr.Number(value=default_seat, precision=0, label="我的座位", scale=1)
            seed = gr.Number(value=default_seed, precision=0, label="随机种子", scale=1)
            start_btn = gr.Button("开始游戏", variant="primary", scale=1)
            reroll_btn = gr.Button("重抽身份", scale=1)
            rules_btn = gr.Button("📖 规则", scale=1, elem_id="rules-btn")

        gr.HTML(RULES_HTML)

        identity = gr.Markdown(render_identity())

        with gr.Row():
            with gr.Column(scale=3):
                log = gr.Textbox(
                    label="对局日志", lines=26, interactive=False,
                    autoscroll=True,
                )
            with gr.Column(scale=2, min_width=360):
                table = gr.HTML(render_table())
                action = gr.Markdown("尚未开始。")
                inp = gr.Textbox(
                    label="你的输入",
                    placeholder="例如：2,4  /  同意  /  成功  /  4　（回车提交，Shift+回车换行）",
                    lines=2,
                    elem_id="cmd-input",
                )
                submit_btn = gr.Button("提交", variant="primary", elem_id="cmd-submit")
                feedback = gr.Markdown("")

        result = gr.Markdown("")
        reflections = gr.Markdown("")

        timer = gr.Timer(1.0)

        start_btn.click(
            on_start, [players, seat, seed],
            [identity, table, log, action, inp, feedback, reflections],
        )
        reroll_btn.click(
            on_start_random, [players, seat],
            [identity, table, log, action, inp, feedback, reflections],
        )
        submit_btn.click(on_submit, inp, [feedback, inp, log, action, table])
        inp.submit(on_submit, inp, [feedback, inp, log, action, table])
        timer.tick(on_tick, None, [log, action, result, table, reflections])
        demo.load(None, None, None, js=_ENTER_JS)
        demo.load(None, None, None, js=_RULES_JS)

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="阿瓦隆 AI 对局")
    parser.add_argument("--players", type=int, default=5, help="人数 5~10")
    parser.add_argument("--seat", type=int, default=1, help="你的座位号")
    parser.add_argument("--seed", type=int, default=20241117, help="随机种子")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    build_ui(args.players, args.seat, args.seed).launch(
        server_port=args.port,
        inbrowser=not args.no_browser,
        theme=gr.themes.Soft(),
    )

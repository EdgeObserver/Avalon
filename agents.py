"""阿瓦隆 Agent 工厂：一个座位 = 一个 AgentScope Agent。

两个入口：

- build_model()   —— 构造 OpenAI 兼容的 chat model（默认指向 OpenCode Go 的 DeepSeek）
- create_agent()  —— 按角色生成提示词，返回一个可直接 reply() 的 Agent

关于 AgentScope 2.x 的两个坑（v0.x 的老教程在这里全部失效）：

1. model 不再接收 api_key 字符串，而是接收 OpenAICredential 对象；
2. 多人同桌必须用 OpenAIMultiAgentFormatter，否则所有发言会被揉成同一个 user 消息，
   模型分不清「谁说的」，社交推理直接失效。
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from dotenv import load_dotenv

from agentscope.agent import Agent, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from prompts import build_system_prompt

load_dotenv()

# --------------------------------------------------------------------------- #
# 模型配置
# --------------------------------------------------------------------------- #

#: OpenCode Go 的 OpenAI 兼容端点
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
#: 默认模型：快、便宜，够做社交推理
DEFAULT_MODEL = "deepseek-v4.1-flash"
#: 依次尝试这些环境变量来拿 key
API_KEY_ENVS = ("OPENCODE_API_KEY", "OPENCODE_GO_API_KEY")


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    for env in API_KEY_ENVS:
        value = os.getenv(env, "").strip()
        if value:
            return value
    raise RuntimeError(
        "没有找到 OpenCode Go 的 API key。请设置环境变量 "
        f"{' 或 '.join(API_KEY_ENVS)}，或在项目根目录的 .env 里填写。"
    )


def build_model(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    thinking_enable: bool = False,
    reasoning_effort: str | None = None,
    context_size: int = 128_000,
    max_retries: int = 3,
    session_id: str | None = None,
    user_agent: str = "avalon-agent/0.1",
) -> OpenAIChatModel:
    """构造一个 OpenAI 兼容的 chat model。

    Args:
        api_key: 不传则从 OPENCODE_API_KEY / OPENCODE_GO_API_KEY 读取。
        base_url: 不传则用 OpenCode Go 的默认端点。
        model: 模型 id，默认 deepseek-v4.1-flash。
        temperature: 0.8 左右比较适合"撒谎"和"即兴发言"。
        max_tokens: 一次发言的字数上限，别给太大。
        thinking_enable: 默认 False。**这个参数很关键**：开启 thinking 时，
            推理模型前几轮只吐 thinking 块、不吐正文，AgentScope 的 ReAct
            循环会一直认为"还没给出最终答案"，最后撞上 max_iters 返回一句
            "The maximum reasoning-acting iterations are exceeded."。
            桌游发言要短、要快，默认关掉思考。
        reasoning_effort: 可选 low / medium / high，None 表示用服务端默认。
        context_size: 上下文窗口，用于 AgentScope 的上下文压缩判断。
        max_retries: 网络失败重试次数。
        session_id: **OpenCode Go 必需**。服务端要靠它做路由和 prompt 缓存，
            缺了直接 400：``Request is missing x-opencode-session``。
            整个对局共用一个即可；不传会自动生成一个。
        user_agent: 文档要求客户端表明身份，别用 SDK 默认名。
    """
    credential = OpenAICredential(
        api_key=_resolve_api_key(api_key),
        base_url=base_url or os.getenv("OPENCODE_BASE_URL") or DEFAULT_BASE_URL,
    )

    client_kwargs: dict[str, Any] = {}
    if "opencode" in str(credential.base_url):
        client_kwargs["default_headers"] = {
            "x-opencode-session": session_id or f"avalon-{uuid.uuid4().hex[:16]}",
            "User-Agent": user_agent,
        }

    return OpenAIChatModel(
        credential=credential,
        model=model or os.getenv("AVALON_MODEL") or DEFAULT_MODEL,
        parameters=OpenAIChatModel.Parameters(
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enable=thinking_enable,
            reasoning_effort=reasoning_effort,
        ),
        stream=False,  # 桌面游戏不需要逐字流式，整条返回更好解析
        max_retries=max_retries,
        context_size=context_size,
        formatter=OpenAIMultiAgentFormatter(),
        client_kwargs=client_kwargs,
    )


# --------------------------------------------------------------------------- #
# Agent 工厂
# --------------------------------------------------------------------------- #


def agent_name(seat: int) -> str:
    """座位的公开代号。

    刻意只用座位号、不用角色名——Agent 的 name 会出现在每条消息里，
    一旦写成「梅林」，整局游戏的隐藏身份就当场泄露了。
    """
    return f"玩家{seat}"


def create_agent(
    role: str,
    seat: int,
    *,
    total_players: int = 7,
    private_info: str = "",
    private_channel: str | None = None,
    extra_instructions: str = "",
    model: OpenAIChatModel | None = None,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    thinking_enable: bool = False,
    reasoning_effort: str | None = None,
    max_iters: int = 3,
) -> Agent:
    """按角色初始化一个阿瓦隆 Agent。

    Args:
        role: 角色，见 prompts.ALL_ROLES。
        seat: 座位号，从 1 开始。
        total_players: 总人数，决定规则里的任务人数表。
        private_info: 该角色独有的信息，例如
            「你的同伙是玩家3（莫甘娜）、玩家5（刺客）」。由游戏引擎生成。
        private_channel: 私密频道名，例如「红方频道」。仅当该座位是频道成员时
            传入；好人和奥伯伦传 None。
        extra_instructions: 追加指令，通常是本轮的临场任务。
        model: 复用同一个 model 实例可以省掉重复建客户端；不传就新建。
        temperature / max_tokens / thinking_enable / reasoning_effort:
            透传给 build_model。
        max_iters: ReAct 最大轮数。这几个 agent 不用工具，正常情况下第一轮
            就出正文，给 3 足够。

    Returns:
        已注入角色提示词的 Agent 实例。
    """
    system_prompt = build_system_prompt(
        role,
        seat,
        total_players=total_players,
        private_info=private_info,
        private_channel=private_channel,
        extra_instructions=extra_instructions,
    )

    if model is None:
        model = build_model(
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enable=thinking_enable,
            reasoning_effort=reasoning_effort,
        )

    return Agent(
        name=agent_name(seat),
        system_prompt=system_prompt,
        model=model,
        react_config=ReActConfig(max_iters=max_iters),
    )


def extract_text(msg: Msg) -> str:
    """从 reply() 返回的 Msg 中取出纯文本。

    v2 的 content 是内容块列表（text / thinking / tool_call / tool_result），
    引擎只想要玩家说的那句话，所以统一在这里剥掉其它块。
    """
    parts = [
        block.text
        for block in msg.content
        if getattr(block, "type", "") == "text"
    ]
    return "\n".join(p.strip() for p in parts if p.strip())

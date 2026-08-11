# =============================================================================
# 【文件头】llm_client.py —— LLM 直调薄封装（DashScope 为主，DeepSeek 可选）
# 职责：发起生成调用并提供三件小事：
#         1. to_thread 桥接（SDK/requests 是同步的，放进工作线程不阻塞事件循环）
#         2. 对网络/限流类异常做指数退避重试（解析错误不重试）
#         3. 可选 JSON mode（response_format={"type": "json_object"}）
# 服务选择（显式、非通用抽象，见 AGENTS.md 2026-08-11 DeepSeek 例外）：
#   - 模型名以 deepseek- 开头 → DeepSeek OpenAI 兼容 API（api.deepseek.com），
#     key 从 DEEPSEEK_API_KEY 环境变量读取（缺失即报错）
#   - 其余模型 → 阿里云百炼 DashScope（默认主服务，key 由调用方直传）
# 安全：key 只存在于本对象/环境变量中，绝不打印、不落日志、不序列化进任何
#       返回结构。
# 【注意】本文件不 import 禁止的上层编排组件，也不依赖旧 Agent SDK。
#       不引入通用 provider 抽象层。
# =============================================================================

"""Thin async wrapper around the LLM chat APIs.

DashScope is the default and primary service (direct ``Generation.call``, same
SDK family as the lesson-store embedding / rerank). As an explicit, non-generic
extension, model names with the ``deepseek-`` prefix are routed to DeepSeek's
OpenAI-compatible REST API. Responsibilities are deliberately limited:
  1. bridge synchronous calls through ``asyncio.to_thread``,
  2. retry transient failures (network / rate-limit) with exponential backoff,
  3. optional JSON mode via ``response_format``.
"""

import asyncio
import os
import time
from typing import Optional

import dashscope
import requests

# -- Retry policy ----------------------------------------------------------
# 重试只对网络/限流类异常生效（瞬时故障，重试有希望）；解析/业务错误不重试。
# 参数收敛为模块级常量，便于测试与调优。
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0  # 指数退避基数（秒）：1s -> 2s
REQUEST_TIMEOUT = 120   # 单次生成请求超时（秒）

# -- DeepSeek（可选 provider，AGENTS.md 2026-08-11 例外）---------------------
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


def _is_deepseek_model(model: str) -> bool:
    """显式路由：模型名以 deepseek- 前缀开头即走 DeepSeek API。"""
    return bool(model and model.startswith("deepseek-"))


# 命中任一关键词视为瞬时故障，值得重试；大小写不敏感。
_RETRYABLE_TOKENS = (
    "throttl", "rate limit", "limit", "timeout", "connection",
    "eof", "temporary", "unavailable", "retry",
)


def _is_retryable(err: Exception) -> bool:
    """是否值得重试：网络/超时类异常，或错误消息命中瞬时故障关键词。"""
    if isinstance(err, (ConnectionError, TimeoutError, OSError)):
        return True
    msg = str(err).lower()
    return any(tok in msg for tok in _RETRYABLE_TOKENS)


class LLMClient:
    """无状态 DashScope 聊天客户端（一个实例可被并发安全地复用）。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.2,
        enable_thinking: bool = False,
        max_attempts: int = MAX_ATTEMPTS,
        deepseek_api_key: Optional[str] = None,
    ):
        # 【安全】密钥只存在实例上，仅用于每次调用；不打印、不落盘。
        # api_key 是 DashScope 主服务 key；deepseek_api_key 是可选 DeepSeek key
        # （前端双输入框场景），未传时 DeepSeek 分支回退到 api_key / 环境变量。
        self._api_key = api_key
        self._deepseek_api_key = deepseek_api_key
        self.model = model
        self.temperature = temperature
        # 思考模式默认关闭（与旧 Qwen-Agent 配置一致）；仅 QWEN_ENABLE_THINKING=1
        # 时开启。思考输出含 reasoning_content，但直调只取 output.text。
        self.enable_thinking = enable_thinking
        self.max_attempts = max_attempts

    # -- 同步核心（在工作线程里执行） -----------------------------------------

    def sync_call(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """一次带重试的 DashScope 生成调用，返回助手文本（同步）。"""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                return self._call_once(system, user, json_mode)
            except Exception as err:  # noqa: BLE001 - 统一交给 _is_retryable 分类
                last_err = err
                if not _is_retryable(err) or attempt == self.max_attempts - 1:
                    raise
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
        raise last_err  # 理论不可达；防御性抛出

    def _call_once(self, system: str, user: str, json_mode: bool) -> str:
        """单次生成调用；非 200 状态码视为业务失败并抛 RuntimeError。

        模型名以 deepseek- 开头时路由到 DeepSeek OpenAI 兼容 API，否则走
        DashScope（默认主服务）。分支显式且固定，不引入通用 provider 抽象。
        """
        if _is_deepseek_model(self.model):
            return self._call_once_deepseek(system, user, json_mode)
        return self._call_once_dashscope(system, user, json_mode)

    def _call_once_dashscope(self, system: str, user: str, json_mode: bool) -> str:
        """DashScope Generation.call；兼容经典 output.text 与 preview 的 choices 结构。"""
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "api_key": self._api_key,
            "timeout": REQUEST_TIMEOUT,
        }
        if json_mode:
            # 【主流程】协议层约束输出为合法 JSON，配合宽容解析链兜底。
            params["response_format"] = {"type": "json_object"}
        if self.enable_thinking:
            params["enable_thinking"] = True
        resp = dashscope.Generation.call(**params)
        if getattr(resp, "status_code", 500) != 200:
            raise RuntimeError(
                f"DashScope generation failed: {getattr(resp, 'code', '')} "
                f"{getattr(resp, 'message', '')}"
            )
        # 【模型兼容】DashScope 响应结构有两种：
        #   - 经典模型（qwen-plus / qwen3.7-max-2026-05-17 等）：output.text
        #   - preview 系列（如 qwen3.7-max-preview）：OpenAI-compatible 的
        #     output.choices[0].message.content，顶层 output.text 为 None 占位
        # 取文本时两者兼容，取不到即视为失败（由重试/上层兜底处理）。
        out = resp.output
        text = out.get("text") if isinstance(out, dict) else None
        if not text and isinstance(out, dict) and out.get("choices"):
            try:
                text = out["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = None
        if not text:
            raise RuntimeError("DashScope generation returned empty text")
        return text

    def _call_once_deepseek(self, system: str, user: str, json_mode: bool) -> str:
        """DeepSeek OpenAI 兼容 API（requests 直调，零新增依赖）。

        key 来源（AGENTS.md 例外条款，优先级从高到低）：
          1. 前端 DeepSeek key（LLMClient.deepseek_api_key，双输入框场景）
          2. 前端通用 key（LLMClient.api_key，兼容旧前端只填一个框）
          3. DEEPSEEK_API_KEY 环境变量
        三者皆无则报错。DeepSeek 不区分 enable_thinking（deepseek-reasoner
        自带推理），因此不传该参数。
        """
        api_key = self._deepseek_api_key or self._api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key for DeepSeek: pass a DeepSeek key to the optimizer "
                "or set the DEEPSEEK_API_KEY environment variable"
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(
            DEEPSEEK_API_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"DeepSeek generation failed: {resp.status_code} {resp.text[:200]}"
            )
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = None
        if not text:
            raise RuntimeError("DeepSeek generation returned empty text")
        return text

    # -- 异步桥接 -------------------------------------------------------------

    async def ask(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """线程桥接的异步调用：模型思考时事件循环仍能处理轮询/SSE。"""
        return await asyncio.to_thread(self.sync_call, system, user, json_mode=json_mode)

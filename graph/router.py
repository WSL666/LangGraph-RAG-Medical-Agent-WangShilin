"""LangGraph 条件边判定函数。"""

from __future__ import annotations

from .state import GraphState


_MODE_FALLBACK_CHAIN = ["hybrid", "bm25", "embedding"]
_MAX_RETRIEVE_RETRY = 2


def route_after_cache(state: GraphState) -> str:
    """命中缓存走 stream_cache，否则进入意图判定。"""
    return "hit" if state.get("cache_hit") else "miss"


def route_after_intent(state: GraphState) -> str:
    """按意图三路分流。"""
    intent = state.get("intent", "medical")
    if intent == "chitchat":
        return "chitchat"
    if intent == "refuse":
        return "refuse"
    return "medical"


def route_after_retrieve(state: GraphState) -> str:
    """检索为空且未达上限则切换模式重试，否则进入生成。"""
    docs = state.get("docs", [])
    retry = state.get("retry_count", 0)

    if docs:
        return "generate"
    if retry >= _MAX_RETRIEVE_RETRY:
        return "generate"
    return "retry"


def next_retrieval_mode(state: GraphState) -> dict:
    """retry 时切换到下一种检索模式，并把 retry_count + 1。

    注意：本函数被 retry 节点（实际是 retrieve 自身的前置 hop）使用，
    在 build.py 中以 lambda 形式包装为节点。
    """
    current = state.get("retrieval_mode", "hybrid")
    try:
        idx = _MODE_FALLBACK_CHAIN.index(current)
    except ValueError:
        idx = 0
    next_mode = _MODE_FALLBACK_CHAIN[min(idx + 1, len(_MODE_FALLBACK_CHAIN) - 1)]
    return {
        "retrieval_mode": next_mode,
        "retry_count": state.get("retry_count", 0) + 1,
    }

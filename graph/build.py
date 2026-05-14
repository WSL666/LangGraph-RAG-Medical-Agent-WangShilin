"""装配 LangGraph：节点 + 边 + 编译。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    cache_lookup_node,
    chitchat_node,
    generate_node,
    intent_route_node,
    query_rewrite_node,
    retrieve_node,
    save_cache_node,
    stream_cache_node,
)
from .router import (
    next_retrieval_mode,
    route_after_cache,
    route_after_intent,
    route_after_retrieve,
)
from .state import GraphState


def build_graph():
    """构建并编译医疗 RAG 状态图。

    入口：cache_lookup
    出口：END（命中缓存 / 拒答 / 完成生成 + 写缓存）
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("cache_lookup", cache_lookup_node)
    workflow.add_node("stream_cache", stream_cache_node)
    workflow.add_node("intent_route", intent_route_node)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("retrieve_retry", next_retrieval_mode)
    workflow.add_node("generate", generate_node)
    workflow.add_node("save_cache", save_cache_node)

    workflow.add_edge(START, "cache_lookup")

    workflow.add_conditional_edges(
        "cache_lookup",
        route_after_cache,
        {"hit": "stream_cache", "miss": "intent_route"},
    )
    workflow.add_edge("stream_cache", "save_cache")

    workflow.add_conditional_edges(
        "intent_route",
        route_after_intent,
        {"medical": "query_rewrite", "chitchat": "chitchat", "refuse": END},
    )
    workflow.add_edge("chitchat", "save_cache")

    workflow.add_edge("query_rewrite", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"retry": "retrieve_retry", "generate": "generate"},
    )
    workflow.add_edge("retrieve_retry", "retrieve")

    workflow.add_edge("generate", "save_cache")
    workflow.add_edge("save_cache", END)

    return workflow.compile()

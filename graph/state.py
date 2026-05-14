"""图共享状态定义。"""

from typing import Any, Literal, TypedDict


Intent = Literal["medical", "chitchat", "refuse"]
RetrievalMode = Literal["hybrid", "bm25", "embedding"]


class GraphState(TypedDict, total=False):
    """LangGraph 节点之间共享的状态字段。"""

    query: str
    rewritten_query: str
    intent: Intent
    docs: list[dict[str, Any]]
    retrieval_text: str
    retrieval_mode: RetrievalMode
    retry_count: int
    answer: str
    cache_hit: bool
    chat_history: list[dict[str, str]]

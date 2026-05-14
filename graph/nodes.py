"""LangGraph 节点实现。

每个节点接收当前 GraphState，返回需要更新的字段（部分字典）。
所有"工具"调用（ES、Redis、LLM）通过本模块顶部的全局单例注入，
便于单元测试时 mock。
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from ES_search import main_ES
from redis_search import REDIS

from .state import GraphState


_REDIS: REDIS | None = None
_LLM_FAST: ChatOpenAI | None = None
_LLM_CHAT: ChatOpenAI | None = None


def _redis() -> REDIS:
    global _REDIS
    if _REDIS is None:
        _REDIS = REDIS()
    return _REDIS


def _make_llm(streaming: bool, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        temperature=temperature,
        streaming=streaming,
    )


def _llm_fast() -> ChatOpenAI:
    """用于意图识别 / 改写：低温度、非流式。"""
    global _LLM_FAST
    if _LLM_FAST is None:
        _LLM_FAST = _make_llm(streaming=False, temperature=0.0)
    return _LLM_FAST


def _llm_chat() -> ChatOpenAI:
    """用于最终生成 / 闲聊：可流式。"""
    global _LLM_CHAT
    if _LLM_CHAT is None:
        _LLM_CHAT = _make_llm(streaming=True, temperature=0.7)
    return _LLM_CHAT


# -------------------------- 节点实现 --------------------------


def cache_lookup_node(state: GraphState) -> dict[str, Any]:
    """查询 Redis 缓存，命中则把答案写入 state。"""
    query = state["query"]
    cached = _redis().get_answer(query)
    if cached:
        return {"cache_hit": True, "answer": cached}
    return {"cache_hit": False}


def stream_cache_node(state: GraphState) -> dict[str, Any]:
    """命中缓存时模拟流式输出（按字符切块），让前端体验一致。"""
    answer: str = state.get("answer", "")
    writer = get_stream_writer()
    chunk_size = 8
    for i in range(0, len(answer), chunk_size):
        writer({"chunk": answer[i : i + chunk_size]})
    return {"answer": answer}


_SENSITIVE_PATTERN = re.compile(
    r"(自杀|自残|爆炸|毒品|武器|色情|赌博)", re.IGNORECASE
)

def intent_route_node(state: GraphState) -> dict[str, Any]:
    """用LLM 判定 闲聊/医疗/拒答。"""
    query = state["query"]
    if _SENSITIVE_PATTERN.search(query):
        return {"intent": "refuse", "answer": "抱歉，该问题涉及敏感内容，无法回答。建议您寻求专业人士帮助。"}

    prompt = (
    "你是专业的对话路由分类器,执行严格的单标签分类,仅输出一个英文单词:medical / chitchat / refuse\n\n"
    "分类定义：\n"
    "- medical:用户明确咨询健康、疾病、症状、用药、处方、检查报告、诊疗方案、就医指导等医疗相关诉求\n"
    "- chitchat:无任何医疗诉求，仅为日常问候、闲聊、情绪表达、社交对话\n"
    "- refuse:包含任何危害自身/他人、违规违法、色情暴力、敏感危险内容，例如：自杀|自残|爆炸|毒品|武器|色情|赌博\n"
    f"输入内容：{query}\n"
    "分类结果："
        )
    try:
        resp = _llm_fast().invoke([HumanMessage(content=prompt)])
        text = (resp.content or "").strip().lower()
    except Exception:
        text = "medical"

    if "chitchat" in text:
        return {"intent": "chitchat"}
    return {"intent": "medical"}


def chitchat_node(state: GraphState) -> dict[str, Any]:
    """闲聊分支：不走 RAG，直接对话回应。"""
    query = state["query"]
    history = state.get("chat_history", [])[-4:]
    messages = [SystemMessage(content="你是友好的医疗助手，请用简洁中文回应用户的闲聊或问候，不要提及内部实现。")]
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(SystemMessage(content=f"(上一轮助手回复) {content}"))
    messages.append(HumanMessage(content=query))

    full = ""
    for chunk in _llm_chat().stream(messages):
        text = getattr(chunk, "content", "") or ""
        if text:
            full += text
    return {"answer": full}


def query_rewrite_node(state: GraphState) -> dict[str, Any]:
    """把口语化问题改写成更适合 ES 检索的关键词式查询。"""
    query = state["query"]

    prompt = (
    "你是医疗检索问句改写专家，请对用户原始医疗问题做标准化精简改写，用于知识库检索。\n"
    "严格遵循以下规则：\n"
    "1. 精简为单行简短中文检索问句，不拆分、不换行、不输出多余内容；\n"
    "2. 必须完整保留核心实体：疾病名称、身体部位、症状、药品、检查项目、诊疗方式、病因、禁忌人群；\n"
    "3. 删除口语助词、冗余修饰、情绪感叹、无关客套话、重复表述；\n"
    "4. 统一口语表述为专业通俗说法，不新增原意没有的信息、不扩写、不编造；\n"
    "5. 只输出改写后的检索短句，不要解释、不要分析、不要额外说明。\n\n"
    f"原始医疗问题：{query}\n"
    "标准化检索改写："
      )
    try:
        resp = _llm_fast().invoke([HumanMessage(content=prompt)])
        text = (resp.content or "").strip().splitlines()[0].strip()
        return {"rewritten_query": text or query}
    except Exception:
        return {"rewritten_query": query}


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """调用 ES 检索；按当前 retrieval_mode 执行。"""
    q = state.get("rewritten_query") or state["query"]
    mode = state.get("retrieval_mode", "hybrid")
    top_k = int(os.getenv("RAG_TOP_K", "3"))
    try:
        docs = main_ES(q, mode=mode) or []
        if not isinstance(docs, list):
            docs = []
        docs = docs[:top_k]
    except Exception:
        docs = []
    return {"docs": docs, "retrieval_text": _format_docs(docs)}


_USEFUL_FIELDS = [
    "name",
    "desc",
    "prevent",
    "cause",
    "symptom",
    "acompany",
    "cure_department",
    "cure_way",
    "common_drug",
    "check",
    "recommand_drug",
]


def _format_docs(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "未检索到可用医学资料。"
    chunks = []
    for i, doc in enumerate(docs, 1):
        lines = [f"[资料{i}]"]
        for field in _USEFUL_FIELDS:
            v = doc.get(field)
            if v:
                lines.append(f"{field}: {v}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def generate_node(state: GraphState) -> dict[str, Any]:
    """基于检索资料生成最终回答（流式由外层捕获）。"""
    query = state["query"]
    retrieval_text = state.get("retrieval_text", "未检索到可用医学资料。")
    history = state.get("chat_history", [])[-4:]

    sys_prompt = (
        "你是专业医疗问答助手，请基于给定资料给出清晰、稳妥的中文回答。\n"
        "要求：\n"
        "1) 回答要简洁、可执行；\n"
        "2) 若资料不足，明确说明并给出就医建议；\n"
        "3) 不要提及\"根据上下文/根据检索结果\"等字样；\n"
        "4) 不输出任何数据库或系统实现细节。"
    )
    user_prompt = (
        f"用户问题：{query}\n\n"
        f"参考资料：\n{retrieval_text}"
    )

    messages: list[Any] = [SystemMessage(content=sys_prompt)]
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(SystemMessage(content=f"(上一轮助手回复) {content}"))
    messages.append(HumanMessage(content=user_prompt))

    full = ""
    try:
        for chunk in _llm_chat().stream(messages):
            text = getattr(chunk, "content", "") or ""
            if text:
                full += text
    except Exception as e:
        full = f"模型调用失败，请检查 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL 配置。错误: {e}"
    return {"answer": full}


def save_cache_node(state: GraphState) -> dict[str, Any]:
    """非空答案写回 Redis 供下次命中。"""
    answer = state.get("answer", "")
    if state.get("intent") == "refuse":
        return {}
    if answer and not state.get("cache_hit"):
        try:
            _redis().save_qa(state["query"], answer)
        except Exception:
            pass
    return {}

"""医疗 RAG 聊天系统：基于 LangGraph 的状态图编排。

对外仅暴露 ``MedicalChatSystem.process_query``，与 Gradio 端的接口完全兼容。
内部把"缓存查 → 意图分流 → 改写 → 检索（含回退）→ 生成 → 写缓存"组织成
有向状态图，由 ``graph.build_graph()`` 编译生成。
"""

from __future__ import annotations

import os
from typing import Any

from graph import build_graph


def _load_env_from_file() -> None:
    """从当前目录的 .env 加载环境变量（不覆盖已存在值）。"""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f".env 加载失败，将继续使用系统环境变量: {e}")


_load_env_from_file()


class MedicalChatSystem:
    """LangGraph 版医疗 RAG 系统。

    使用方式与原版一致：

        sys = MedicalChatSystem()
        for partial in sys.process_query(query, chat_history):
            ...  # partial 为更新后的 chat_history（含流式 token 拼接）
    """

    def __init__(self, max_history_size: int = 10):
        self.max_history_size = max_history_size
        self.cache_history: list[dict[str, str]] = []
        self.graph = build_graph()

    # ---------------- 历史管理 ----------------

    def _trim_history(self) -> None:
        if len(self.cache_history) > self.max_history_size:
            keep_from = len(self.cache_history) - (self.max_history_size // 2)
            self.cache_history = self.cache_history[keep_from:]

    # ---------------- 主入口 ----------------

    def process_query(self, query: str, chat_history: list[dict[str, str]]):
        """执行一次问答；以生成器形式产出累计的 chat_history（供 Gradio 实时刷新）。"""
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": ""})
        self.cache_history.append({"role": "user", "content": query})

        initial_state: dict[str, Any] = {
            "query": query,
            "chat_history": list(self.cache_history),
            "retrieval_mode": "hybrid",
            "retry_count": 0,
            "cache_hit": False,
        }

        full_response = ""
        final_state: dict[str, Any] = {}

        try:
            for mode, payload in self.graph.stream(
                initial_state,
                stream_mode=["messages", "custom", "values"],
            ):
                if mode == "messages":
                    chunk, meta = payload
                    node = meta.get("langgraph_node", "")
                    if node not in {"generate", "chitchat"}:
                        continue
                    text = getattr(chunk, "content", "") or ""
                    if text:
                        full_response += text
                        chat_history[-1] = {"role": "assistant", "content": full_response}
                        yield chat_history

                elif mode == "custom":
                    text = (payload or {}).get("chunk", "")
                    if text:
                        full_response += text
                        chat_history[-1] = {"role": "assistant", "content": full_response}
                        yield chat_history

                elif mode == "values":
                    final_state = payload

        except Exception as e:
            err_text = (
                "模型调用失败，请检查 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL 配置。"
                f"错误: {e}"
            )
            chat_history[-1] = {"role": "assistant", "content": err_text}
            yield chat_history
            return

        final_answer = full_response or final_state.get("answer", "")
        if not final_answer:
            final_answer = "暂未生成回答，请稍后再试。"
        chat_history[-1] = {"role": "assistant", "content": final_answer}
        yield chat_history

        self.cache_history.append({"role": "assistant", "content": final_answer})
        self._trim_history()


if __name__ == "__main__":
    MedicalChatSystem()

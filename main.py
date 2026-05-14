"""项目入口：启动基于 LangGraph + RAG 的医疗智能体 Gradio Web 界面。

直接运行：
    python main.py
等价于：
    python gradio_show.py
"""

from __future__ import annotations


def main() -> None:
    # 延迟导入，避免依赖未装时 import 失败导致 CLI 不可用
    from gradio_show import demo
    import os

    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7939"))
    demo.launch(server_name=server_name, server_port=server_port, share=False)


if __name__ == "__main__":
    main()

import os

import gradio as gr

from gradio_background import MedicalChatSystem

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="医疗 RAG 问答",
    css="""
    footer {visibility: hidden}
    .gradio-container {min-height: 100vh !important;}
    """,
) as demo:
    gr.Markdown("# 基于LangGraph与RAG的医疗智能体问答系统")

    chatbot = gr.Chatbot(type="messages", label="对话")
    msg = gr.Textbox(
        label="输入问题",
        placeholder="例如：感冒有什么症状？高血压要注意什么？",
    )
    clear = gr.ClearButton([msg, chatbot], value="清空")
    gr.Markdown("### 示例")
    gr.Examples(
        examples=["感冒", "高血压", "糖尿病", "头痛怎么缓解"],
        inputs=msg,
    )

    chat_system = MedicalChatSystem(max_history_size=10)

    def submit_query(message, history):
        if not message.strip():
            return "", history
        for updated_history in chat_system.process_query(message, history):
            yield "", updated_history

    msg.submit(submit_query, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: (None, []), None, [msg, chatbot], queue=False)


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7939"))
    demo.launch(server_name=server_name, server_port=server_port, share=False)

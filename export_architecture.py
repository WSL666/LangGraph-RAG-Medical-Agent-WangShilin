"""导出 LangGraph 架构图为 PNG / Mermaid 源码。

运行方式：
    python export_architecture.py

成功后会在 docs/ 目录生成：
    - docs/architecture.png   （图片，可直接插入论文 / PPT）
    - docs/architecture.mmd   （mermaid 源码，便于在线再画）
"""

from __future__ import annotations

import os
import sys

from graph import build_graph


def main() -> None:
    g = build_graph()
    drawable = g.get_graph()

    mermaid_text = drawable.draw_mermaid()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(out_dir, exist_ok=True)
    mmd_path = os.path.join(out_dir, "architecture.mmd")
    png_path = os.path.join(out_dir, "architecture.png")

    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid_text)
    print(f"[OK] Mermaid 源码已写入：{mmd_path}")

    try:
        png_bytes = drawable.draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"[OK] PNG 已写入：{png_path}")
    except Exception as e:
        print(f"[WARN] PNG 生成失败（通常是无法访问 mermaid.ink）: {e}")
        print("      你可以把 architecture.mmd 内容贴到 https://mermaid.live 在线渲染并下载。")
        sys.exit(0)


if __name__ == "__main__":
    main()

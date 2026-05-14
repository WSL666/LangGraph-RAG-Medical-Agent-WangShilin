"""Redis 缓存初始化 / 联通性测试脚本。

主项目运行时使用根目录的 ``redis_search.py``；本脚本仅用于：
    1. 验证 Redis 是否能正常连接；
    2. 预热若干常见问答，便于演示缓存命中链路。
"""

import os

import redis


class REDIS:
    def __init__(self) -> None:
        host = os.getenv("REDIS_HOST", "127.0.0.1")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD", "") or None
        db = int(os.getenv("REDIS_DB", "0"))
        self.redis_node = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True,
        )

    def get_answer(self, question: str) -> str | None:
        return self.redis_node.get(f"qa:{question}")

    def save_qa(self, question: str, answer: str, ttl_seconds: int = 360) -> None:
        self.redis_node.setex(f"qa:{question}", ttl_seconds, answer)
        print(f"已写入 Redis: {question}")


if __name__ == "__main__":
    client = REDIS()
    seed_qa = {
        "什么是人工智能？": "人工智能是研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统的一门技术科学。",
        "每天建议喝多少水？": "成年人每天建议饮用 1500–2000 毫升水，具体量可根据活动量、环境温度调整。",
        "Python 是什么类型的编程语言？": "Python 是一种面向对象、解释型的高级编程语言，以简洁易读的语法著称。",
        "地球自转一周需要多长时间？": "地球自转一周约为 24 小时（精确值约 23 小时 56 分 4 秒）。",
        "如何缓解轻度失眠？": "保持规律作息、睡前避免使用电子设备、营造安静黑暗的睡眠环境，可帮助缓解轻度失眠。",
    }
    for q, a in seed_qa.items():
        client.save_qa(q, a)

    while True:
        user_input = input("请输入问题（Ctrl+C 退出）：").strip()
        if not user_input:
            continue
        result = client.get_answer(user_input)
        if result:
            print(f"查询结果：{result}")
        else:
            print(f"未找到 '{user_input}' 的答案")

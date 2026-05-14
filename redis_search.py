import redis
import os

class REDIS:
    def __init__(self):
        redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD", "")
        redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_node = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password if redis_password else None,
            db=redis_db,
            decode_responses=True,
        )
        self.enabled = True
        try:
            self.redis_node.ping()
        except Exception as e:
            self.enabled = False
            print(f"Redis不可用，已降级为无缓存模式: {e}")

    # 查询
    def get_answer(self,question):
        if not self.enabled:
            return None
        try:
            val = self.redis_node.get(f"qa:{question}")
            print("连接成功 redis数据库")
            return val
        except Exception as e:
            self.enabled = False
            print(f"Redis查询失败，已降级为无缓存模式: {e}")
            return None

    # 保存
    def save_qa(self,question,answer):
        if not self.enabled:
            return
        try:
            self.redis_node.setex(f"qa:{question}", 360, answer)
            print("保存成功 redis数据库")
        except Exception as e:
            self.enabled = False
            print(f"Redis写入失败，已降级为无缓存模式: {e}")


if __name__=="__main__":
    REDIS()


# from redis_search import REDIS
#
# query = "头痛怎么缓解？"
# print(REDIS().get_answer(query))


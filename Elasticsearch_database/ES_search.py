#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time
from datetime import datetime

from elasticsearch import Elasticsearch
from openai import OpenAI
from tqdm import tqdm


TEXT_FIELDS = [
    "content",
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
INDEX_NAME = os.getenv("ES_INDEX_NAME", "medical_articles")
EMBEDDING_FIELD = os.getenv("ES_EMBEDDING_FIELD", "embedding")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))
EMBEDDING_TEXT_MAX_CHARS = int(os.getenv("EMBEDDING_TEXT_MAX_CHARS", "3000"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "6"))
EMBEDDING_RATE_LIMIT_SLEEP_SEC = float(os.getenv("EMBEDDING_RATE_LIMIT_SLEEP_SEC", "5"))
EMBEDDING_MIN_INTERVAL_SEC = float(os.getenv("EMBEDDING_MIN_INTERVAL_SEC", "0.3"))


def _load_env_from_file() -> None:
    """从 Ultimate_Edition/.env 加载配置，方便直接运行本脚本。"""
    env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]
    for env_path in env_paths:
        if not os.path.exists(env_path):
            continue
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


def connect_to_elasticsearch(host="localhost", port=9200):
    """连接到Elasticsearch服务器。"""
    es_host = os.getenv("ES_HOST", host)
    es_port = int(os.getenv("ES_PORT", str(port)))
    es_user = os.getenv("ES_USER", "elastic")
    es_password = os.getenv("ES_PASSWORD", "")

    es = Elasticsearch(
        f"http://{es_host}:{es_port}",
        basic_auth=(es_user, es_password),
        verify_certs=False,
        ssl_show_warn=False,
    )

    if es.ping():
        print("成功连接到Elasticsearch")
        return es

    print("无法连接到Elasticsearch")
    return None


def create_index(es, index_name):
    """创建支持 BM25 + dense_vector 的医学文档索引。"""
    try:
        mapping = {
            "settings": {},
            "mappings": {
                "properties": {
                    "name": {"type": "text"},
                    "desc": {"type": "text"},
                    "category": {"type": "keyword"},
                    "prevent": {"type": "text"},
                    "cause": {"type": "text"},
                    "symptom": {"type": "text"},
                    "yibao_status": {"type": "keyword"},
                    "get_prob": {"type": "text"},
                    "easy_get": {"type": "text"},
                    "get_way": {"type": "text"},
                    "acompany": {"type": "text"},
                    "cure_department": {"type": "keyword"},
                    "cure_way": {"type": "keyword"},
                    "cure_lasttime": {"type": "text"},
                    "cured_prob": {"type": "text"},
                    "common_drug": {"type": "keyword"},
                    "cost_money": {"type": "text"},
                    "check": {"type": "keyword"},
                    "do_eat": {"type": "keyword"},
                    "not_eat": {"type": "keyword"},
                    "recommand_eat": {"type": "keyword"},
                    "recommand_drug": {"type": "keyword"},
                    "drug_detail": {"type": "keyword"},
                    "published_date": {"type": "date"},
                    EMBEDDING_FIELD: {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        }

        if es.indices.exists(index=index_name):
            delete_index(es, index_name)

        es.indices.create(index=index_name, body=mapping)
        print(f"索引 '{index_name}' 创建成功")
    except Exception as e:
        print(f"创建索引时出错: {e}")


def insert_document(es, index_name, doc_id, document):
    """插入文档到索引中。"""
    try:
        if "_id" in document:
            del document["_id"]

        result = es.index(index=index_name, id=doc_id, document=document)
        return result
    except Exception as e:
        print(f"插入文档时出错: {e}")
        return None


def delete_index(es, index_name):
    """删除索引。"""
    try:
        if es.indices.exists(index=index_name):
            result = es.indices.delete(index=index_name)
            print(f"索引 '{index_name}' 删除成功")
            return result

        print(f"索引 '{index_name}' 不存在")
        return None
    except Exception as e:
        print(f"删除索引时出错: {e}")
        return None


def load_medical_data(json_file_path):
    """从JSON文件加载医疗数据。"""
    data = []
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    print(f"成功加载 {len(data)} 条医疗数据")
                    return data

                print("成功加载 1 条医疗数据")
                return [data]
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"跳过无效的JSON行: {line}")
                print(f"成功加载 {len(data)} 条医疗数据")
                return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return []


def _embedding_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
        api_key=api_key,
    )


def _stringify_value(value):
    if isinstance(value, list):
        return "、".join(str(item) for item in value if item)
    return str(value) if value else ""


def build_embedding_text(document):
    """将医学字段拼成用于向量化的语义文本。"""
    chunks = []
    for field in TEXT_FIELDS:
        value = _stringify_value(document.get(field))
        if value:
            chunks.append(f"{field}: {value}")
    text = "\n".join(chunks).strip()

    # 兼容非标准结构，尽量避免出现空 input 导致 400 invalid parameter
    if not text:
        fallback_parts = []
        for key, value in document.items():
            if key in {EMBEDDING_FIELD, "published_date"}:
                continue
            val = _stringify_value(value).strip()
            if val:
                fallback_parts.append(f"{key}: {val}")
        text = "\n".join(fallback_parts).strip()

    return text[:EMBEDDING_TEXT_MAX_CHARS]


def _create_embedding_with_retry(client, text: str) -> list[float] | None:
    """单条重试，避免一条超长或异常文档拖垮整批。"""
    if not text or not text.strip():
        return None
    retry_text = text
    for attempt in range(EMBEDDING_MAX_RETRIES):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=retry_text)
            return response.data[0].embedding
        except Exception as e:
            if _is_rate_limit_error(e):
                sleep_s = EMBEDDING_RATE_LIMIT_SLEEP_SEC * (2 ** min(attempt, 4))
                print(f"单条触发限流，{sleep_s:.1f}s 后重试...")
                time.sleep(sleep_s)
                continue
            retry_text = retry_text[: max(500, len(retry_text) // 2)]
            last_error = e
    print(f"单条向量生成失败，已跳过该文档: {last_error}")
    return None


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "tpm limit reached" in message


def create_embeddings(texts):
    """批量生成文档向量。"""
    client = _embedding_client()
    if client is None:
        print("未配置 OPENAI_API_KEY，将只导入BM25索引，不写入向量字段")
        return [None for _ in texts]

    embeddings = [None for _ in texts]
    for start in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="正在生成向量..."):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        valid_items = [
            (start + offset, text)
            for offset, text in enumerate(batch)
            if text and text.strip()
        ]
        if not valid_items:
            continue

        valid_indices = [item[0] for item in valid_items]
        valid_texts = [item[1] for item in valid_items]
        success = False
        for attempt in range(EMBEDDING_MAX_RETRIES):
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=valid_texts)
                for global_idx, emb_item in zip(valid_indices, response.data):
                    embeddings[global_idx] = emb_item.embedding
                success = True
                break
            except Exception as e:
                if _is_rate_limit_error(e):
                    sleep_s = EMBEDDING_RATE_LIMIT_SLEEP_SEC * (2 ** min(attempt, 4))
                    print(f"批量触发限流，{sleep_s:.1f}s 后重试...")
                    time.sleep(sleep_s)
                    continue
                print(f"批量生成向量失败，将改为逐条重试: {e}")
                break

        if not success:
            for global_idx, text in valid_items:
                embeddings[global_idx] = _create_embedding_with_retry(client, text)

        # 轻度节流，减少连续请求触发 TPM
        if EMBEDDING_MIN_INTERVAL_SEC > 0:
            time.sleep(EMBEDDING_MIN_INTERVAL_SEC)
    return embeddings


def insert_medical_data(es, index_name, medical_data):
    """批量插入医疗数据，同时写入 embedding 供稠密检索使用。"""
    embedding_texts = [build_embedding_text(doc) for doc in medical_data]
    embeddings = create_embeddings(embedding_texts)

    for i, doc in tqdm(enumerate(medical_data), desc="正在插入医疗数据..."):
        doc["published_date"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if "_id" in doc:
            del doc["_id"]
        if embeddings[i]:
            doc[EMBEDDING_FIELD] = embeddings[i]
        insert_document(es, index_name, i + 1, doc)

    es.indices.refresh(index=index_name)
    print("所有医疗数据已成功插入并刷新索引")


def main():
    _load_env_from_file()
    global INDEX_NAME, EMBEDDING_FIELD, EMBEDDING_MODEL, EMBEDDING_DIM
    global EMBEDDING_BATCH_SIZE, EMBEDDING_TEXT_MAX_CHARS
    global EMBEDDING_MAX_RETRIES, EMBEDDING_RATE_LIMIT_SLEEP_SEC, EMBEDDING_MIN_INTERVAL_SEC
    INDEX_NAME = os.getenv("ES_INDEX_NAME", "medical_articles")
    EMBEDDING_FIELD = os.getenv("ES_EMBEDDING_FIELD", "embedding")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))
    EMBEDDING_TEXT_MAX_CHARS = int(os.getenv("EMBEDDING_TEXT_MAX_CHARS", "3000"))
    EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "6"))
    EMBEDDING_RATE_LIMIT_SLEEP_SEC = float(os.getenv("EMBEDDING_RATE_LIMIT_SLEEP_SEC", "5"))
    EMBEDDING_MIN_INTERVAL_SEC = float(os.getenv("EMBEDDING_MIN_INTERVAL_SEC", "0.3"))

    es = connect_to_elasticsearch()
    if not es:
        return

    index_name = INDEX_NAME
    create_index(es, index_name)

    json_file_path = os.path.join(os.path.dirname(__file__), "medical.json")
    medical_data = load_medical_data(json_file_path)

    if not medical_data:
        print("未加载到任何医疗数据")
        return

    insert_medical_data(es, index_name, medical_data)
    print("数据插入完成!")


if __name__ == "__main__":
    main()

import os
from typing import Any

from elasticsearch import Elasticsearch
from openai import OpenAI


TEXT_FIELDS = [
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


def _index_name() -> str:
    return os.getenv("ES_INDEX_NAME", "medical_articles")


def _embedding_field() -> str:
    return os.getenv("ES_EMBEDDING_FIELD", "embedding")


def _embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def _retrieval_mode() -> str:
    return os.getenv("RAG_RETRIEVAL_MODE", "hybrid").lower()


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


def _embedding_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
        api_key=api_key,
    )


def get_query_embedding(query_string: str) -> list[float] | None:
    """调用 embedding 模型，将用户问题转成向量。"""
    client = _embedding_client()
    if client is None:
        return None

    try:
        response = client.embeddings.create(model=_embedding_model(), input=query_string)
        return response.data[0].embedding
    except Exception as e:
        print(f"生成查询向量失败，将只使用BM25检索: {e}")
        return None


def _has_embedding_field(es, index_name: str) -> bool:
    """检查索引里是否已经有向量字段，避免旧索引查询报错。"""
    try:
        mapping = es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]["mappings"].get("properties", {})
        return _embedding_field() in properties
    except Exception as e:
        print(f"检查向量字段失败，将只使用BM25检索: {e}")
        return False


def search_bm25_documents(es, index_name, query_string, size=3):
    """BM25/关键词检索。"""
    try:
        query = {
            "multi_match": {
                "query": query_string,
                "fields": TEXT_FIELDS,
                "type": "best_fields",
                "lenient": True,
            }
        }

        # 先禁用 highlight，避免部分字段类型不兼容导致 search_phase_execution_exception
        result = es.search(index=index_name, query=query, size=size)
        return result
    except Exception as e:
        details = getattr(e, "body", None) or getattr(e, "info", None)
        print(f"BM25检索时出错: {e}; details={details}")
        return None


def search_embedding_documents(es, index_name, query_vector, size=3):
    """稠密向量检索，按余弦相似度召回。"""
    embedding_field = _embedding_field()
    try:
        query = {
            "script_score": {
                "query": {"exists": {"field": embedding_field}},
                "script": {
                    "source": (
                        f"cosineSimilarity(params.query_vector, '{embedding_field}') + 1.0"
                    ),
                    "params": {"query_vector": query_vector},
                },
            }
        }
        return es.search(index=index_name, query=query, size=size)
    except Exception as e:
        details = getattr(e, "body", None) or getattr(e, "info", None)
        print(f"向量检索时出错: {e}; details={details}")
        return None


def _hits(results: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not results or "hits" not in results:
        return []
    return results["hits"].get("hits", [])


def _tag_hits(hits: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    for hit in hits:
        hit["_retrieval"] = [source_name]
    return hits


def _merge_hits_by_rrf(
    bm25_hits: list[dict[str, Any]],
    embedding_hits: list[dict[str, Any]],
    top_k: int,
    rrf_k: int | None = None,
    w_bm25: float | None = None,
    w_emb: float | None = None,
) -> list[dict[str, Any]]:
    """用 RRF 融合两路排名，避免不同打分体系直接相加。

    rrf_k / 两路权重均可由环境变量调整，便于做消融实验：
    - RAG_RRF_K       : RRF 平滑常数 k，默认 60
    - RAG_RRF_W_BM25  : BM25 路权重，默认 1.0
    - RAG_RRF_W_EMB   : Embedding 路权重，默认 1.0
    """
    if rrf_k is None:
        rrf_k = int(os.getenv("RAG_RRF_K", "60"))
    if w_bm25 is None:
        w_bm25 = float(os.getenv("RAG_RRF_W_BM25", "1.0"))
    if w_emb is None:
        w_emb = float(os.getenv("RAG_RRF_W_EMB", "1.0"))

    merged: dict[str, dict[str, Any]] = {}

    for source_name, hits, weight in (
        ("bm25", bm25_hits, w_bm25),
        ("embedding", embedding_hits, w_emb),
    ):
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit.get("_id") or str(hit.get("_source", {}))
            if doc_id not in merged:
                merged[doc_id] = {
                    "_source": hit.get("_source", {}),
                    "_score": 0.0,
                    "_retrieval": [],
                }
            merged[doc_id]["_score"] += weight / (rrf_k + rank)
            merged[doc_id]["_retrieval"].append(source_name)

    sorted_hits = sorted(merged.values(), key=lambda item: item["_score"], reverse=True)
    return sorted_hits[:top_k]


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _merge_hits_by_weighted_sum(
    bm25_hits: list[dict[str, Any]],
    embedding_hits: list[dict[str, Any]],
    top_k: int,
    alpha: float,
) -> list[dict[str, Any]]:
    """min-max 归一化两路打分后加权求和，作为 RRF 的对照融合方式。"""
    bm25_norm = _min_max_normalize([float(hit.get("_score", 0.0)) for hit in bm25_hits])
    emb_norm = _min_max_normalize(
        [float(hit.get("_score", 0.0)) for hit in embedding_hits]
    )

    merged: dict[str, dict[str, Any]] = {}
    for hit, score in zip(bm25_hits, bm25_norm):
        doc_id = hit.get("_id") or str(hit.get("_source", {}))
        merged[doc_id] = {
            "_source": hit.get("_source", {}),
            "_score": alpha * score,
            "_retrieval": ["bm25"],
        }

    for hit, score in zip(embedding_hits, emb_norm):
        doc_id = hit.get("_id") or str(hit.get("_source", {}))
        if doc_id in merged:
            merged[doc_id]["_score"] += (1 - alpha) * score
            merged[doc_id]["_retrieval"].append("embedding")
        else:
            merged[doc_id] = {
                "_source": hit.get("_source", {}),
                "_score": (1 - alpha) * score,
                "_retrieval": ["embedding"],
            }

    sorted_hits = sorted(merged.values(), key=lambda item: item["_score"], reverse=True)
    return sorted_hits[:top_k]


def _sources_from_hits(
    hits: list[dict[str, Any]], include_metadata: bool = False
) -> list[dict[str, Any]]:
    docs = []
    for hit in hits:
        source = dict(hit.get("_source", {}))
        if include_metadata:
            source["_retrieval_id"] = hit.get("_id")
            source["_retrieval_score"] = hit.get("_score")
            source["_retrieval_sources"] = hit.get("_retrieval", [])
        docs.append(source)
    return docs


def retrieve_documents(
    query: str,
    mode: str | None = None,
    top_k: int | None = None,
    candidate_size: int | None = None,
    include_metadata: bool = False,
    es=None,
) -> list[dict[str, Any]]:
    """按指定模式检索文档：bm25、embedding、hybrid 或 weighted。"""
    mode = (mode or _retrieval_mode()).lower()
    if mode not in {"bm25", "embedding", "hybrid", "weighted"}:
        raise ValueError("mode 必须是 bm25、embedding、hybrid 或 weighted")

    if es is None:
        es = connect_to_elasticsearch()
    if es is None:
        return []

    index_name = _index_name()
    size = top_k or int(os.getenv("RAG_TOP_K", "3"))
    candidate_size = candidate_size or int(
        os.getenv("RAG_CANDIDATE_SIZE", str(size * 3))
    )
    try:
        if not es.indices.exists(index=index_name):
            print(f"ES索引不存在: {index_name}")
            return []
    except Exception as e:
        print(f"检查索引存在性失败: {e}")
        return []

    bm25_hits = []
    if mode in {"bm25", "hybrid", "weighted"}:
        bm25_results = search_bm25_documents(es, index_name, query, size=candidate_size)
        bm25_hits = _tag_hits(_hits(bm25_results), "bm25")

    embedding_hits = []
    if mode in {"embedding", "hybrid", "weighted"} and _has_embedding_field(es, index_name):
        query_vector = get_query_embedding(query)
        if query_vector:
            embedding_results = search_embedding_documents(
                es, index_name, query_vector, size=candidate_size
            )
            embedding_hits = _tag_hits(_hits(embedding_results), "embedding")

    if mode == "bm25":
        return _sources_from_hits(bm25_hits[:size], include_metadata)
    if mode == "embedding":
        return _sources_from_hits(embedding_hits[:size], include_metadata)

    if mode == "weighted":
        alpha = float(os.getenv("RAG_HYBRID_ALPHA", "0.5"))
        alpha = min(max(alpha, 0.0), 1.0)
        merged_hits = _merge_hits_by_weighted_sum(
            bm25_hits, embedding_hits, top_k=size, alpha=alpha
        )
        return _sources_from_hits(merged_hits, include_metadata)

    merged_hits = _merge_hits_by_rrf(bm25_hits, embedding_hits, top_k=size)
    return _sources_from_hits(merged_hits, include_metadata)


def main_ES(x, mode: str | None = None):
    # 默认使用 hybrid，仍兼容原来的 main_ES(query) 调用方式。
    return retrieve_documents(x, mode=mode)
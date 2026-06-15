import os
import time
from datetime import datetime, timezone
from typing import List, Optional

import boto3
import uvicorn
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# metrics
from monitoring.helper import push_metric
from monitoring.retrieval_metrics import compute_similarity_and_precision_like, compute_recall_like

# Load environment variables
load_dotenv()

# env constants
AWS_REGION = os.environ["AWS_REGION"]
AWS_ACCOUNT_ID = os.environ["AWS_ACCOUNT_ID"]
BEDROCK_EMBEDDING_MODEL_ID = os.environ["BEDROCK_EMBEDDING_MODEL_ID"]

S3_VECTOR_BUCKET = os.environ["S3_VECTOR_BUCKET_NAME"]   # name of your S3 vector bucket
S3_VECTOR_INDEX = os.environ["S3_VECTOR_INDEX_NAME"]     # index name inside the vector bucket

PUT_VECTORS_BATCH_SIZE = 100
HOST = os.environ["VECTOR_DB_HOST"]
PORT = int(os.environ["VECTOR_DB_PORT"])

# aws clients
config = Config(
    read_timeout=60,
    connect_timeout=10,
    retries={"max_attempts": 3}
)
bedrock_rt = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=config)
s3_vectors = boto3.client("s3vectors", region_name=AWS_REGION, config=config)

app = FastAPI(title="RAG Vector Store")


# ---------- helpers ----------

def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------- models ----------

class EmbeddingItem(BaseModel):
    id: str
    text: str
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    query: str


# ---------- /store endpoint ----------

@app.post("/store")
def store_embeddings(items: List[EmbeddingItem]):
    vectors_payload = []
    for item in items:
        embed_response = bedrock_rt.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL_ID,
            body={"inputText": item.text}
        )
        embedding = embed_response  # replace with actual parsed embedding
        vectors_payload.append({
            "key": item.id,
            "data": {"float32": embedding},
            "metadata": item.metadata or {}
        })

    stored = 0
    try:
        embedding_start_time = time.time()
        for batch in chunk_list(vectors_payload, PUT_VECTORS_BATCH_SIZE):
            s3_vectors.put_vectors(
                indexArn=f"arn:aws:s3vectors:{AWS_REGION}:{AWS_ACCOUNT_ID}:bucket/{S3_VECTOR_BUCKET}/index/{S3_VECTOR_INDEX}",
                vectors=batch
            )
            stored += len(batch)

        # ingestion metrics
        embedding_latency = time.time() - embedding_start_time
        push_metric("IndexSizeVectors", stored, "Count", namespace="RAG/Embeddings")
        push_metric("EmbeddingVectorsLatency", embedding_latency, "Seconds", namespace="RAG/Embeddings")

        last_update_time = datetime.now(timezone.utc)
        freshness_days = (datetime.now(timezone.utc) - last_update_time).days
        push_metric("IndexFreshnessDays", freshness_days, None, namespace="RAG/Embeddings")

        return {"status": "success", "stored": stored}

    except s3_vectors.exceptions.TooManyRequestsException as e:
        push_metric("IngestionError", 1, "Count", namespace="RAG/Embeddings")
        raise HTTPException(status_code=429, detail=f"Rate limited by S3 Vectors: {str(e)}")
    except Exception as e:
        push_metric("EmbeddingError", 1, "Count", namespace="RAG/Embeddings")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------- /search endpoint ----------

@app.post("/search")
async def search_query(request: QueryRequest):
    start_time = time.time()
    query = request.query

    try:
        # get query embedding
        embed_response = bedrock_rt.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL_ID,
            body={"inputText": query}
        )
        query_vec_f32 = embed_response  # replace with actual parsed embedding

        top_k = 5
        response = s3_vectors.query_vectors(
            indexArn=f"arn:aws:s3vectors:{AWS_REGION}:{AWS_ACCOUNT_ID}:bucket/{S3_VECTOR_BUCKET}/index/{S3_VECTOR_INDEX}",
            topK=top_k,
            queryVector={"float32": query_vec_f32},
            filter=None,
            returnMetadata=True,
            returnDistance=True
        )

        # retrieval metrics
        latency = time.time() - start_time
        push_metric("QueryLatency", latency, "Seconds", namespace="RAG/VectorDB")
        push_metric("IndexHealthStatus", 1, "Status", namespace="RAG/VectorDB")

        vectors = response.get("vectors", [])
        payload = [
            {
                "id": v.get("key"),
                "score": v.get("distance"),
                "metadata": v.get("metadata", {})
            }
            for v in vectors
        ]

        # similarity + precision-like metric
        avg_sim, precision_like = compute_similarity_and_precision_like(
            payload, metric="COSINE", top_weighted=True
        )
        push_metric("AvgSimilarity", avg_sim, "None", namespace="RAG/VectorDB")
        push_metric("PrecisionProxy", precision_like, "None", namespace="RAG/VectorDB")

        # recall-like metric
        recall_like = compute_recall_like(vectors=vectors)
        push_metric("RecallProxy", recall_like, "None", namespace="RAG/VectorDB")

        return payload

    except s3_vectors.exceptions.AccessDeniedException as e:
        push_metric("QueryFailures", 1, "Count", namespace="RAG/VectorDB")
        push_metric("IndexHealthStatus", 0, "Status", namespace="RAG/VectorDB")
        raise HTTPException(status_code=403, detail=f"Access denied: {e}")
    except Exception as e:
        push_metric("QueryFailures", 1, "Count", namespace="RAG/VectorDB")
        push_metric("IndexHealthStatus", 0, "Status", namespace="RAG/VectorDB")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("vector_store.index:app", host=HOST, port=PORT, reload=True)

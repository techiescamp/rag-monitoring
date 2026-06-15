import os
import time
from datetime import datetime, timezone

import boto3
import requests
import uvicorn
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from monitoring.helper import push_metric

# Load environment variables
load_dotenv()

# env constants
AWS_REGION = os.environ["AWS_REGION"]
BEDROCK_LLM_MODEL = os.environ["BEDROCK_LLM_MODEL_ID"]
VECTOR_DB_URL = os.environ["VECTOR_DB_URL"]   # e.g. http://localhost:8001

# aws clients
config = Config(
    connect_timeout=60,
    read_timeout=10,
    retries={'max_attempts': 3}
)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=config)

app = FastAPI(title="RAG Main Backend")


class QueryRequest(BaseModel):
    query: str


# simple in-memory conversation memory placeholder
class Memory:
    def save_context(self, inputs, outputs):
        pass


memory = Memory()


def evaluate_faithfulness(bedrock_client, model_id, context, answer):
    """LLM-as-judge: how grounded is the answer in the context (0-1)."""
    judge_prompt = f"""
    Context:\n{context}\n
    Answer:\n{answer}\n
    On a scale of 0 to 1, how well is the answer grounded in the context?
    Reply with only a number between 0 and 1.
    """
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": judge_prompt}]}]
        )
        score_text = response["output"]["message"]["content"][0]["text"]
        score = float(score_text.strip())
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"Faithfulness eval failed: {e}")
        return None


@app.post("/query")
async def query_rag(request: QueryRequest):
    # end-to-end metric start
    end_to_end_start = time.time()

    query = request.query
    search_payload = {"query": query}

    # ---- retrieval ----
    retrieval_start_time = time.time()
    try:
        response = requests.post(f"{VECTOR_DB_URL}/search", json=search_payload, timeout=15)

        retrieval_latency = time.time() - retrieval_start_time
        push_metric("RetrievalLatency", retrieval_latency, "Seconds", namespace="RAG/Retrieval")

        if response.status_code != 200:
            push_metric("RetrievalFailures", 1, "Count", namespace="RAG/Retrieval")
            raise HTTPException(status_code=502, detail=f"Vector DB Error: {response.text}")

    except HTTPException:
        raise
    except Exception as e:
        push_metric("RetrievalFailures", 1, "Count", namespace="RAG/Retrieval")
        raise HTTPException(status_code=502, detail=f"Vector DB connection error: {e}")

    results = response.json()

    num_docs = len(results)
    push_metric("RetrievedDocsCount", num_docs, "Count", namespace="RAG/Retrieval")

    # build context + sources from retrieved docs
    context = " ".join(doc.get("metadata", {}).get("text", "") for doc in results)
    source_list = [doc.get("metadata", {}).get("source") for doc in results]

    system_prompt = "You are a helpful assistant. Answer using only the provided context."
    messages = []
    messages.append({"role": "user", "content": [{"text": f"Context: {context}\nQuestion: {query}"}]})

    # ---- generation ----
    generation_start_time = time.time()
    response = bedrock.converse(
        modelId=BEDROCK_LLM_MODEL,
        system=[{'text': system_prompt}],
        messages=messages,
        inferenceConfig={
            "maxTokens": 512,
            "temperature": 0.7
        }
    )
    llm_latency = time.time() - generation_start_time
    push_metric('LLMLatency', llm_latency, 'Seconds', namespace="RAG/Generation")

    output = response["output"]["message"]["content"][0]["text"]

    # response length
    response_length = len(output.split())
    push_metric('ResponseLength', response_length, 'Count', namespace="RAG/Generation")

    # hallucination rate (word-overlap heuristic)
    output_sentences = [p.strip() for p in output.split('.') if p.strip()]
    hallucinated = 0
    context_words = context.lower().split()

    for sent in output_sentences:
        sent_words = sent.lower().split()
        overlap = sum(1 for w in sent_words if w in context_words)
        if overlap < 3:  # fewer than 3 overlapping words -> likely hallucinated
            hallucinated += 1

    hallucination_ratio = hallucinated / max(len(output_sentences), 1)
    push_metric('HallucinationRate', hallucination_ratio, 'None', namespace="RAG/Generation")

    # token usage metrics
    response_metadata = response.get("ResponseMetadata", {})
    usage_data = response_metadata.get('usage', {})
    if usage_data:
        input_tokens = usage_data.get('inputTokens', 0)
        output_tokens = usage_data.get('outputTokens', 0)
        total_tokens = usage_data.get('totalTokens', 0)

        push_metric("LLMInputTokens", input_tokens, 'Count', namespace="RAG/Generation")
        push_metric("LLMOutputTokens", output_tokens, 'Count', namespace="RAG/Generation")
        push_metric("LLMTotalTokens", total_tokens, 'Count', namespace="RAG/Generation")

    # bedrock-reported latency
    metrics_data = response.get('metrics', {})
    if metrics_data:
        llm_latency_bedrock = metrics_data.get('latencyMs', 0)
        push_metric("LLMLatencyBedrock", llm_latency_bedrock, 'Milliseconds', namespace="RAG/Generation")

    # faithfulness score
    score = evaluate_faithfulness(bedrock, BEDROCK_LLM_MODEL, context, output)
    if score is not None:
        push_metric("FaithfulnessScore", score, "None", namespace="RAG/Generation")

    memory.save_context({'question': query}, {'answer': output})

    # end-to-end latency
    end_to_end_latency = time.time() - end_to_end_start
    push_metric("EndToEndLatency", end_to_end_latency, "Seconds", namespace="RAG/Generation")

    return {"answer": output, "source": source_list}


if __name__ == "__main__":
    uvicorn.run("main_backend.app:app", host="0.0.0.0", port=8000, reload=True)

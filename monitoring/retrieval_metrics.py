import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from monitoring.helper import normalize_similarity


def compute_similarity_and_precision_like(payload, metric='COSINE', top_weighted=False):
    """
    payload: list of retrieved docs with 'score' from S3 Vectors
    Returns:
        avg_similarity: average similarity of retrieved docs to query
        precision_like: proxy precision (same as avg_similarity or top-weighted)
    """
    scores = [doc.get('score') for doc in payload if doc.get('score') is not None]
    if not scores:
        avg_similarity = 0.0
        precision_like = 0.0
    else:
        similarities = [normalize_similarity(d, metric) for d in scores]
        avg_similarity = float(np.mean(similarities))
        if top_weighted:
            weights = np.linspace(len(similarities), 1, len(similarities))
            precision_like = float(np.average(similarities, weights=weights))
        else:
            precision_like = avg_similarity
    return avg_similarity, precision_like


def compute_recall_like(vectors):
    """Measures diversity among top-K retrieved chunks."""
    retrieved_vecs = [np.array(v['data']['float32']) for v in vectors if v.get('data')]
    if len(retrieved_vecs) < 2:
        return 0.0
    sims = cosine_similarity(retrieved_vecs)
    np.fill_diagonal(sims, 0)
    diversity = 1 - np.mean(sims)  # higher = more diversity
    return diversity

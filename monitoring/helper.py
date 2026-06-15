import os
import boto3
import numpy as np
from datetime import datetime, timezone

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)


def push_metric(name, value, unit='None', namespace='RAG/Custom'):
    """Push a custom metric to CloudWatch."""
    try:
        cloudwatch.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    "MetricName": name,
                    "Timestamp": datetime.now(timezone.utc),
                    "Value": value,
                    "Unit": str(unit) if unit else "None"
                }
            ]
        )
    except Exception as e:
        print(f"CloudWatch metric push failed: {e}")


def normalize_similarity(distance, metric='COSINE'):
    """
    Convert S3 Vector distance to a similarity score (0-1).
    - COSINE: similarity = 1 - distance
    - EUCLIDEAN: similarity = 1 / (1 + distance)
    """
    if distance is None:
        return 0.0
    if metric.upper() == 'COSINE':
        return 1.0 - distance
    elif metric.upper() == 'EUCLIDEAN':
        return 1.0 / (1.0 + distance)
    return 0.0

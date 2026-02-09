"""
Diagnostic to check if the similarity formula 1/(1+L2) is appropriate
for the embedding model in use (databricks-gte-large-en).

The hypothesis: L2 distances for 1024-dim embeddings can be very large,
making 1/(1+L2) give very low similarities even for semantically similar texts.
Cosine similarity might be a better metric.

Run with:
    python -m pytest tests/dao_ai/genie/test_similarity_formula_diagnostic.py -v -s
"""

import os
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from dao_ai.config import (  # noqa: E402
    DatabaseModel,
    GenieContextAwareCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie.cache import PostgresContextAwareGenieService  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("DATABRICKS_WAREHOUSE_ID"),
        reason="DATABRICKS_WAREHOUSE_ID not set",
    ),
]


@pytest.fixture
def embedding_service() -> PostgresContextAwareGenieService:
    """Create a service just to access the embedding model."""
    database = DatabaseModel(instance_name="retail-consumer-goods")
    wh_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    assert wh_id
    warehouse = WarehouseModel(warehouse_id=wh_id)

    test_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    params = GenieContextAwareCacheParametersModel(
        database=database,
        warehouse=warehouse,
        embedding_model="databricks-gte-large-en",
        table_name=f"test_sim_formula_{test_id}",
        prompt_history_table=f"test_sim_formula_prompts_{test_id}",
        time_to_live_seconds=3600,
        similarity_threshold=0.85,
    )

    mock_impl = Mock()
    mock_impl.space_id = "test-formula-space"

    service = PostgresContextAwareGenieService(
        impl=mock_impl,
        parameters=params,
        workspace_client=None,
    )
    service.initialize()

    yield service

    try:
        service.drop_tables()
    except Exception:
        pass


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def l2_distance(a: list[float], b: list[float]) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.linalg.norm(a_arr - b_arr))


def l2_to_similarity(dist: float) -> float:
    """Convert L2 distance to similarity using 1/(1+d)."""
    return 1.0 / (1.0 + dist)


class TestSimilarityFormulaDiagnostic:
    """Compare cosine similarity vs L2-based similarity for the embedding model."""

    QUESTION_PAIRS = [
        (
            "What are the ingredients in a vanilla latte?",
            "What ingredients does a vanilla latte have?",
            "nearly identical",
        ),
        (
            "What are the ingredients in a vanilla latte?",
            "Tell me the ingredients for a vanilla latte",
            "same intent, different phrasing",
        ),
        (
            "What are the ingredients in a vanilla latte?",
            "What is in a vanilla latte?",
            "abbreviated form",
        ),
        (
            "What are the ingredients in a vanilla latte?",
            "How many stores are in California?",
            "completely different",
        ),
        (
            "What are the top selling items?",
            "What are our best selling products?",
            "synonym substitution",
        ),
    ]

    def test_similarity_comparison(
        self, embedding_service: PostgresContextAwareGenieService
    ) -> None:
        """Compare cosine vs L2-based similarity for multiple question pairs."""
        embeddings_model = embedding_service._embeddings

        print("\n" + "=" * 90)
        print(
            f"{'Pair':<30} {'Cosine Sim':>12} {'L2 Dist':>12} {'1/(1+L2)':>12} {'Threshold':>12}"
        )
        print("=" * 90)

        all_questions = []
        for q1, q2, label in self.QUESTION_PAIRS:
            all_questions.extend([q1, q2])

        # Batch embed all questions at once for efficiency
        all_embeddings = embeddings_model.embed_documents(all_questions)

        for i, (q1, q2, label) in enumerate(self.QUESTION_PAIRS):
            emb1 = all_embeddings[i * 2]
            emb2 = all_embeddings[i * 2 + 1]

            cos_sim = cosine_similarity(emb1, emb2)
            l2_dist = l2_distance(emb1, emb2)
            l2_sim = l2_to_similarity(l2_dist)

            threshold_status = "PASS" if l2_sim >= 0.85 else "FAIL"

            print(
                f"  {label:<28} {cos_sim:>12.4f} {l2_dist:>12.4f} {l2_sim:>12.4f} {threshold_status:>12}"
            )

        # Also show embedding norms (are they normalized?)
        print("\n  Embedding norms (first question of each pair):")
        for i, (q1, _, label) in enumerate(self.QUESTION_PAIRS):
            emb = all_embeddings[i * 2]
            norm = float(np.linalg.norm(np.array(emb)))
            print(f"    {label:<28} norm={norm:.4f}")

        print("\n  Summary:")
        print("  - If cosine similarity is high but 1/(1+L2) is low,")
        print(
            "    the issue is that L2 distance is too large for unnormalized vectors."
        )
        print("  - Fix: use cosine distance (<=> in pg_vector) instead of L2 (<->)")
        print("    or normalize embeddings before storing/querying.")

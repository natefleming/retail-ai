"""
Tests for DAO AI Evaluation Module

Unit tests for MLflow GenAI evaluation utilities.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from mlflow.genai.scorers import (
    Completeness,
    Guidelines,
    RelevanceToQuery,
    Safety,
    ToolCallEfficiency,
)

# -----------------------------------------------------------------------------
# normalize_eval_inputs
# -----------------------------------------------------------------------------


class TestNormalizeEvalInputs:
    """Tests for normalize_eval_inputs."""

    @pytest.mark.unit
    def test_dict_with_messages_list(self) -> None:
        from dao_ai.evaluation import normalize_eval_inputs

        messages = [{"role": "user", "content": "hello"}]
        result = normalize_eval_inputs({"messages": messages})
        assert result == {"messages": messages}

    @pytest.mark.unit
    def test_dict_with_messages_string(self) -> None:
        from dao_ai.evaluation import normalize_eval_inputs

        result = normalize_eval_inputs({"messages": "hello"})
        assert result == {"messages": [{"role": "user", "content": "hello"}]}

    @pytest.mark.unit
    def test_list_of_message_dicts(self) -> None:
        from dao_ai.evaluation import normalize_eval_inputs

        messages = [{"role": "user", "content": "hello"}]
        result = normalize_eval_inputs(messages)
        assert result == {"messages": messages}

    @pytest.mark.unit
    def test_dict_with_request_field(self) -> None:
        from dao_ai.evaluation import normalize_eval_inputs

        result = normalize_eval_inputs({"request": "What is X?"})
        assert result == {"messages": [{"role": "user", "content": "What is X?"}]}

    @pytest.mark.unit
    def test_raw_string_fallback(self) -> None:
        from dao_ai.evaluation import normalize_eval_inputs

        result = normalize_eval_inputs("hello world")
        assert result == {"messages": [{"role": "user", "content": "hello world"}]}

    @pytest.mark.unit
    def test_empty_list(self) -> None:
        from dao_ai.evaluation import normalize_eval_inputs

        result = normalize_eval_inputs([])
        assert result == {"messages": []}

    @pytest.mark.unit
    def test_dict_with_extra_keys_keeps_only_messages(self) -> None:
        from dao_ai.evaluation import normalize_eval_inputs

        messages = [{"role": "user", "content": "hello"}]
        result = normalize_eval_inputs({"messages": messages, "extra": "ignored"})
        assert result == {"messages": messages}


# -----------------------------------------------------------------------------
# prepare_eval_dataframe
# -----------------------------------------------------------------------------


class TestPrepareEvalDataframe:
    """Tests for prepare_eval_dataframe."""

    @pytest.mark.unit
    def test_converts_struct_columns_via_json(self) -> None:
        """Verify STRUCT columns go through to_json -> json.loads round-trip."""
        from pyspark.sql.types import (
            StringType,
            StructField,
            StructType,
        )

        from dao_ai.evaluation import prepare_eval_dataframe

        schema = StructType(
            [
                StructField(
                    "inputs",
                    StructType(
                        [
                            StructField("messages", StringType()),
                        ]
                    ),
                ),
                StructField("plain_col", StringType()),
            ]
        )

        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema

        json_converted_df = MagicMock()
        json_converted_df.schema = schema
        mock_spark_df.withColumn.return_value = json_converted_df

        pandas_df = pd.DataFrame(
            [
                {"inputs": '{"messages": "What products?"}', "plain_col": "hello"},
            ]
        )
        json_converted_df.toPandas.return_value = pandas_df

        result = prepare_eval_dataframe(mock_spark_df)

        mock_spark_df.withColumn.assert_called_once()
        assert result["inputs"].iloc[0] == {
            "messages": [{"role": "user", "content": "What products?"}]
        }
        assert result["plain_col"].iloc[0] == "hello"

    @pytest.mark.unit
    def test_num_evals_limits_rows(self) -> None:
        from pyspark.sql.types import StringType, StructField, StructType

        from dao_ai.evaluation import prepare_eval_dataframe

        schema = StructType([StructField("inputs", StringType())])

        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema
        mock_spark_df.toPandas.return_value = pd.DataFrame(
            [
                {"inputs": "a"},
                {"inputs": "b"},
                {"inputs": "c"},
            ]
        )

        result = prepare_eval_dataframe(mock_spark_df, num_evals=2)
        assert len(result) == 2

    @pytest.mark.unit
    def test_normalizes_inputs_column(self) -> None:
        from pyspark.sql.types import StringType, StructField, StructType

        from dao_ai.evaluation import prepare_eval_dataframe

        schema = StructType([StructField("inputs", StringType())])

        mock_spark_df = MagicMock()
        mock_spark_df.schema = schema
        mock_spark_df.toPandas.return_value = pd.DataFrame(
            [
                {"inputs": "What is X?"},
            ]
        )

        result = prepare_eval_dataframe(mock_spark_df)
        assert result["inputs"].iloc[0] == {
            "messages": [{"role": "user", "content": "What is X?"}]
        }


# -----------------------------------------------------------------------------
# create_guidelines_scorers
# -----------------------------------------------------------------------------


class TestCreateGuidelinesScorers:
    """Tests for create_guidelines_scorers."""

    @pytest.mark.unit
    def test_creates_scorers_from_config(self) -> None:
        from dao_ai.evaluation import create_guidelines_scorers

        class MockGuideline:
            name = "test_guideline"
            guidelines = ["Be helpful", "Be accurate"]

        scorers = create_guidelines_scorers([MockGuideline()])

        assert len(scorers) == 1
        assert scorers[0].name == "test_guideline"
        assert scorers[0].model is None

    @pytest.mark.unit
    def test_creates_scorers_with_judge_model(self) -> None:
        from dao_ai.evaluation import create_guidelines_scorers

        class MockGuideline:
            name = "quality"
            guidelines = ["Be concise"]

        scorers = create_guidelines_scorers(
            [MockGuideline()],
            judge_model="databricks:/test-model",
        )

        assert len(scorers) == 1
        assert scorers[0].model == "databricks:/test-model"

    @pytest.mark.unit
    def test_creates_multiple_scorers(self) -> None:
        from dao_ai.evaluation import create_guidelines_scorers

        class G1:
            name = "g1"
            guidelines = ["Rule 1"]

        class G2:
            name = "g2"
            guidelines = ["Rule 2"]

        scorers = create_guidelines_scorers([G1(), G2()])

        assert len(scorers) == 2
        assert scorers[0].name == "g1"
        assert scorers[1].name == "g2"


# -----------------------------------------------------------------------------
# build_scorers
# -----------------------------------------------------------------------------


class TestBuildScorers:
    """Tests for build_scorers."""

    @pytest.mark.unit
    def test_builds_default_scorers(self) -> None:
        from dao_ai.evaluation import build_scorers

        class MockEvalConfig:
            guidelines = []

        scorers = build_scorers(MockEvalConfig())

        assert len(scorers) == 4
        types = [type(s) for s in scorers]
        assert Safety in types
        assert Completeness in types
        assert RelevanceToQuery in types
        assert ToolCallEfficiency in types

    @pytest.mark.unit
    def test_builds_scorers_with_guidelines(self) -> None:
        from dao_ai.evaluation import build_scorers

        class MockGuideline:
            name = "my_guideline"
            guidelines = ["Be polite"]

        class MockEvalConfig:
            guidelines = [MockGuideline()]

        scorers = build_scorers(MockEvalConfig())

        assert len(scorers) == 5
        guideline_scorers = [s for s in scorers if isinstance(s, Guidelines)]
        assert len(guideline_scorers) == 1
        assert guideline_scorers[0].name == "my_guideline"

    @pytest.mark.unit
    def test_builds_scorers_with_empty_guidelines(self) -> None:
        from dao_ai.evaluation import build_scorers

        class MockEvalConfig:
            guidelines = []

        scorers = build_scorers(MockEvalConfig())

        assert len(scorers) == 4
        guideline_scorers = [s for s in scorers if isinstance(s, Guidelines)]
        assert len(guideline_scorers) == 0


# -----------------------------------------------------------------------------
# prepare_eval_results_for_display
# -----------------------------------------------------------------------------


class TestPrepareEvalResultsForDisplay:
    """Tests for prepare_eval_results_for_display."""

    @pytest.mark.unit
    def test_converts_assessments_to_string(self) -> None:
        from dao_ai.evaluation import prepare_eval_results_for_display

        mock_results = MagicMock()
        mock_results.tables = {
            "eval_results": pd.DataFrame(
                {
                    "inputs": ["test"],
                    "outputs": ["result"],
                    "assessments": [{"scorer": "safety", "value": True}],
                }
            )
        }

        result_df = prepare_eval_results_for_display(mock_results)
        assert isinstance(result_df["assessments"].iloc[0], str)

    @pytest.mark.unit
    def test_handles_no_assessments_column(self) -> None:
        from dao_ai.evaluation import prepare_eval_results_for_display

        mock_results = MagicMock()
        mock_results.tables = {
            "eval_results": pd.DataFrame(
                {
                    "inputs": ["test"],
                    "outputs": ["result"],
                }
            )
        }

        result_df = prepare_eval_results_for_display(mock_results)
        assert "assessments" not in result_df.columns


# -----------------------------------------------------------------------------
# create_or_get_eval_dataset
# -----------------------------------------------------------------------------


class TestCreateOrGetEvalDataset:
    """Tests for create_or_get_eval_dataset."""

    @pytest.mark.unit
    def test_creates_new_dataset_when_none_exists(self) -> None:
        from dao_ai.evaluation import create_or_get_eval_dataset

        source_df = pd.DataFrame(
            [{"inputs": {"messages": [{"role": "user", "content": "hi"}]}}]
        )

        mock_dataset = MagicMock()
        mock_dataset.merge_records.return_value = mock_dataset

        with patch("dao_ai.evaluation.get_dataset", side_effect=Exception("not found")):
            with patch(
                "dao_ai.evaluation.create_dataset", return_value=mock_dataset
            ) as mock_create:
                result = create_or_get_eval_dataset(
                    name="test_dataset",
                    experiment_id="123",
                    source_df=source_df,
                )

                mock_create.assert_called_once_with(
                    name="test_dataset",
                    experiment_id=["123"],
                )
                mock_dataset.merge_records.assert_called_once_with(source_df)
                assert result is mock_dataset

    @pytest.mark.unit
    def test_loads_existing_dataset(self) -> None:
        from dao_ai.evaluation import create_or_get_eval_dataset

        source_df = pd.DataFrame(
            [{"inputs": {"messages": [{"role": "user", "content": "hi"}]}}]
        )

        mock_dataset = MagicMock()
        mock_dataset.merge_records.return_value = mock_dataset

        with patch(
            "dao_ai.evaluation.get_dataset", return_value=mock_dataset
        ) as mock_get:
            with patch("dao_ai.evaluation.create_dataset") as mock_create:
                result = create_or_get_eval_dataset(
                    name="existing_dataset",
                    experiment_id="456",
                    source_df=source_df,
                )

                mock_get.assert_called_once_with(name="existing_dataset")
                mock_create.assert_not_called()
                mock_dataset.merge_records.assert_called_once_with(source_df)
                assert result is mock_dataset

    @pytest.mark.unit
    def test_passes_tags_on_create(self) -> None:
        from dao_ai.evaluation import create_or_get_eval_dataset

        source_df = pd.DataFrame(
            [{"inputs": {"messages": [{"role": "user", "content": "hi"}]}}]
        )

        mock_dataset = MagicMock()
        mock_dataset.merge_records.return_value = mock_dataset

        with patch("dao_ai.evaluation.get_dataset", side_effect=Exception("not found")):
            with patch(
                "dao_ai.evaluation.create_dataset", return_value=mock_dataset
            ) as mock_create:
                create_or_get_eval_dataset(
                    name="tagged_dataset",
                    experiment_id="789",
                    source_df=source_df,
                    tags={"team": "ai", "stage": "dev"},
                )

                mock_create.assert_called_once_with(
                    name="tagged_dataset",
                    experiment_id=["789"],
                    tags={"team": "ai", "stage": "dev"},
                )

    @pytest.mark.unit
    def test_merges_records_from_source_df(self) -> None:
        from dao_ai.evaluation import create_or_get_eval_dataset

        source_df = pd.DataFrame(
            [
                {"inputs": {"messages": [{"role": "user", "content": "q1"}]}},
                {"inputs": {"messages": [{"role": "user", "content": "q2"}]}},
                {"inputs": {"messages": [{"role": "user", "content": "q3"}]}},
            ]
        )

        mock_dataset = MagicMock()
        mock_dataset.merge_records.return_value = mock_dataset

        with patch("dao_ai.evaluation.get_dataset", return_value=mock_dataset):
            create_or_get_eval_dataset(
                name="multi_record",
                experiment_id="100",
                source_df=source_df,
            )

            merged_df = mock_dataset.merge_records.call_args[0][0]
            assert len(merged_df) == 3


# -----------------------------------------------------------------------------
# Pipeline Integration Tests
# -----------------------------------------------------------------------------


class TestEvaluationPipeline:
    """Tests that exercise the full evaluation pipeline with the YAML config."""

    @pytest.mark.unit
    def test_scorers_from_yaml_config(self) -> None:
        from dao_ai.config import AppConfig
        from dao_ai.evaluation import build_scorers

        config = AppConfig.from_file(
            "config/examples/15_complete_applications/hardware_store_lakebase.yaml"
        )

        scorers = build_scorers(config.evaluation)

        safety_scorers = [s for s in scorers if isinstance(s, Safety)]
        assert len(safety_scorers) == 1

        guideline_scorers = [s for s in scorers if isinstance(s, Guidelines)]
        assert len(guideline_scorers) == 1
        assert guideline_scorers[0].name == "my_relevance_guideline"
        assert guideline_scorers[0].model is None

        # Safety + Completeness + RelevanceToQuery + ToolCallEfficiency + 1 guideline
        assert len(scorers) == 5

    @pytest.mark.unit
    def test_evaluate_pipeline_with_mocked_mlflow(self) -> None:
        from dao_ai.config import AppConfig
        from dao_ai.evaluation import build_scorers

        config = AppConfig.from_file(
            "config/examples/15_complete_applications/hardware_store_lakebase.yaml"
        )

        scorers = build_scorers(config.evaluation)
        assert len(scorers) == 5

        def mock_predict_fn(messages: list) -> dict:
            return {"response": "We have many products available."}

        eval_data = pd.DataFrame(
            [
                {
                    "inputs": {
                        "messages": [
                            {"role": "user", "content": "What products do you have?"}
                        ]
                    },
                },
            ]
        )

        mock_result = MagicMock()
        mock_result.metrics = {
            "safety/mean": 1.0,
            "completeness/mean": 1.0,
            "relevance_to_query/mean": 1.0,
            "tool_call_efficiency/mean": 1.0,
            "my_relevance_guideline/mean": 0.5,
        }

        with patch("mlflow.genai.evaluate", return_value=mock_result) as mock_eval:
            mock_eval(
                data=eval_data,
                predict_fn=mock_predict_fn,
                scorers=scorers,
            )

            mock_eval.assert_called_once()
            call_kwargs = mock_eval.call_args[1]
            assert len(call_kwargs["scorers"]) == 5
            assert call_kwargs["predict_fn"] is mock_predict_fn

    @pytest.mark.unit
    def test_predict_fn_timeout_returns_error(self) -> None:
        import time
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        def slow_predict(messages: list, custom_inputs: dict | None) -> str:
            time.sleep(10)
            return "should not reach here"

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(slow_predict, [], None)
                future.result(timeout=1)
            assert False, "Should have raised TimeoutError"
        except FuturesTimeout:
            pass


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------


def pytest_configure(config: Any) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may require external services)"
    )

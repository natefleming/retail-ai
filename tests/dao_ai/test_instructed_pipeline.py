"""Unit tests for the backend-agnostic instructed-retrieval pipeline.

Covers `execute_instructed_pipeline` with a fake `run_search` fixture:
- Router mode selection (standard vs instructed, fallback, no-router path).
- Decompose → parallel search → RRF merge (happy path).
- Empty decomposition → fallback single call.
- Exception in decompose → fallback single call.
- Empty merge → fallback single call.
- FlashRank invocation position.
- Instruction-aware rerank called for instructed, skipped for standard+auto_bypass.
- Verifier retry loop: passed / warn / retry-then-pass / exhausted paths.
- Standard mode never retries.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from dao_ai.config import (
    ColumnInfo,
    DecompositionModel,
    FilterItem,
    InferenceEndpointModel,
    InstructedRetrieverModel,
    InstructionAwareRerankModel,
    RouterModel,
    SearchQuery,
    VerificationResult,
    VerifierModel,
)
from dao_ai.tools.instructed_pipeline import (
    _decide_mode,
    execute_instructed_pipeline,
)


@pytest.fixture()
def columns() -> list[ColumnInfo]:
    return [
        ColumnInfo(name="category", type="string"),
        ColumnInfo(name="priority", type="number"),
    ]


@pytest.fixture()
def llm() -> InferenceEndpointModel:
    return InferenceEndpointModel(name="databricks-claude-sonnet-4-5")


@pytest.fixture()
def decomposition(llm: InferenceEndpointModel) -> DecompositionModel:
    return DecompositionModel(model=llm, max_subqueries=2, rrf_k=60)


@pytest.fixture()
def instructed_cfg(
    llm: InferenceEndpointModel,
    decomposition: DecompositionModel,
    columns: list[ColumnInfo],
) -> InstructedRetrieverModel:
    return InstructedRetrieverModel(columns=columns, decomposition=decomposition)


def _docs(*ids: str) -> list[Document]:
    return [
        Document(
            page_content=f"content of {i}",
            metadata={"id": i, "primary_key": i},
        )
        for i in ids
    ]


class TestDecideMode:
    def test_no_router_no_instructed_is_standard(self) -> None:
        mode, bypass = _decide_mode(
            query="hi",
            router_config=None,
            instructed_config=None,
            instructed_columns=[],
        )
        assert mode == "standard"
        assert bypass is True

    def test_no_router_with_instructed_is_instructed(
        self, instructed_cfg: InstructedRetrieverModel
    ) -> None:
        mode, bypass = _decide_mode(
            query="hi",
            router_config=None,
            instructed_config=instructed_cfg,
            instructed_columns=instructed_cfg.columns,
        )
        assert mode == "instructed"
        assert bypass is False

    def test_router_no_llm_uses_default_mode(
        self, instructed_cfg: InstructedRetrieverModel
    ) -> None:
        rc = RouterModel(default_mode="instructed", auto_bypass=True)
        mode, bypass = _decide_mode(
            query="hi",
            router_config=rc,
            instructed_config=instructed_cfg,
            instructed_columns=instructed_cfg.columns,
        )
        assert mode == "instructed"
        assert bypass is True

    def test_router_llm_success(
        self,
        instructed_cfg: InstructedRetrieverModel,
        llm: InferenceEndpointModel,
    ) -> None:
        rc = RouterModel(model=llm, default_mode="standard", auto_bypass=True)
        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch(
                "dao_ai.tools.instructed_pipeline.route_query",
                return_value="instructed",
            ),
        ):
            mock_llm.return_value = MagicMock()
            mode, _ = _decide_mode(
                query="Milwaukee power tools under $200",
                router_config=rc,
                instructed_config=instructed_cfg,
                instructed_columns=instructed_cfg.columns,
            )
        assert mode == "instructed"

    def test_router_llm_failure_falls_back(
        self,
        instructed_cfg: InstructedRetrieverModel,
        llm: InferenceEndpointModel,
    ) -> None:
        rc = RouterModel(model=llm, default_mode="instructed", auto_bypass=False)
        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch(
                "dao_ai.tools.instructed_pipeline.route_query",
                side_effect=RuntimeError("boom"),
            ),
        ):
            mock_llm.return_value = MagicMock()
            mode, bypass = _decide_mode(
                query="hi",
                router_config=rc,
                instructed_config=instructed_cfg,
                instructed_columns=instructed_cfg.columns,
            )
        assert mode == "instructed"
        assert bypass is False


class TestStandardMode:
    def test_standard_runs_one_search(self, columns: list[ColumnInfo]) -> None:
        calls: list[tuple[str, dict]] = []

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            calls.append((q, f))
            return _docs("a", "b")

        docs = execute_instructed_pipeline(
            run_search=run_search,
            query="hello",
            base_filters={"category": "x"},
            instructed_config=None,
            router_config=None,
            verifier_config=None,
            decomposition_config=None,
            instruction_rerank_config=None,
            instructed_columns=columns,
        )
        assert [d.metadata["id"] for d in docs] == ["a", "b"]
        assert calls == [("hello", {"category": "x"})]


class TestInstructedMode:
    def test_decomposes_and_merges(
        self, instructed_cfg: InstructedRetrieverModel
    ) -> None:
        calls: list[str] = []

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            calls.append(q)
            return _docs(f"result_for_{q}")

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query") as mock_decompose,
        ):
            mock_llm.return_value = MagicMock()
            mock_decompose.return_value = [
                SearchQuery(
                    text="sub1", filters=[FilterItem(key="category", value="a")]
                ),
                SearchQuery(
                    text="sub2", filters=[FilterItem(key="category", value="b")]
                ),
            ]
            docs = execute_instructed_pipeline(
                run_search=run_search,
                query="complex",
                base_filters={},
                instructed_config=instructed_cfg,
                router_config=None,
                verifier_config=None,
                decomposition_config=instructed_cfg.decomposition,
                instruction_rerank_config=None,
                instructed_columns=instructed_cfg.columns,
            )

        assert set(calls) == {"sub1", "sub2"}
        assert len(docs) == 2
        assert {d.metadata["id"] for d in docs} == {
            "result_for_sub1",
            "result_for_sub2",
        }

    def test_bad_type_filter_coerced_before_dispatch(
        self, instructed_cfg: InstructedRetrieverModel
    ) -> None:
        """Regression: LLM emits `{priority: "3"}` (string) on int column.

        The pipeline must coerce "3" → 3 BEFORE calling run_search — the
        earlier lakebase failure was decomposition returning a string
        value and Postgres 500'ing on the numeric column.
        """
        captured: list[dict[str, Any]] = []

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            captured.append(dict(f))
            return _docs("ok")

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query") as mock_decompose,
        ):
            mock_llm.return_value = MagicMock()
            mock_decompose.return_value = [
                SearchQuery(
                    text="high priority",
                    filters=[FilterItem(key="priority", value="3")],
                ),
            ]
            execute_instructed_pipeline(
                run_search=run_search,
                query="q",
                base_filters={},
                instructed_config=instructed_cfg,
                router_config=None,
                verifier_config=None,
                decomposition_config=instructed_cfg.decomposition,
                instruction_rerank_config=None,
                instructed_columns=instructed_cfg.columns,
            )
        # Coerced from "3" → 3 before dispatch.
        assert captured == [{"priority": 3}]

    def test_uncoercible_filter_dropped_and_search_continues(
        self, instructed_cfg: InstructedRetrieverModel
    ) -> None:
        """Regression of the earlier lakebase 500: `{priority: "high"}`
        on an int col is dropped-and-warned so the search still runs
        (degraded, one fewer filter) rather than hard-failing."""
        captured: list[dict[str, Any]] = []

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            captured.append(dict(f))
            return _docs("ok")

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query") as mock_decompose,
        ):
            mock_llm.return_value = MagicMock()
            mock_decompose.return_value = [
                SearchQuery(
                    text="most important",
                    filters=[
                        FilterItem(key="category", value="auth"),
                        FilterItem(key="priority", value="high"),  # bad
                    ],
                ),
            ]
            execute_instructed_pipeline(
                run_search=run_search,
                query="q",
                base_filters={},
                instructed_config=instructed_cfg,
                router_config=None,
                verifier_config=None,
                decomposition_config=instructed_cfg.decomposition,
                instruction_rerank_config=None,
                instructed_columns=instructed_cfg.columns,
            )
        # `priority: "high"` dropped; `category: "auth"` kept.
        assert captured == [{"category": "auth"}]

    def test_empty_decomposition_falls_back(
        self, instructed_cfg: InstructedRetrieverModel
    ) -> None:
        calls: list[tuple[str, dict]] = []

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            calls.append((q, f))
            return _docs("fallback")

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query", return_value=[]),
        ):
            mock_llm.return_value = MagicMock()
            docs = execute_instructed_pipeline(
                run_search=run_search,
                query="hi",
                base_filters={"a": 1},
                instructed_config=instructed_cfg,
                router_config=None,
                verifier_config=None,
                decomposition_config=instructed_cfg.decomposition,
                instruction_rerank_config=None,
                instructed_columns=instructed_cfg.columns,
            )
        assert calls == [("hi", {"a": 1})]
        assert docs[0].metadata["id"] == "fallback"

    def test_decomposition_exception_falls_back(
        self, instructed_cfg: InstructedRetrieverModel
    ) -> None:
        calls: list[tuple[str, dict]] = []

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            calls.append((q, f))
            return _docs("fallback")

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch(
                "dao_ai.tools.instructed_pipeline.decompose_query",
                side_effect=RuntimeError("LLM down"),
            ),
        ):
            mock_llm.return_value = MagicMock()
            docs = execute_instructed_pipeline(
                run_search=run_search,
                query="hi",
                base_filters={"cat": "x"},
                instructed_config=instructed_cfg,
                router_config=None,
                verifier_config=None,
                decomposition_config=instructed_cfg.decomposition,
                instruction_rerank_config=None,
                instructed_columns=instructed_cfg.columns,
            )
        assert calls == [("hi", {"cat": "x"})]
        assert docs

    def test_empty_merge_falls_back(
        self, instructed_cfg: InstructedRetrieverModel
    ) -> None:
        calls: list[str] = []

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            calls.append(q)
            return _docs("x")

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query") as mock_decompose,
            patch("dao_ai.tools.instructed_pipeline.rrf_merge", return_value=[]),
        ):
            mock_llm.return_value = MagicMock()
            mock_decompose.return_value = [
                SearchQuery(text="sub1", filters=None),
            ]
            docs = execute_instructed_pipeline(
                run_search=run_search,
                query="original",
                base_filters={},
                instructed_config=instructed_cfg,
                router_config=None,
                verifier_config=None,
                decomposition_config=instructed_cfg.decomposition,
                instruction_rerank_config=None,
                instructed_columns=instructed_cfg.columns,
            )
        assert "sub1" in calls
        assert "original" in calls
        assert docs


class TestFlashRankPosition:
    def test_runs_between_search_and_instruction_rerank(
        self, columns: list[ColumnInfo]
    ) -> None:
        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            return _docs("a", "b", "c")

        fake_ranker = MagicMock(name="Ranker")
        with patch("dao_ai.tools.vector_search.rerank_documents") as mock_rerank:
            mock_rerank.return_value = _docs("c", "b", "a")
            docs = execute_instructed_pipeline(
                run_search=run_search,
                query="q",
                base_filters={},
                instructed_config=None,
                router_config=None,
                verifier_config=None,
                decomposition_config=None,
                instruction_rerank_config=None,
                instructed_columns=columns,
                ranker=fake_ranker,
                rerank_config=MagicMock(),
            )

        mock_rerank.assert_called_once()
        called_docs = mock_rerank.call_args.args[1]
        assert [d.metadata["id"] for d in called_docs] == ["a", "b", "c"]
        assert [d.metadata["id"] for d in docs] == ["c", "b", "a"]

    def test_skipped_when_ranker_none(self, columns: list[ColumnInfo]) -> None:
        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            return _docs("a")

        with patch("dao_ai.tools.vector_search.rerank_documents") as mock_rerank:
            execute_instructed_pipeline(
                run_search=run_search,
                query="q",
                base_filters={},
                instructed_config=None,
                router_config=None,
                verifier_config=None,
                decomposition_config=None,
                instruction_rerank_config=None,
                instructed_columns=columns,
                ranker=None,
                rerank_config=None,
            )
        mock_rerank.assert_not_called()


class TestInstructionRerank:
    def test_skipped_for_standard_auto_bypass(
        self, columns: list[ColumnInfo], llm: InferenceEndpointModel
    ) -> None:
        rc = RouterModel(default_mode="standard", auto_bypass=True)
        instr_rerank = InstructionAwareRerankModel(model=llm)

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            return _docs("a")

        with patch(
            "dao_ai.tools.instructed_pipeline.instruction_aware_rerank"
        ) as mock_iar:
            execute_instructed_pipeline(
                run_search=run_search,
                query="q",
                base_filters={},
                instructed_config=InstructedRetrieverModel(
                    columns=columns, rerank=instr_rerank
                ),
                router_config=rc,
                verifier_config=None,
                decomposition_config=None,
                instruction_rerank_config=instr_rerank,
                instructed_columns=columns,
            )
        mock_iar.assert_not_called()

    def test_called_for_instructed_mode(
        self, columns: list[ColumnInfo], llm: InferenceEndpointModel
    ) -> None:
        instr_rerank = InstructionAwareRerankModel(model=llm)

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            return _docs("a")

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query", return_value=[]),
            patch(
                "dao_ai.tools.instructed_pipeline.instruction_aware_rerank",
                return_value=_docs("rerank_a"),
            ) as mock_iar,
        ):
            mock_llm.return_value = MagicMock()
            docs = execute_instructed_pipeline(
                run_search=run_search,
                query="q",
                base_filters={},
                instructed_config=InstructedRetrieverModel(
                    columns=columns,
                    decomposition=DecompositionModel(model=llm),
                    rerank=instr_rerank,
                ),
                router_config=None,
                verifier_config=None,
                decomposition_config=DecompositionModel(model=llm),
                instruction_rerank_config=instr_rerank,
                instructed_columns=columns,
            )
        mock_iar.assert_called_once()
        assert docs[0].metadata["id"] == "rerank_a"


class TestVerifierRetryLoop:
    def _base(
        self,
        columns: list[ColumnInfo],
        instructed_cfg: InstructedRetrieverModel,
        run_search,
    ) -> dict[str, Any]:
        return dict(
            run_search=run_search,
            query="q",
            base_filters={},
            instructed_config=instructed_cfg,
            router_config=None,
            decomposition_config=instructed_cfg.decomposition,
            instruction_rerank_config=None,
            instructed_columns=columns,
        )

    def test_passed_stops_loop(
        self,
        columns: list[ColumnInfo],
        instructed_cfg: InstructedRetrieverModel,
        llm: InferenceEndpointModel,
    ) -> None:
        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            return _docs("a")

        verifier = VerifierModel(model=llm, on_failure="retry", max_retries=3)
        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query", return_value=[]),
            patch("dao_ai.tools.instructed_pipeline.verify_results") as mock_verify,
        ):
            mock_llm.return_value = MagicMock()
            mock_verify.return_value = VerificationResult(passed=True, confidence=0.9)
            execute_instructed_pipeline(
                **self._base(columns, instructed_cfg, run_search),
                verifier_config=verifier,
            )
        mock_verify.assert_called_once()

    def test_warn_annotates_and_stops(
        self,
        columns: list[ColumnInfo],
        instructed_cfg: InstructedRetrieverModel,
        llm: InferenceEndpointModel,
    ) -> None:
        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            return _docs("a")

        verifier = VerifierModel(model=llm, on_failure="warn", max_retries=3)
        vr = VerificationResult(passed=False, confidence=0.3, feedback="too broad")
        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query", return_value=[]),
            patch(
                "dao_ai.tools.instructed_pipeline.verify_results", return_value=vr
            ) as mock_verify,
            patch(
                "dao_ai.tools.instructed_pipeline.add_verification_metadata"
            ) as mock_annot,
        ):
            mock_llm.return_value = MagicMock()
            mock_annot.return_value = _docs("annotated_a")
            docs = execute_instructed_pipeline(
                **self._base(columns, instructed_cfg, run_search),
                verifier_config=verifier,
            )
        assert mock_verify.call_count == 1
        mock_annot.assert_called_once()
        assert docs[0].metadata["id"] == "annotated_a"

    def test_retry_re_executes_with_feedback(
        self,
        columns: list[ColumnInfo],
        instructed_cfg: InstructedRetrieverModel,
        llm: InferenceEndpointModel,
    ) -> None:
        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            return _docs("a")

        verifier = VerifierModel(model=llm, on_failure="retry", max_retries=3)
        results = [
            VerificationResult(
                passed=False, confidence=0.3, feedback="add brand filter"
            ),
            VerificationResult(passed=True, confidence=0.9),
        ]

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query") as mock_decompose,
            patch(
                "dao_ai.tools.instructed_pipeline.verify_results", side_effect=results
            ) as mock_verify,
        ):
            mock_llm.return_value = MagicMock()
            mock_decompose.return_value = [SearchQuery(text="sub1", filters=None)]
            execute_instructed_pipeline(
                **self._base(columns, instructed_cfg, run_search),
                verifier_config=verifier,
            )

        assert mock_verify.call_count == 2
        second_kwargs = mock_decompose.call_args_list[1].kwargs
        assert second_kwargs.get("previous_feedback") == "add brand filter"

    def test_retries_exhausted_annotates(
        self,
        columns: list[ColumnInfo],
        instructed_cfg: InstructedRetrieverModel,
        llm: InferenceEndpointModel,
    ) -> None:
        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            return _docs("a")

        verifier = VerifierModel(model=llm, on_failure="retry", max_retries=1)
        vr = VerificationResult(passed=False, confidence=0.2, feedback="still bad")

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch("dao_ai.tools.instructed_pipeline.decompose_query") as mock_decompose,
            patch(
                "dao_ai.tools.instructed_pipeline.verify_results", return_value=vr
            ) as mock_verify,
            patch(
                "dao_ai.tools.instructed_pipeline.add_verification_metadata"
            ) as mock_annot,
        ):
            mock_llm.return_value = MagicMock()
            mock_decompose.return_value = [SearchQuery(text="sub1", filters=None)]
            mock_annot.return_value = _docs("exhausted_a")
            docs = execute_instructed_pipeline(
                **self._base(columns, instructed_cfg, run_search),
                verifier_config=verifier,
            )
        assert mock_verify.call_count == 2
        assert mock_annot.call_args.kwargs.get("exhausted") is True
        assert docs[0].metadata["id"] == "exhausted_a"

    def test_standard_mode_does_not_retry(
        self,
        columns: list[ColumnInfo],
        instructed_cfg: InstructedRetrieverModel,
        llm: InferenceEndpointModel,
    ) -> None:
        rc = RouterModel(default_mode="standard", auto_bypass=False)

        def run_search(q: str, f: dict[str, Any]) -> list[Document]:
            return _docs("a")

        verifier = VerifierModel(model=llm, on_failure="retry", max_retries=3)
        vr = VerificationResult(passed=False, confidence=0.4)

        with (
            patch("dao_ai.tools.instructed_pipeline._get_cached_llm") as mock_llm,
            patch(
                "dao_ai.tools.instructed_pipeline.verify_results", return_value=vr
            ) as mock_verify,
            patch(
                "dao_ai.tools.instructed_pipeline.add_verification_metadata"
            ) as mock_annot,
        ):
            mock_llm.return_value = MagicMock()
            mock_annot.return_value = _docs("annotated")
            execute_instructed_pipeline(
                run_search=run_search,
                query="q",
                base_filters={},
                instructed_config=instructed_cfg,
                router_config=rc,
                verifier_config=verifier,
                decomposition_config=instructed_cfg.decomposition,
                instruction_rerank_config=None,
                instructed_columns=columns,
            )
        assert mock_verify.call_count == 1
        mock_annot.assert_called_once()

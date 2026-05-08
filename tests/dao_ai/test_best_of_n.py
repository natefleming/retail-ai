"""Unit tests for BestOfNChatModel + supporting types."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from dao_ai.best_of_n import (
    BestOfNChatModel,
    CandidateScore,
    JudgeDecision,
)


def _make_judge_decision(scores: list[float], reasoning: str = "test") -> JudgeDecision:
    """Helper: build a JudgeDecision with the given per-index scores."""
    return JudgeDecision(
        scores=[
            CandidateScore(index=i, score=s, rationale=f"rationale {i}")
            for i, s in enumerate(scores)
        ],
        reasoning=reasoning,
    )


def _make_generator(candidate_texts: list[str]) -> MagicMock:
    """Build a generator mock that returns candidate_texts[i] on the i-th call.

    The mock supports both `.invoke(...)` (sync) and `.ainvoke(...)` (async),
    each producing the next AIMessage in the sequence.
    """
    generator = MagicMock()
    counter = {"i": 0}

    def _next_message(*_args, **_kwargs):
        text = candidate_texts[counter["i"] % len(candidate_texts)]
        counter["i"] += 1
        return AIMessage(content=text)

    async def _next_message_async(*_args, **_kwargs):
        return _next_message()

    generator.invoke.side_effect = _next_message
    generator.ainvoke.side_effect = _next_message_async
    return generator


def _make_judge_runnable(decision: JudgeDecision | Exception) -> MagicMock:
    """Build a judge_runnable mock that returns `decision` on .invoke()."""
    runnable = MagicMock()
    if isinstance(decision, Exception):
        runnable.invoke.side_effect = decision
    else:
        runnable.invoke.return_value = decision
    return runnable


@pytest.mark.unit
class TestJudgeDecisionShape:
    """Sanity tests on the structured-output Pydantic types."""

    def test_judge_decision_round_trip(self) -> None:
        decision = _make_judge_decision([1.0, 5.0, 9.0])
        assert decision.scores[2].score == 9.0
        assert decision.scores[2].index == 2
        assert decision.reasoning == "test"

    def test_candidate_score_validation(self) -> None:
        score = CandidateScore(index=0, score=7.5, rationale="ok")
        assert score.score == 7.5


@pytest.mark.unit
class TestBestOfNSelection:
    """Verify the wrapper picks the argmax candidate from judge scores."""

    def test_returns_argmax_candidate(self) -> None:
        candidate_texts = [f"answer-{i}" for i in range(4)]
        generator = _make_generator(candidate_texts)
        # Index 2 has the highest score → wrapper should return "answer-2".
        judge = _make_judge_runnable(_make_judge_decision([3.0, 5.0, 9.0, 7.0]))

        wrapper = BestOfNChatModel(generator=generator, judge_runnable=judge, n=4)
        result = wrapper._generate([HumanMessage(content="prompt")])

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "answer-2"
        assert generator.invoke.call_count == 4
        judge.invoke.assert_called_once()

    def test_argmax_breaks_ties_on_lowest_index(self) -> None:
        """If two candidates tie for the highest score, the lower index wins."""
        generator = _make_generator(["a", "b", "c"])
        judge = _make_judge_runnable(_make_judge_decision([8.0, 8.0, 8.0]))

        wrapper = BestOfNChatModel(generator=generator, judge_runnable=judge, n=3)
        result = wrapper._generate([HumanMessage(content="prompt")])

        assert result.generations[0].message.content == "a"


@pytest.mark.unit
class TestBestOfNJudgeFailure:
    """When the judge call raises, fall back to candidate 0 with a warning."""

    def test_judge_exception_returns_candidate_zero(self) -> None:
        generator = _make_generator(["first", "second", "third"])
        judge = _make_judge_runnable(RuntimeError("judge boom"))

        wrapper = BestOfNChatModel(generator=generator, judge_runnable=judge, n=3)
        result = wrapper._generate([HumanMessage(content="prompt")])

        assert result.generations[0].message.content == "first"
        # Generator was still called N times even though judge failed.
        assert generator.invoke.call_count == 3

    def test_empty_scores_returns_candidate_zero(self) -> None:
        generator = _make_generator(["first", "second"])
        # Judge returns a decision with no scores at all.
        decision = JudgeDecision(scores=[], reasoning="nothing to compare")
        judge = _make_judge_runnable(decision)

        wrapper = BestOfNChatModel(generator=generator, judge_runnable=judge, n=2)
        result = wrapper._generate([HumanMessage(content="prompt")])

        assert result.generations[0].message.content == "first"


@pytest.mark.unit
class TestBestOfNShortCircuit:
    """n=1 must skip the judge entirely — no parallelism, no judge call."""

    def test_n_one_does_not_call_judge(self) -> None:
        generator = _make_generator(["only-one"])
        judge = _make_judge_runnable(_make_judge_decision([10.0]))

        wrapper = BestOfNChatModel(generator=generator, judge_runnable=judge, n=1)
        result = wrapper._generate([HumanMessage(content="prompt")])

        assert result.generations[0].message.content == "only-one"
        assert generator.invoke.call_count == 1
        judge.invoke.assert_not_called()


@pytest.mark.unit
class TestBestOfNTemperatureFloor:
    """from_components must apply max(generator_temp, 0.7) unless overridden."""

    def test_low_generator_temp_is_raised_to_floor(self) -> None:
        generator = MagicMock()
        # `.bind(...)` returns a new (mocked) generator. We assert what kwargs
        # were passed in.
        generator.bind = MagicMock(return_value=generator)
        judge = MagicMock()
        judge.with_structured_output = MagicMock(return_value=MagicMock())

        BestOfNChatModel.from_components(
            generator=generator,
            judge=judge,
            n=4,
            generator_temperature=0.1,
            temperature_override=None,
        )

        generator.bind.assert_called_once_with(temperature=0.7)

    def test_high_generator_temp_is_preserved(self) -> None:
        generator = MagicMock()
        generator.bind = MagicMock(return_value=generator)
        judge = MagicMock()
        judge.with_structured_output = MagicMock(return_value=MagicMock())

        BestOfNChatModel.from_components(
            generator=generator,
            judge=judge,
            n=4,
            generator_temperature=0.9,
            temperature_override=None,
        )

        generator.bind.assert_called_once_with(temperature=0.9)

    def test_explicit_override_wins(self) -> None:
        generator = MagicMock()
        generator.bind = MagicMock(return_value=generator)
        judge = MagicMock()
        judge.with_structured_output = MagicMock(return_value=MagicMock())

        BestOfNChatModel.from_components(
            generator=generator,
            judge=judge,
            n=4,
            generator_temperature=0.0,
            temperature_override=1.0,
        )

        generator.bind.assert_called_once_with(temperature=1.0)


@pytest.mark.unit
class TestBestOfNAsync:
    """The async path must run candidates via asyncio.gather and pick argmax."""

    def test_async_returns_argmax_candidate(self) -> None:
        # The repo doesn't use pytest-asyncio, so we drive the coroutine
        # via asyncio.run rather than `async def test_...`.
        generator = _make_generator([f"a-{i}" for i in range(3)])
        judge = _make_judge_runnable(_make_judge_decision([1.0, 9.0, 5.0]))

        wrapper = BestOfNChatModel(generator=generator, judge_runnable=judge, n=3)

        async def _run() -> AIMessage:
            result = await wrapper._agenerate([HumanMessage(content="prompt")])
            return result.generations[0].message

        winning_message = asyncio.run(_run())

        assert winning_message.content == "a-1"
        assert generator.ainvoke.call_count == 3


@pytest.mark.unit
class TestBestOfNConfigValidator:
    """The Pydantic config rejects out-of-range n and accepts the canonical form."""

    def test_n_too_large_rejected(self) -> None:
        from dao_ai.config import BestOfNConfig

        with pytest.raises(ValueError, match=r"best_of_n\.n must be in"):
            BestOfNConfig(n=17, judge="my-endpoint")

    def test_n_too_small_rejected(self) -> None:
        from dao_ai.config import BestOfNConfig

        with pytest.raises(ValueError, match=r"best_of_n\.n must be in"):
            BestOfNConfig(n=0, judge="my-endpoint")

    def test_default_n_is_eight(self) -> None:
        from dao_ai.config import BestOfNConfig

        cfg = BestOfNConfig(judge="my-endpoint")
        assert cfg.n == 8
        assert cfg.temperature_override is None

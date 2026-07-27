"""Tests for AuditReceiptMiddleware factory + pure helpers.

Runtime middleware invocation (awrap_tool_call) is exercised via the
integration path in tests/dao_ai/audit/test_hitl_receipt_emission.py once
the LangGraph plumbing is in place. Here we cover the parts that don't
need a full LangGraph runtime.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any

from dao_ai.config import (
    AuditModel,
    DatabaseModel,
    HumanInTheLoopModel,
    PythonFunctionModel,
    ToolModel,
)
from dao_ai.middleware.audit_receipt import (
    AuditReceiptMiddleware,
    _extract_email,
    _extract_obo_evidence,
    create_audit_middleware_from_tool_models,
)


def _make_python_tool(name: str, audit: AuditModel | None = None) -> ToolModel:
    return ToolModel(
        name=name,
        function=PythonFunctionModel(
            name=f"tests.fixtures.tools.{name}",
            audit=audit,
        ),
    )


class TestCreateAuditMiddlewareFactory:
    def test_no_audit_returns_none(self) -> None:
        """Factory must return None when no tool has audit — the disabled path."""
        tool_models: list[ToolModel] = [_make_python_tool("plain_tool")]
        result = create_audit_middleware_from_tool_models(tool_models)
        assert result is None

    def test_audit_present_returns_middleware(self, monkeypatch: Any) -> None:
        """Factory returns a middleware when at least one tool has audit."""
        # Avoid touching the real Lakebase pool during middleware construction.
        from dao_ai.audit import manager as _mgr

        class _FakeSink:
            def __init__(self, cfg: AuditModel) -> None:
                self.cfg = cfg

        monkeypatch.setattr(
            _mgr.AuditSinkManager,
            "for_config",
            classmethod(lambda cls, c: _FakeSink(c)),  # type: ignore[arg-type]
        )
        # Fake resolves to include audited + non-audited tools.
        audit_cfg = AuditModel(database=DatabaseModel(project="audit-lake"))
        tool_models: list[ToolModel] = [
            _make_python_tool("plain_tool"),
            _make_python_tool("sensitive_tool", audit=audit_cfg),
        ]

        # PythonFunctionModel.as_tools imports the referenced module. We stub
        # it so no import happens.
        import dao_ai.config as _cfg

        monkeypatch.setattr(
            _cfg.PythonFunctionModel,
            "as_tools",
            lambda self, **_: [_FakeTool(self.name.rsplit(".", 1)[-1])],
        )

        result = create_audit_middleware_from_tool_models(tool_models)
        assert isinstance(result, AuditReceiptMiddleware)
        assert "sensitive_tool" in list(result.audited_tools)
        assert "plain_tool" not in list(result.audited_tools)

    def test_audit_with_hitl_still_registered(self, monkeypatch: Any) -> None:
        """A tool with both audit and human_in_the_loop shows up in audited set."""
        from dao_ai.audit import manager as _mgr

        class _FakeSink:
            def __init__(self, cfg: AuditModel) -> None:
                self.cfg = cfg

        monkeypatch.setattr(
            _mgr.AuditSinkManager,
            "for_config",
            classmethod(lambda cls, c: _FakeSink(c)),  # type: ignore[arg-type]
        )
        import dao_ai.config as _cfg

        monkeypatch.setattr(
            _cfg.PythonFunctionModel,
            "as_tools",
            lambda self, **_: [_FakeTool(self.name.rsplit(".", 1)[-1])],
        )

        audit_cfg = AuditModel(database=DatabaseModel(project="audit-lake"))
        hitl_cfg = HumanInTheLoopModel(review_prompt="Approve?")
        tool_models: list[ToolModel] = [
            ToolModel(
                name="refund",
                function=PythonFunctionModel(
                    name="tests.fixtures.tools.refund",
                    audit=audit_cfg,
                    human_in_the_loop=hitl_cfg,
                ),
            ),
        ]
        result = create_audit_middleware_from_tool_models(tool_models)
        assert result is not None
        assert list(result.audited_tools) == ["refund"]


class TestOboExtraction:
    def _make_jwt(self, sub: str = "user-1", exp_epoch: int = 1_800_000_000) -> str:
        header: str = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload_dict: dict[str, Any] = {"sub": sub, "exp": exp_epoch}
        payload: str = (
            base64.urlsafe_b64encode(json.dumps(payload_dict).encode())
            .rstrip(b"=")
            .decode()
        )
        signature: str = (
            base64.urlsafe_b64encode(b"fake-signature").rstrip(b"=").decode()
        )
        return f"{header}.{payload}.{signature}"

    def test_extract_evidence_populated_when_header_present(self) -> None:
        token: str = self._make_jwt(sub="alice@example.com", exp_epoch=1_800_000_000)
        headers: dict[str, Any] = {"X-Forwarded-Access-Token": token}
        raw, exp, sub = _extract_obo_evidence(headers)
        assert raw == token
        assert isinstance(exp, datetime)
        assert exp.tzinfo == timezone.utc
        assert sub == "alice@example.com"

    def test_extract_evidence_missing_header(self) -> None:
        raw, exp, sub = _extract_obo_evidence({})
        assert raw is None and exp is None and sub is None

    def test_extract_evidence_none_headers(self) -> None:
        raw, exp, sub = _extract_obo_evidence(None)
        assert raw is None and exp is None and sub is None

    def test_extract_evidence_lowercase_header_key(self) -> None:
        token: str = self._make_jwt()
        raw, _exp, _sub = _extract_obo_evidence({"x-forwarded-access-token": token})
        assert raw == token

    def test_extract_evidence_malformed_jwt(self) -> None:
        """Non-JWT string is retained verbatim but no claims are extracted."""
        raw, exp, sub = _extract_obo_evidence({"X-Forwarded-Access-Token": "not-a-jwt"})
        # Even a malformed token is preserved verbatim — never fabricate,
        # but never drop the caller-provided value either.
        assert raw == "not-a-jwt"
        assert exp is None and sub is None


class TestEmailExtraction:
    def test_extract_email_case_insensitive(self) -> None:
        assert _extract_email({"X-Forwarded-Email": "a@b"}) == "a@b"
        assert _extract_email({"x-forwarded-email": "c@d"}) == "c@d"

    def test_extract_email_none(self) -> None:
        assert _extract_email(None) is None
        assert _extract_email({}) is None


# ------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------


class _FakeTool:
    """Duck-types the ``name`` attribute the factory reads via getattr."""

    def __init__(self, name: str) -> None:
        self.name: str = name

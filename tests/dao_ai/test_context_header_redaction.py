"""Tests that the caller's OBO bearer token never travels back out.

``dao_ai.apps.handlers`` injects the whole inbound header map onto the
``Context`` so tools can act as the calling user. That map contains a live
bearer in ``x-forwarded-access-token``, so anything that serializes the context
back toward the caller — ``custom_outputs``, or the "copy-paste this config"
templates in the validation middleware — is a credential-disclosure path. The
middleware templates are the worse of the two: they land in assistant message
text, and from there in the MLflow trace and any saved transcript.

These tests pin both directions: nothing credential-shaped goes out, and the
headers still arrive inbound (so the leak is never "fixed" by dropping the
field that OBO depends on).
"""

import json
from unittest.mock import MagicMock

import mlflow
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from dao_ai._tracing.redaction import (
    REDACTED,
    install_trace_redaction,
    redact_credentials,
)
from dao_ai.middleware.message_validation import (
    KEEP_PROVIDED_VALUE,
    CustomFieldValidationMiddleware,
    RequiredField,
    ThreadIdValidationMiddleware,
    UserIdValidationMiddleware,
)
from dao_ai.models import LanggraphResponsesAgent
from dao_ai.state import AgentState, Context, context_configurable_fields

# Obviously-fake so a scanner never has to decide whether this is real.
_FAKE_OBO_TOKEN = "FAKE-OBO-TOKEN-not-a-real-credential"
_FAKE_HEADERS = {
    "x-forwarded-access-token": _FAKE_OBO_TOKEN,
    "x-forwarded-user": "someone@example.com",
    "content-type": "application/json",
}


def create_mock_runtime(context: Context) -> MagicMock:
    """Create a mock runtime with the given context."""
    runtime = MagicMock()
    runtime.context = context
    return runtime


def build_mock_graph() -> MagicMock:
    """A CompiledStateGraph stand-in that answers one turn and holds no state."""
    mock_graph = MagicMock(spec=CompiledStateGraph)
    mock_graph.checkpointer = MagicMock()

    async def mock_ainvoke(*args, **kwargs):
        return {
            "messages": [
                HumanMessage(content="Test question"),
                AIMessage(content="Test response"),
            ]
        }

    async def mock_aget_state(*args, **kwargs):
        snapshot = MagicMock()
        snapshot.values = {}
        return snapshot

    mock_graph.ainvoke = mock_ainvoke
    mock_graph.aget_state = mock_aget_state
    return mock_graph


def json_block(error_message: str) -> dict:
    """Parse the fenced JSON block out of a middleware error message."""
    start = error_message.find("```json") + 7
    end = error_message.find("```", start)
    return json.loads(error_message[start:end].strip())


@pytest.fixture
def tracing_enabled(monkeypatch, tmp_path):
    """Tracing on, pointed at a per-test file store, with span redaction installed.

    ``tests/conftest.py`` disables tracing suite-wide; this undoes that for one
    test. The processor list is global state in MLflow, so it is cleared on the
    way out rather than left registered for whatever runs next.
    """
    monkeypatch.setenv("MLFLOW_TRACE_SAMPLING_RATIO", "1")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    # sqlite, not a `file://` store: MLflow 3.14 refuses the filesystem backend
    # unless MLFLOW_ALLOW_FILE_STORE is set.
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-span-credential-redaction")
    mlflow.tracing.enable()
    install_trace_redaction()
    try:
        yield
    finally:
        mlflow.tracing.configure(span_processors=[])
        mlflow.tracing.disable()


@pytest.mark.unit
class TestContextConfigurableFields:
    """The shared filter every echo site goes through."""

    def test_drops_headers(self) -> None:
        context = Context(user_id="u", thread_id="t", headers=_FAKE_HEADERS)

        assert context_configurable_fields(context) == {}

    def test_keeps_benign_extras(self) -> None:
        context = Context(
            user_id="u", thread_id="t", headers=_FAKE_HEADERS, store_num="87887"
        )

        assert context_configurable_fields(context) == {"store_num": "87887"}

    @pytest.mark.parametrize(
        "secret_field",
        [
            "authorization",
            "x_forwarded_access_token",
            "api_key",
            "session_cookie",
            "db_password",
            "client_secret",
        ],
    )
    def test_drops_secret_shaped_extras(self, secret_field: str) -> None:
        """``Context`` is ``extra="allow"``, so a caller can smuggle a credential
        in as an arbitrary field. Name-shaped filtering catches those too."""
        context = Context(
            user_id="u",
            thread_id="t",
            store_num="87887",
            **{secret_field: _FAKE_OBO_TOKEN},
        )

        fields = context_configurable_fields(context)

        assert secret_field not in fields
        assert _FAKE_OBO_TOKEN not in json.dumps(fields)
        # Still a filter, not a blanket wipe.
        assert fields == {"store_num": "87887"}

    def test_databricks_auth_type_is_not_treated_as_a_secret(self) -> None:
        """Guard the deliberate gap in the key pattern: ``AUTHORIZATION`` must
        not start matching ``DATABRICKS_AUTH_TYPE``, which the diagnostics probe
        surfaces verbatim on purpose."""
        context = Context(user_id="u", thread_id="t", databricks_auth_type="pat")

        assert context_configurable_fields(context) == {"databricks_auth_type": "pat"}

    @pytest.mark.parametrize(
        "benign_field",
        [
            # SESSION as a substring — a session *id* is metadata the caller has
            # to see echoed; a session credential is caught by token/cookie.
            "session_id",
            # KEY as a substring, and the reason segment matching exists.
            "monkey_wrench",
            # TOKEN/SECRET as substrings of ordinary domain words.
            "tokenization_mode",
            "secretary_id",
        ],
    )
    def test_keeps_extras_that_merely_contain_a_secret_word(
        self, benign_field: str
    ) -> None:
        """The filter matches whole words, not substrings. Over-matching here is
        not a safe default: the caller reads its own fields back out of this
        block, so a swallowed field is a broken round-trip."""
        context = Context(user_id="u", thread_id="t", **{benign_field: "plain-value"})

        assert context_configurable_fields(context) == {benign_field: "plain-value"}

    @pytest.mark.parametrize(
        "secret_field",
        [
            "api_key",
            "apiKey",
            "apikey",
            "access_token",
            "accessToken",
            "x-forwarded-access-token",
            "Authorization",
            "refresh_token",
        ],
    )
    def test_still_drops_credential_names_in_every_casing(
        self, secret_field: str
    ) -> None:
        """Precision must not cost recall on the shapes that matter: separators,
        camelCase and bare compounds all resolve to the same segments."""
        context = Context(user_id="u", thread_id="t", **{secret_field: _FAKE_OBO_TOKEN})

        fields = context_configurable_fields(context)

        assert secret_field not in fields
        assert _FAKE_OBO_TOKEN not in json.dumps(fields)


@pytest.mark.unit
class TestCustomOutputsOmitHeaders:
    """The side-channel path: custom_outputs on a successful response."""

    def _request(self) -> ResponsesAgentRequest:
        from mlflow.types.responses import Message

        return ResponsesAgentRequest(
            input=[Message(role="user", content="Test question")],
            custom_inputs={
                "configurable": {
                    "thread_id": "t-1",
                    "user_id": "test_user",
                    "store_num": "87887",
                    "headers": _FAKE_HEADERS,
                }
            },
        )

    def test_predict_omits_headers(self) -> None:
        agent = LanggraphResponsesAgent(build_mock_graph())

        response = agent.predict(self._request())

        assert isinstance(response, ResponsesAgentResponse)
        blob = json.dumps(response.custom_outputs)
        assert _FAKE_OBO_TOKEN not in blob
        assert "x-forwarded-access-token" not in blob
        assert "headers" not in response.custom_outputs["configurable"]
        # Filtered, not blanked — the caller's own field still round-trips.
        assert response.custom_outputs["configurable"]["store_num"] == "87887"

    def test_predict_stream_omits_headers(self) -> None:
        """The streaming path builds its own custom_outputs; if it ever diverges
        from ``predict``, this is what catches it."""
        agent = LanggraphResponsesAgent(build_mock_graph())

        blob = "".join(
            json.dumps(chunk.custom_outputs)
            for chunk in agent.predict_stream(self._request())
            if chunk.custom_outputs
        )

        assert blob, "expected at least one chunk to carry custom_outputs"
        assert _FAKE_OBO_TOKEN not in blob
        assert "x-forwarded-access-token" not in blob


@pytest.mark.unit
class TestMiddlewareTemplatesOmitHeaders:
    """The worse path: the token spliced into user-facing error text."""

    @pytest.mark.parametrize(
        "middleware,context",
        [
            # user_id missing
            (
                UserIdValidationMiddleware(),
                Context(user_id=None, thread_id="t-1", headers=_FAKE_HEADERS),
            ),
            # user_id contains a dot
            (
                UserIdValidationMiddleware(),
                Context(user_id="first.last", thread_id="t-1", headers=_FAKE_HEADERS),
            ),
            # thread_id missing
            (
                ThreadIdValidationMiddleware(),
                Context(user_id="u", thread_id=None, headers=_FAKE_HEADERS),
            ),
            # required custom field missing
            (
                CustomFieldValidationMiddleware(
                    [RequiredField(name="store_num", description="Store number")]
                ),
                Context(user_id="u", thread_id="t-1", headers=_FAKE_HEADERS),
            ),
        ],
        ids=["missing_user_id", "dotted_user_id", "missing_thread_id", "custom_field"],
    )
    def test_error_text_omits_token(self, middleware: object, context: Context) -> None:
        state: AgentState = {"messages": [HumanMessage(content="hi")]}

        with pytest.raises(ValueError) as exc_info:
            middleware.validate(state, create_mock_runtime(context))

        error = str(exc_info.value)
        assert _FAKE_OBO_TOKEN not in error
        assert "x-forwarded-access-token" not in error
        assert "headers" not in json_block(error)["configurable"]


@pytest.mark.unit
class TestHeadersStillReachContext:
    """Inbound guard. Headers are load-bearing for OBO and for the user_id
    fallback — the fix must not be "delete the field"."""

    def test_headers_survive_request_conversion(self) -> None:
        from mlflow.types.responses import Message

        agent = LanggraphResponsesAgent(build_mock_graph())
        request = ResponsesAgentRequest(
            input=[Message(role="user", content="hi")],
            custom_inputs={
                "configurable": {"thread_id": "t-1", "headers": _FAKE_HEADERS}
            },
        )

        context = agent._convert_request_to_context(request)

        assert context.headers is not None
        assert context.headers["x-forwarded-access-token"] == _FAKE_OBO_TOKEN
        # user_id falls back to x-forwarded-user, dots normalized for the
        # memory namespace.
        assert context.user_id == "someone@example_com"


@pytest.mark.unit
class TestFilteredFieldIsNotReportedAsMissing:
    """A required field the caller *did* send, whose name is credential-shaped.

    The filter removes it from the echoed block, so the copy-paste template must
    not fall through to ``example_value`` — that would tell the caller to paste a
    placeholder over a key they already sent correctly, and the request would then
    fail downstream with a wrong credential instead of a missing one.
    """

    @staticmethod
    def _middleware() -> CustomFieldValidationMiddleware:
        """Mirrors ``examples/12_middleware/custom_field_validation.yaml``."""
        return CustomFieldValidationMiddleware(
            [
                RequiredField(
                    name="external_api_key",
                    description="API key for the external service",
                    example_value="sk-proj-xxxxxxxxxxxxx",
                ),
                RequiredField(
                    name="store_num",
                    description="Store number",
                    example_value="87887",
                ),
            ]
        )

    def test_provided_secret_field_is_marked_keep_not_replaced(self) -> None:
        context = Context(
            user_id="u",
            thread_id="t-1",
            external_api_key=_FAKE_OBO_TOKEN,
            headers=_FAKE_HEADERS,
        )
        state: AgentState = {"messages": [HumanMessage(content="hi")]}

        with pytest.raises(ValueError) as exc_info:
            self._middleware().validate(state, create_mock_runtime(context))

        error = str(exc_info.value)
        configurable = json_block(error)["configurable"]

        # The value the caller sent never comes back out.
        assert _FAKE_OBO_TOKEN not in error
        # ...and it is not overwritten with the example, which is the bug.
        assert configurable["external_api_key"] == KEEP_PROVIDED_VALUE
        assert "sk-proj-xxxxxxxxxxxxx" not in error
        # Only the genuinely absent field is reported missing.
        assert "**store_num**" in error
        assert "The following required fields are missing: **store_num**" in error
        assert configurable["store_num"] == "87887"

    def test_absent_secret_field_still_gets_the_example(self) -> None:
        """The keep-marker is for *provided* fields only — a caller who sent
        nothing still needs to be shown the shape of the value."""
        context = Context(user_id="u", thread_id="t-1", headers=_FAKE_HEADERS)
        state: AgentState = {"messages": [HumanMessage(content="hi")]}

        with pytest.raises(ValueError) as exc_info:
            self._middleware().validate(state, create_mock_runtime(context))

        error = str(exc_info.value)
        configurable = json_block(error)["configurable"]

        assert configurable["external_api_key"] == "sk-proj-xxxxxxxxxxxxx"
        assert KEEP_PROVIDED_VALUE not in error
        assert "**external_api_key**" in error


@pytest.mark.unit
class TestSpanRedaction:
    """The trace sink. Filtering the outbound paths does nothing for it: the
    bearer is on the request *inbound*, and ``apredict`` is ``@mlflow.trace``-d,
    so the span serializes it as an input attribute."""

    def test_redacts_nested_request_payload(self) -> None:
        payload = {
            "request": {
                "input": [{"role": "user", "content": "what is my discount?"}],
                "custom_inputs": {
                    "configurable": {
                        "thread_id": "t-1",
                        "store_num": "87887",
                        "headers": dict(_FAKE_HEADERS),
                    }
                },
            }
        }

        redacted = redact_credentials(payload)

        headers = redacted["request"]["custom_inputs"]["configurable"]["headers"]
        assert headers["x-forwarded-access-token"] == REDACTED
        # Only the credential-shaped header goes; the rest of the map is useful
        # in a trace and stays.
        assert headers["x-forwarded-user"] == "someone@example.com"
        assert headers["content-type"] == "application/json"
        configurable = redacted["request"]["custom_inputs"]["configurable"]
        assert configurable["store_num"] == "87887"
        assert configurable["thread_id"] == "t-1"
        # The user's own question is untouched — this filter reads key names only.
        assert redacted["request"]["input"][0]["content"] == "what is my discount?"

    def test_leaves_payloads_without_credentials_alone(self) -> None:
        payload = {"question": "hi", "session_id": "s-1", "rows": [1, 2, 3]}

        assert redact_credentials(payload) == payload

    def test_survives_a_self_referential_payload(self) -> None:
        """Depth-capped, so a cycle returns rather than recursing forever."""
        payload: dict[str, object] = {"api_key": _FAKE_OBO_TOKEN}
        payload["self"] = payload

        redacted = redact_credentials(payload)

        assert redacted["api_key"] == REDACTED

    def test_exported_trace_has_no_token(self, tracing_enabled) -> None:
        """End to end through MLflow: register the processor, run a traced
        function whose argument carries the header map, read the trace back."""

        @mlflow.trace(name="apredict_stand_in")
        def traced(request: dict) -> dict:
            return {"answer": "42", "echoed_headers": request["headers"]}

        traced({"headers": dict(_FAKE_HEADERS), "store_num": "87887"})

        trace_id = mlflow.get_last_active_trace_id()
        trace = mlflow.get_trace(trace_id)
        assert trace is not None, f"trace {trace_id} not found"

        serialized = trace.to_json()
        assert _FAKE_OBO_TOKEN not in serialized
        assert REDACTED in serialized
        # The span is redacted, not dropped or emptied.
        assert "87887" in serialized
        assert "someone@example.com" in serialized

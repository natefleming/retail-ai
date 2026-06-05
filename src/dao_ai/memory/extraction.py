"""
Background memory extraction for DAO AI agents.

This module wraps langmem's ``create_memory_store_manager`` and
``ReflectionExecutor`` to provide automatic memory extraction that runs
after each conversation turn without adding latency to responses.

Two integration modes are supported:

- **Inline extraction** -- ``MemoryStoreManager.ainvoke()`` is called
  directly in the agent pipeline. Simple but adds latency.
- **Background extraction** -- ``ReflectionExecutor.submit()`` schedules
  extraction in a background thread with debouncing.

Usage::

    from dao_ai.memory.extraction import create_extraction_manager

    manager, executor = create_extraction_manager(
        model="databricks-claude-sonnet-4",
        store=store,
        namespace=("memory", "{user_id}"),
        schemas=["user_profile", "preference"],
    )

    # Background extraction after agent response
    executor.submit(
        {"messages": conversation_messages},
        config={"configurable": {"user_id": "alice"}},
        after_seconds=0,
    )

    # Search memories
    results = await manager.asearch(
        query="user preferences",
        config={"configurable": {"user_id": "alice"}},
    )
"""

from __future__ import annotations

import contextvars
import threading
from typing import TYPE_CHECKING, Any, Union

from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.runnables import RunnableWithFallbacks
from langgraph.store.base import BaseStore
from langmem import ReflectionExecutor, create_memory_store_manager
from loguru import logger

from dao_ai.memory.schemas import resolve_schemas

if TYPE_CHECKING:
    from langmem.knowledge.extraction import MemoryStoreManager
    from langmem.reflection import LocalReflectionExecutor
    from pydantic import BaseModel


class _ContextAwareReflector:
    """Proxy reflector that runs ``invoke()`` inside a per-payload captured context.

    ``langmem.LocalReflectionExecutor.submit(payload, ...)`` is data-based:
    the worker thread later calls ``self._reflector.invoke(task.payload)``
    with no propagation of the caller's ``contextvars.Context``. Because
    MLflow's active-span association is stored in a ``ContextVar``, the
    inner ``query_gen.invoke(...)`` call sees no parent span and opens a
    new root trace.

    This proxy sits between ``LocalReflectionExecutor`` and the real
    ``MemoryStoreManager``. ``LazyReflectionExecutor.submit`` captures the
    caller's ``Context`` at submit time and ``attach(payload, ctx)`` stores
    it keyed by ``id(payload)``. When the langmem worker thread eventually
    calls our ``invoke(payload)``, we pop that context and run the inner
    ``invoke`` via ``ctx.run(...)`` so MLflow's active-span ContextVar is
    set during the LLM call, and the autolog spans nest under the parent.

    Pass-through of every other attribute (including ``namespace``,
    ``search``, ``asearch``, ``__enter__``, ``__exit__``) keeps langmem's
    own protocol intact.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        # langmem.LocalReflectionExecutor.__init__ reads .namespace directly.
        self.namespace = inner.namespace
        # Keyed by id(payload) — langmem queues the dict by reference and
        # the worker passes the SAME object into invoke().
        self._ctx_by_payload_id: dict[int, contextvars.Context] = {}
        self._ctx_lock = threading.Lock()

    def attach(self, payload: Any, ctx: contextvars.Context) -> None:
        """Associate ``payload`` (by identity) with a captured ``ctx``."""
        with self._ctx_lock:
            self._ctx_by_payload_id[id(payload)] = ctx

    def invoke(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        with self._ctx_lock:
            ctx = self._ctx_by_payload_id.pop(id(payload), None)
        if ctx is None:
            return self._inner.invoke(payload, *args, **kwargs)
        return ctx.run(self._inner.invoke, payload, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Pass through every other attribute (namespace properties,
        # search, asearch, store, etc.) to the real manager.
        return getattr(self._inner, name)

ModelSpec = Union[str, LanguageModelLike]


def _unwrap_to_base_chat_model(model: ModelSpec) -> str | BaseChatModel:
    """Ensure *model* is a ``str`` or ``BaseChatModel`` for langmem.

    ``langmem.create_memory_store_manager`` only accepts ``str`` or
    ``BaseChatModel``.  When the caller passes a ``RunnableWithFallbacks``
    (produced by ``InferenceEndpointModel.as_chat_model()`` when fallbacks are configured),
    we extract the primary runnable so langmem can use it directly.
    """
    if isinstance(model, str) or isinstance(model, BaseChatModel):
        return model
    if isinstance(model, RunnableWithFallbacks):
        return _unwrap_to_base_chat_model(model.runnable)
    raise TypeError(
        f"Cannot convert {type(model).__name__} to BaseChatModel for langmem. "
        "Pass a model name string or a BaseChatModel instance."
    )


def create_extraction_manager(
    model: ModelSpec,
    store: BaseStore,
    namespace: tuple[str, ...] = ("memory", "{user_id}"),
    schemas: list[str] | None = None,
    instructions: str | None = None,
    enable_inserts: bool = True,
    enable_deletes: bool = False,
    query_model: ModelSpec | None = None,
    query_limit: int = 5,
) -> MemoryStoreManager:
    """Create a memory store manager for extraction and search.

    Wraps ``langmem.create_memory_store_manager`` with DAO AI schema
    resolution and Databricks model names.

    Args:
        model: LLM for memory extraction -- either a model name string
            understood by LangChain (e.g. ``"gpt-4o"``) or a pre-built
            ``LanguageModelLike`` instance (e.g. ``ChatDatabricks``).
        store: The LangGraph ``BaseStore`` to persist memories into.
        namespace: Namespace tuple for memory isolation. Template variables
            like ``"{user_id}"`` are resolved from ``config.configurable``
            at runtime.
        schemas: Optional list of schema names to use (e.g.
            ``["user_profile", "preference"]``). Resolved via
            :func:`~dao_ai.memory.schemas.resolve_schemas`. When ``None``,
            langmem uses its default unstructured ``Memory`` schema.
        instructions: Custom extraction instructions. When ``None``, langmem
            uses its built-in instructions.
        enable_inserts: Whether to allow creating new memory entries.
        enable_deletes: Whether to allow deleting outdated memories.
        query_model: Optional separate model for memory search queries.
            Accepts a model name string or ``LanguageModelLike`` instance.
        query_limit: Maximum number of memories to retrieve for context
            during extraction.

    Returns:
        A configured ``MemoryStoreManager`` Runnable.
    """
    resolved_schemas: list[type[BaseModel]] | None = None
    if schemas:
        resolved_schemas = resolve_schemas(schemas)
        logger.debug(
            "Resolved memory schemas",
            schema_names=schemas,
            schema_classes=[s.__name__ for s in resolved_schemas],
        )

    kwargs: dict[str, Any] = {
        "namespace": namespace,
        "store": store,
        "enable_inserts": enable_inserts,
        "enable_deletes": enable_deletes,
        "query_limit": query_limit,
    }

    if resolved_schemas is not None:
        kwargs["schemas"] = resolved_schemas
    if instructions is not None:
        kwargs["instructions"] = instructions
    if query_model is not None:
        kwargs["query_model"] = _unwrap_to_base_chat_model(query_model)

    manager: MemoryStoreManager = create_memory_store_manager(
        _unwrap_to_base_chat_model(model), **kwargs
    )

    logger.info(
        "Memory extraction manager created",
        model=model,
        namespace=namespace,
        schemas=schemas,
        query_model=query_model,
    )

    return manager


class LazyReflectionExecutor:
    """Defers ``LocalReflectionExecutor`` creation to the first ``submit()`` call.

    ``langmem.LocalReflectionExecutor`` starts a **non-daemon** background
    thread (``daemon=False``) in ``__init__``.  When the executor is created
    during Model Serving initialization, this thread prevents the container
    from passing its readiness probe, causing a deployment timeout.

    This wrapper stores the construction arguments and creates the real
    executor lazily on first ``submit()``, which only happens during
    inference -- well after the endpoint has been deployed and is healthy.
    """

    def __init__(self, manager: MemoryStoreManager, store: BaseStore) -> None:
        self._manager = manager
        # Wrap the real manager so the langmem worker thread executes the
        # invoke() under the caller's captured contextvars (preserves the
        # active MLflow span). See _ContextAwareReflector docstring.
        self._proxy_manager = _ContextAwareReflector(manager)
        self._store = store
        self._executor: LocalReflectionExecutor | None = None
        self._lock = threading.Lock()

    def _ensure_executor(self) -> LocalReflectionExecutor:
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = ReflectionExecutor(
                        self._proxy_manager, store=self._store
                    )
                    logger.debug("ReflectionExecutor created (deferred)")
        return self._executor

    def submit(self, payload: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Forward to the real executor, creating it on first call.

        Captures the caller's contextvars and associates it with ``payload``
        by identity. The proxy reflector replays the context when langmem's
        worker thread invokes the manager.
        """
        ctx = contextvars.copy_context()
        self._proxy_manager.attach(payload, ctx)
        return self._ensure_executor().submit(payload, *args, **kwargs)


def create_reflection_executor(
    manager: MemoryStoreManager,
    store: BaseStore,
) -> LazyReflectionExecutor:
    """Create a lazy background reflection executor for async memory extraction.

    Returns a ``LazyReflectionExecutor`` that defers the actual
    ``langmem.ReflectionExecutor`` construction (and its background
    thread) until the first ``submit()`` call.  This prevents the
    background thread from interfering with Model Serving readiness
    probes during endpoint deployment.

    Args:
        manager: The ``MemoryStoreManager`` to use for extraction.
        store: The ``BaseStore`` for persistence.

    Returns:
        A ``LazyReflectionExecutor`` that exposes ``submit()`` for
        scheduling background extraction.
    """
    executor = LazyReflectionExecutor(manager, store)

    logger.debug("Lazy reflection executor created (thread deferred)")

    return executor


__all__ = [
    "LazyReflectionExecutor",
    "create_extraction_manager",
    "create_reflection_executor",
]

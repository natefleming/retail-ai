"""Lifecycle contracts for Databricks-backed config models.

Resource lifecycle has two orthogonal concerns:

- :class:`Provisionable` — a model that can be pushed to the workspace via
  ``create()`` (which transparently does insert-or-update with a diff check
  when the resource already exists).
- :class:`Refreshable` — a model that can pull live workspace state into
  its structured fields via ``refresh()``.

A model can implement either or both. :class:`ManagedResource` is sugar for
"both" — the full read/write lifecycle expected of dual-mode resources
like :class:`~dao_ai.config.GenieRoomModel` and
:class:`~dao_ai.config.VectorStoreModel`.

The pattern mirrors well-trodden Python ORM ergonomics: ``create()``
parallels SQLAlchemy ``session.add() + session.commit()``; ``refresh()``
parallels Django's ``Model.refresh_from_db()`` and SQLAlchemy's
``session.refresh``.
"""

from abc import ABC, abstractmethod
from typing import Self


class Provisionable(ABC):
    """A model that creates (or idempotently updates) a workspace resource.

    Implementations should:

    - Be idempotent: re-running with no changes is a no-op.
    - Prefer ``update`` over ``create`` when the resource already exists,
      guarded by a diff check against the live state.
    - Write any server-assigned identifier (e.g. ``space_id``) back to the
      model so subsequent calls see a fully-resolved instance.
    """

    @abstractmethod
    def create(self) -> None:
        """Provision or update the underlying Databricks resource."""


class Refreshable(ABC):
    """A model that hydrates its structured fields from live workspace state.

    Contract:

    - Mutates fields in place; does not reconstruct the model.
      (Reconstruction would re-fire ``model_validator(mode="after")``,
      which would double-run ``ResourcesModel`` auto-population.)
    - Is idempotent: two consecutive calls produce the same state.
    - Is lossless: sub-models that round-trip through external JSON should
      use ``extra="allow"`` so unmodeled server metadata survives the
      ``refresh() → create()`` cycle.
    - Is read-only against the workspace.
    """

    @abstractmethod
    def refresh(self, *, force: bool = False) -> Self:
        """Fetch live state and write it into the model's structured fields.

        Args:
            force: If True, invalidate any cached server response before
                re-fetching. Use after mutating writes so subsequent reads
                see post-write state.

        Returns:
            self, for chaining.
        """


class ManagedResource(Provisionable, Refreshable, ABC):
    """A resource with the full provisioning + hydration lifecycle.

    Convenience for resources that implement both contracts. Equivalent to
    inheriting from :class:`Provisionable` and :class:`Refreshable` directly.
    """

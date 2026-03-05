"""
Structured memory schemas for DAO AI agents.

Provides typed Pydantic schemas for use with langmem's memory extraction
system. These schemas guide what information is extracted and how it is
stored, enabling richer memory than unstructured strings.

Three memory types are provided:

- **UserProfile** (semantic, profile-style): A single consolidated document
  per user that accumulates name, preferences, communication style, and
  domain context over time.
- **PreferenceMemory** (semantic, collection-style): Individual preference
  records that can be searched independently.
- **EpisodeMemory** (episodic, collection-style): Records of past
  interactions that capture situation, approach, and outcome for
  experience-based learning.

Schemas are referenced by name in YAML config and resolved at runtime via
:func:`resolve_schemas`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """A consolidated profile of a user, updated over time.

    Stored as a single document per user (profile-style memory). New
    information merges into the existing profile rather than creating
    separate records.
    """

    name: str = Field(default="", description="The user's name")
    preferred_name: str = Field(
        default="", description="How the user prefers to be addressed"
    )
    role: str = Field(default="", description="The user's role or job title")
    organization: str = Field(default="", description="The user's organization or team")
    communication_style: str = Field(
        default="",
        description="How the user prefers to communicate (formal, casual, etc.)",
    )
    expertise: list[str] = Field(
        default_factory=list,
        description="Areas of expertise or technical skills",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="General preferences and settings",
    )
    goals: list[str] = Field(
        default_factory=list,
        description="Current goals or objectives the user is working toward",
    )
    context: str = Field(
        default="",
        description="Additional context about the user's situation or needs",
    )


class PreferenceMemory(BaseModel):
    """An individual preference expressed by the user.

    Stored as a collection (multiple records per user). Each preference
    captures a specific category and the user's stated or inferred
    preference within that category.
    """

    category: str = Field(
        ..., description="Category of the preference (e.g. 'response_style', 'tools')"
    )
    preference: str = Field(
        ..., description="The specific preference expressed by the user"
    )
    context: str = Field(
        default="",
        description="Context in which the preference was expressed",
    )


class EpisodeMemory(BaseModel):
    """A record of a notable past interaction.

    Captures the situation, the approach taken, and the outcome so the
    agent can learn from experience and apply similar strategies in
    future interactions.
    """

    situation: str = Field(..., description="The situation or problem that arose")
    approach: str = Field(..., description="The approach or action taken to address it")
    outcome: str = Field(
        ..., description="The result and why the approach worked or didn't"
    )
    lesson: str = Field(
        default="",
        description="Key takeaway or lesson learned from this interaction",
    )


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "user_profile": UserProfile,
    "preference": PreferenceMemory,
    "episode": EpisodeMemory,
}


def resolve_schemas(names: list[str]) -> list[type[BaseModel]]:
    """Resolve schema names from config to Pydantic model classes.

    Args:
        names: List of schema names (e.g. ``["user_profile", "preference"]``).

    Returns:
        List of Pydantic model classes.

    Raises:
        ValueError: If a schema name is not found in the registry.
    """
    schemas: list[type[BaseModel]] = []
    for name in names:
        schema = SCHEMA_REGISTRY.get(name)
        if schema is None:
            available = ", ".join(sorted(SCHEMA_REGISTRY.keys()))
            raise ValueError(
                f"Unknown memory schema '{name}'. Available schemas: {available}"
            )
        schemas.append(schema)
    return schemas


__all__ = [
    "EpisodeMemory",
    "PreferenceMemory",
    "SCHEMA_REGISTRY",
    "UserProfile",
    "resolve_schemas",
]

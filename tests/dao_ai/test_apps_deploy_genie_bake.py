"""The SDK Apps deploy path must bake Genie room details, like the DABs path.

``_bake_genie_room_details`` had exactly one call site — the DABs App-bundle
writer — so only ``dao-ai agent up --mode apps`` baked. ``workflow up --mode
apps`` goes through the SDK ``_deploy_app`` and shipped the config text unbaked,
which silently drops the **example-questions** block: the description survives
because the container's ``ensure_resolved`` back-fills it under ``CAN_RUN``, but
the serialized space payload needs ``CAN_EDIT``, which a deployed app SP does not
hold. Those questions are the routing signal the feature exists to provide.

These exercise ``_app_config_content`` rather than ``_deploy_app`` — the config
selection was extracted precisely so this is testable without standing up a
workspace client, a wheel build, and a `uv lock`.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from dao_ai.providers.databricks import _app_config_content

# Stand-in for the live spaces; only ids listed here are "readable".
_FAKE_SPACES: dict[str, dict[str, Any]] = {
    "space-a": {
        "name": "Orders Space",
        "description": "Answers questions about retail orders.",
        "sample_questions": ["How many orders shipped late?", "Revenue by region?"],
    },
}

_CONFIG_YAML: str = dedent(
    """
    resources:
      genie_rooms:
        orders:
          space_id: space-a
    """
).lstrip()


@pytest.fixture
def fake_genie_discovery(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch ``ensure_resolved``, the seam the baker uses, so no client is built.

    Same fixture shape as ``test_config_vars``' bake tests. Returns the looked-up
    space ids so a test can assert whether discovery ran at all.
    """
    from dao_ai.config import GenieRoomModel

    looked_up: list[str] = []

    def _fake_ensure_resolved(self: GenieRoomModel) -> None:
        looked_up.append(str(self.space_id))
        details = _FAKE_SPACES.get(str(self.space_id))
        if details is None:
            raise ValueError(f"space {self.space_id} is not readable")
        for field, value in details.items():
            setattr(self, field, value)

    monkeypatch.setattr(GenieRoomModel, "ensure_resolved", _fake_ensure_resolved)
    return looked_up


def _fake_config(
    *,
    rendered_yaml: str | None = None,
    source_config_path: str | None = None,
    model_dump: dict[str, Any] | None = None,
) -> Any:
    """The three attributes ``_app_config_content`` reads off an AppConfig."""
    return SimpleNamespace(
        rendered_yaml=rendered_yaml,
        source_config_path=source_config_path,
        model_dump=lambda **_kwargs: model_dump or {},
    )


@pytest.mark.unit
class TestAppConfigContent:
    def test_rendered_yaml_branch_is_baked(
        self, fake_genie_discovery: list[str]
    ) -> None:
        """The ``params=``-substituted text — the shape ``workflow up`` produces."""
        content, origin = _app_config_content(_fake_config(rendered_yaml=_CONFIG_YAML))

        room = yaml.safe_load(content.decode("utf-8"))["resources"]["genie_rooms"][
            "orders"
        ]
        assert room["sample_questions"] == _FAKE_SPACES["space-a"]["sample_questions"]
        assert room["name"] == "Orders Space"
        assert room["space_id"] == "space-a"
        assert "rendered_yaml" in origin
        assert fake_genie_discovery == ["space-a"]

    def test_source_file_branch_is_baked(
        self, tmp_path: Path, fake_genie_discovery: list[str]
    ) -> None:
        """The legacy ``from_file`` shape: raw text read back off disk."""
        source = tmp_path / "config.yaml"
        source.write_text(_CONFIG_YAML, encoding="utf-8")

        content, origin = _app_config_content(
            _fake_config(source_config_path=str(source))
        )

        room = yaml.safe_load(content.decode("utf-8"))["resources"]["genie_rooms"][
            "orders"
        ]
        assert room["sample_questions"] == _FAKE_SPACES["space-a"]["sample_questions"]
        assert str(source) in origin

    def test_python_built_config_is_not_baked(
        self, fake_genie_discovery: list[str]
    ) -> None:
        """A programmatic AppConfig dumps objects ``initialize()`` already resolved
        with the deployer's credentials, so there is nothing to discover — and no
        text predating resolution to discover it into."""
        dumped = {"resources": {"genie_rooms": {"orders": {"space_id": "space-a"}}}}

        content, origin = _app_config_content(_fake_config(model_dump=dumped))

        assert yaml.safe_load(content.decode("utf-8")) == dumped
        assert origin == "in-memory AppConfig (programmatic)"
        assert fake_genie_discovery == [], "the model_dump branch must not discover"

    def test_rendered_yaml_wins_over_source_file(
        self, tmp_path: Path, fake_genie_discovery: list[str]
    ) -> None:
        """Substituted text beats the raw file — otherwise a ``${param.X}`` deploy
        would ship unsubstituted placeholders."""
        source = tmp_path / "config.yaml"
        source.write_text("resources: {}\n", encoding="utf-8")

        content, _ = _app_config_content(
            _fake_config(rendered_yaml=_CONFIG_YAML, source_config_path=str(source))
        )

        assert "orders" in content.decode("utf-8")

    def test_bake_failure_falls_back_to_unbaked_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Discovery is best-effort. The baker itself no-ops on a parse failure or
        an unreadable space, but it does not guard the caller, and a deploy must
        never be blocked by a lookup."""
        import dao_ai.apps.bundle as apps_bundle

        def _explode(_rendered: str) -> str:
            raise RuntimeError("genie is down")

        monkeypatch.setattr(apps_bundle, "_bake_genie_room_details", _explode)

        content, origin = _app_config_content(_fake_config(rendered_yaml=_CONFIG_YAML))

        assert content == _CONFIG_YAML.encode("utf-8")
        assert "rendered_yaml" in origin

    def test_non_utf8_source_file_is_uploaded_unchanged(self, tmp_path: Path) -> None:
        """The source branch reads bytes; only the bake needs text. A file that
        will not decode still deploys, exactly as it did before the bake existed.
        """
        source = tmp_path / "config.yaml"
        raw = "resources: {}  # café\n".encode("latin-1")
        source.write_bytes(raw)

        content, _ = _app_config_content(_fake_config(source_config_path=str(source)))

        assert content == raw

    def test_config_without_genie_rooms_is_untouched(
        self, fake_genie_discovery: list[str]
    ) -> None:
        """Byte-for-byte identical, so the bake cannot reformat an unrelated
        config on its way out."""
        src = "app:\n  name: plain\n"

        content, _ = _app_config_content(_fake_config(rendered_yaml=src))

        assert content == src.encode("utf-8")
        assert fake_genie_discovery == []


@pytest.mark.unit
class TestBakedDescriptionSurvivesReload:
    """Whatever the bake writes, the *container* has to be able to re-validate.

    ``GenieRoomModel.description`` is capped at
    ``APP_RESOURCE_DESCRIPTION_MAX_LENGTH`` (200) to match the Apps platform limit
    on ``AppResource.description``. A Genie space description is workspace data,
    though, and nothing stops it being longer — so baking it in verbatim produced
    a config that passed at deploy time and then failed the identical cap inside
    the container, with ``AppConfig.from_file`` raising ``string_too_long`` before
    the app could serve a single request.

    These go through the real ``ensure_resolved`` (not the patched seam the tests
    above use), because the clip lives at that one point of adoption — which is
    also what keeps the Model Serving ``model_config`` dump loadable.
    """

    _LONG_DESCRIPTION: str = "Retail order analytics. " * 20  # 480 chars

    @pytest.fixture
    def long_description_space(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A live space whose description exceeds the config cap."""
        from unittest.mock import Mock

        space = Mock()
        space.space_id = "space-long"
        space.title = "Orders Space"
        space.description = self._LONG_DESCRIPTION
        space.warehouse_id = None
        space.serialized_space = "{}"

        client = Mock()
        client.genie.get_space.return_value = space
        monkeypatch.setattr("dao_ai.config.WorkspaceClient", lambda *a, **k: client)

    def test_resolution_clips_the_description(self, long_description_space: None) -> None:
        from dao_ai.config import APP_RESOURCE_DESCRIPTION_MAX_LENGTH, GenieRoomModel

        room = GenieRoomModel(space_id="space-long")
        room.ensure_resolved()

        assert room.description is not None
        assert len(room.description) <= APP_RESOURCE_DESCRIPTION_MAX_LENGTH
        assert room.description.endswith("…")
        # Not a mid-word cut followed by a space: the clip rstrips before the
        # ellipsis so "analytics. …" never appears.
        assert not room.description.endswith(" …")
        assert room.description.startswith("Retail order analytics.")

    def test_baked_config_reloads_in_the_container(
        self, long_description_space: None, tmp_path: Path
    ) -> None:
        """End to end: bake, then re-parse the uploaded bytes the way the app does."""
        from dao_ai.config import AppConfig

        rendered = dedent(
            """
            resources:
              genie_rooms:
                orders:
                  space_id: space-long
            """
        ).lstrip()

        content, _ = _app_config_content(_fake_config(rendered_yaml=rendered))

        uploaded = tmp_path / "model_config.yaml"
        uploaded.write_bytes(content)
        reloaded = AppConfig.from_file(str(uploaded), initialize=False)

        description = reloaded.resources.genie_rooms["orders"].description
        assert description is not None and description.endswith("…")

    def test_a_short_description_is_left_exactly_as_written(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clipping is for the overlong case only — no stray ellipsis, no rstrip."""
        from unittest.mock import Mock

        from dao_ai.config import GenieRoomModel

        space = Mock()
        space.title = "Orders Space"
        space.description = "Answers questions about retail orders. "
        space.warehouse_id = None
        space.serialized_space = "{}"
        client = Mock()
        client.genie.get_space.return_value = space
        monkeypatch.setattr("dao_ai.config.WorkspaceClient", lambda *a, **k: client)

        room = GenieRoomModel(space_id="space-short")
        room.ensure_resolved()

        assert room.description == "Answers questions about retail orders. "

"""Tests for the LLM-in-hooks scaffolding (gemma4 / Ollama integration).

These tests inject a ``FakeProvider`` via monkeypatch — no real Ollama
roundtrip happens in CI. We cover:

* the cache: failed probes are sticky within a process (no retry-storm)
* the success path: gemma topics merge with keyword themes
* the JSON-recovery path: prose-wrapped JSON still parses
* the failure paths: classify error / malformed JSON / empty response
  all return ``[]`` so the deterministic keyword path takes over
* the gate: when ``hook_llm_themes`` is False, no LLM call happens
"""

from __future__ import annotations

import json

import pytest

from mempalace import hooks_cli


class FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.model = "fake"
        self.provider = "fake"
        self.raw = {}


class FakeProvider:
    """Drop-in provider that returns a scripted response or raises."""

    def __init__(self, *, response: str | None = None, raise_on_call: Exception | None = None):
        self._response = response
        self._raise = raise_on_call
        self.calls: list[tuple[str, str]] = []

    def check_available(self):
        return True, "ok"

    def classify(self, system: str, user: str, json_mode: bool = True):
        self.calls.append((system, user))
        if self._raise is not None:
            raise self._raise
        return FakeResponse(self._response or "")


@pytest.fixture(autouse=True)
def _reset_hook_llm_cache():
    """Each test starts with an empty hook-LLM cache."""
    hooks_cli._HOOK_LLM_CACHE.clear()
    yield
    hooks_cli._HOOK_LLM_CACHE.clear()


@pytest.fixture
def llm_on(monkeypatch):
    """Force the LLM-themes flag on regardless of user config."""
    monkeypatch.setenv("MEMPALACE_HOOK_LLM_THEMES", "1")
    monkeypatch.setenv("MEMPALACE_HOOK_LLM_DIARY", "0")
    monkeypatch.setenv("MEMPALACE_HOOK_LLM_PRECOMPACT_KG", "0")


@pytest.fixture
def llm_off(monkeypatch):
    """Force every LLM-in-hook flag off."""
    monkeypatch.setenv("MEMPALACE_HOOK_LLM_THEMES", "0")
    monkeypatch.setenv("MEMPALACE_HOOK_LLM_DIARY", "0")
    monkeypatch.setenv("MEMPALACE_HOOK_LLM_PRECOMPACT_KG", "0")


def _install_provider(monkeypatch, provider):
    """Make ``_get_hook_llm`` return ``provider`` regardless of Ollama state."""

    def fake_get():
        return provider

    monkeypatch.setattr(hooks_cli, "_get_hook_llm", fake_get)


# ── Config plumbing ────────────────────────────────────────────────────────


def test_config_defaults_on(monkeypatch, tmp_path):
    """Fresh config dir: every hook_llm_* flag defaults to True."""
    monkeypatch.delenv("MEMPALACE_HOOK_LLM_THEMES", raising=False)
    monkeypatch.delenv("MEMPALACE_HOOK_LLM_DIARY", raising=False)
    monkeypatch.delenv("MEMPALACE_HOOK_LLM_PRECOMPACT_KG", raising=False)
    from mempalace.config import MempalaceConfig

    cfg = MempalaceConfig(config_dir=tmp_path)
    assert cfg.hook_llm_themes is True
    assert cfg.hook_llm_diary is True
    assert cfg.hook_llm_precompact_kg is True
    assert cfg.hook_llm_provider == "ollama"
    assert cfg.hook_llm_model == "gemma4:e4b"
    assert cfg.hook_llm_timeout_s == 10


def test_config_env_override_off(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_HOOK_LLM_THEMES", "0")
    from mempalace.config import MempalaceConfig

    cfg = MempalaceConfig(config_dir=tmp_path)
    assert cfg.hook_llm_themes is False


def test_config_env_override_on_when_file_off(monkeypatch, tmp_path):
    """Env beats config file."""
    (tmp_path / "config.json").write_text(json.dumps({"hooks": {"llm_themes": False}}))
    monkeypatch.setenv("MEMPALACE_HOOK_LLM_THEMES", "1")
    from mempalace.config import MempalaceConfig

    cfg = MempalaceConfig(config_dir=tmp_path)
    assert cfg.hook_llm_themes is True


def test_config_invalid_timeout_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_HOOK_LLM_TIMEOUT_S", "not-a-number")
    from mempalace.config import MempalaceConfig

    cfg = MempalaceConfig(config_dir=tmp_path)
    assert cfg.hook_llm_timeout_s == 10  # default


# ── _get_hook_llm cache behavior ───────────────────────────────────────────


def test_get_hook_llm_caches_unavailable(monkeypatch):
    """A failed check_available is sticky — no retry-storm across hook fires."""

    class Unavailable:
        def __init__(self, *_, **__):
            pass

        def check_available(self):
            return False, "not running"

    calls = {"n": 0}

    def fake_get_provider(*_args, **_kwargs):
        calls["n"] += 1
        return Unavailable()

    monkeypatch.setattr("mempalace.llm_client.get_provider", fake_get_provider)

    assert hooks_cli._get_hook_llm() is None
    assert hooks_cli._get_hook_llm() is None
    assert hooks_cli._get_hook_llm() is None
    assert calls["n"] == 1, "get_provider should be called once and the failure cached"


# ── _llm_themes contract ──────────────────────────────────────────────────


def test_llm_themes_happy_path(monkeypatch, llm_on):
    fake = FakeProvider(response=json.dumps({"topics": ["embedding", "voyage AI", "hooks"]}))
    _install_provider(monkeypatch, fake)
    out = hooks_cli._llm_themes(["msg about voyage embedding"])
    assert out == ["embedding", "voyage AI", "hooks"]
    assert len(fake.calls) == 1


def test_llm_themes_dedup_case_insensitive(monkeypatch, llm_on):
    fake = FakeProvider(response=json.dumps({"topics": ["Voyage", "voyage", "VOYAGE"]}))
    _install_provider(monkeypatch, fake)
    out = hooks_cli._llm_themes(["x"])
    assert out == ["Voyage"]


def test_llm_themes_recovers_from_prose_wrapped_json(monkeypatch, llm_on):
    """Some local models still wrap JSON despite ``format=json``."""
    fake = FakeProvider(
        response='Sure, here you go:\n```json\n{"topics": ["alpha", "beta"]}\n```'
    )
    _install_provider(monkeypatch, fake)
    out = hooks_cli._llm_themes(["x"])
    assert out == ["alpha", "beta"]


def test_llm_themes_returns_empty_on_classify_error(monkeypatch, llm_on):
    fake = FakeProvider(raise_on_call=RuntimeError("boom"))
    _install_provider(monkeypatch, fake)
    assert hooks_cli._llm_themes(["x"]) == []


def test_llm_themes_returns_empty_on_garbage(monkeypatch, llm_on):
    fake = FakeProvider(response="not json at all, no braces")
    _install_provider(monkeypatch, fake)
    assert hooks_cli._llm_themes(["x"]) == []


def test_llm_themes_returns_empty_when_topics_missing(monkeypatch, llm_on):
    fake = FakeProvider(response=json.dumps({"foo": "bar"}))
    _install_provider(monkeypatch, fake)
    assert hooks_cli._llm_themes(["x"]) == []


def test_llm_themes_drops_unsafe_names(monkeypatch, llm_on):
    """sanitize_name rejects path-traversal and control bytes."""
    fake = FakeProvider(
        response=json.dumps({"topics": ["good", "../bad", "also/bad", "fine"]})
    )
    _install_provider(monkeypatch, fake)
    out = hooks_cli._llm_themes(["x"])
    assert out == ["good", "fine"]


def test_llm_themes_returns_empty_when_provider_none(monkeypatch, llm_on):
    monkeypatch.setattr(hooks_cli, "_get_hook_llm", lambda: None)
    assert hooks_cli._llm_themes(["x"]) == []


def test_llm_themes_returns_empty_for_empty_messages(monkeypatch, llm_on):
    fake = FakeProvider(response=json.dumps({"topics": ["should-not-be-called"]}))
    _install_provider(monkeypatch, fake)
    assert hooks_cli._llm_themes([]) == []
    assert fake.calls == []


# ── _extract_themes (gate behavior) ────────────────────────────────────────


def test_extract_themes_skips_llm_when_flag_off(monkeypatch, llm_off):
    fake = FakeProvider(response=json.dumps({"topics": ["should-not-be-used"]}))
    _install_provider(monkeypatch, fake)
    out = hooks_cli._extract_themes(["voyage backend embedding"], max_themes=3)
    # Pure keyword path
    assert "voyage" in out
    assert fake.calls == [], "LLM should never be called when hook_llm_themes is off"


def test_extract_themes_unions_llm_first(monkeypatch, llm_on):
    fake = FakeProvider(response=json.dumps({"topics": ["alpha", "beta"]}))
    _install_provider(monkeypatch, fake)
    out = hooks_cli._extract_themes(
        ["voyage backend embedding", "another message about hooks"], max_themes=3
    )
    # LLM topics come first, then keyword fallbacks fill remaining slots
    assert out[0] == "alpha"
    assert out[1] == "beta"
    assert len(out) <= 7


def test_extract_themes_falls_back_to_keyword_on_llm_failure(monkeypatch, llm_on):
    fake = FakeProvider(raise_on_call=RuntimeError("network"))
    _install_provider(monkeypatch, fake)
    out = hooks_cli._extract_themes(["voyage backend embedding"], max_themes=3)
    assert "voyage" in out  # keyword path survived

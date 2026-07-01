"""Per-profile isolation for the Honcho client cache.

Single-process multi-profile runtimes (desktop ``tui_gateway``, the multiplexed
gateway) switch profile via the ``HERMES_HOME`` ContextVar override, not the
process env. A process-global client singleton pinned every profile to
whichever built it first, so profile B's memory reads/writes hit profile A's
workspace under profile A's credentials. The client is now cached per profile
(keyed by resolved HERMES_HOME); these tests drive two overrides and assert
each profile gets its own client built from its own config.
"""

import sys
import types

import pytest

from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from plugins.memory.honcho import client as honcho_client
from plugins.memory.honcho.client import (
    HonchoClientConfig,
    get_honcho_client,
    reset_honcho_client,
)


@pytest.fixture(autouse=True)
def _reset_clients():
    reset_honcho_client()
    yield
    reset_honcho_client()


def _install_fake_honcho_sdk(monkeypatch):
    """Make ``from honcho import Honcho`` resolve to a kwargs-recording fake."""

    class _FakeHoncho:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_mod = types.ModuleType("honcho")
    fake_mod.Honcho = _FakeHoncho
    monkeypatch.setitem(sys.modules, "honcho", fake_mod)


def _cfg(workspace: str, api_key: str) -> HonchoClientConfig:
    # base_url + timeout set so _build() skips the load_config() fallback path.
    return HonchoClientConfig(
        api_key=api_key,
        base_url="https://api.honcho.dev",
        timeout=30.0,
        workspace_id=workspace,
        environment="production",
    )


def _under_override(home, fn):
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_distinct_clients_per_profile(monkeypatch, tmp_path):
    """Profile B must not inherit profile A's frozen client/workspace."""
    _install_fake_honcho_sdk(monkeypatch)
    prof_a = tmp_path / "profA"
    prof_b = tmp_path / "profB"
    prof_a.mkdir()
    prof_b.mkdir()

    client_a = _under_override(prof_a, lambda: get_honcho_client(_cfg("ws_a", "key_a")))
    client_b = _under_override(prof_b, lambda: get_honcho_client(_cfg("ws_b", "key_b")))

    assert client_a is not client_b
    assert client_a.kwargs["workspace_id"] == "ws_a"
    assert client_b.kwargs["workspace_id"] == "ws_b"
    assert client_a.kwargs["api_key"] == "key_a"
    assert client_b.kwargs["api_key"] == "key_b"


def test_same_profile_reuses_cached_client(monkeypatch, tmp_path):
    """Repeat calls under one profile hit the cache, not a rebuild."""
    _install_fake_honcho_sdk(monkeypatch)
    prof_a = tmp_path / "profA"
    prof_a.mkdir()

    first = _under_override(prof_a, lambda: get_honcho_client(_cfg("ws_a", "key_a")))
    second = _under_override(prof_a, lambda: get_honcho_client(_cfg("ws_a", "key_a")))
    assert first is second


def test_reset_clears_all_profiles(monkeypatch, tmp_path):
    """reset_honcho_client() drops every profile's cached client."""
    _install_fake_honcho_sdk(monkeypatch)
    prof_a = tmp_path / "profA"
    prof_b = tmp_path / "profB"
    prof_a.mkdir()
    prof_b.mkdir()

    a1 = _under_override(prof_a, lambda: get_honcho_client(_cfg("ws_a", "key_a")))
    b1 = _under_override(prof_b, lambda: get_honcho_client(_cfg("ws_b", "key_b")))

    reset_honcho_client()

    a2 = _under_override(prof_a, lambda: get_honcho_client(_cfg("ws_a", "key_a")))
    b2 = _under_override(prof_b, lambda: get_honcho_client(_cfg("ws_b", "key_b")))
    assert a2 is not a1
    assert b2 is not b1

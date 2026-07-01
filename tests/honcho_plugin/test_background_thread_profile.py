"""Honcho background threads must resolve the client under the active profile.

The per-profile client cache keys on the resolved HERMES_HOME read from the
ContextVar override. A bare ``threading.Thread`` starts with an empty context,
so the key reverts to the launch/default profile — filing the active profile's
client under the wrong slot (or serving it another profile's cached client).
Every honcho daemon spawn is wrapped in ``propagate_context_to_thread`` so the
worker inherits the request's profile; these tests prove that wrap is
load-bearing.
"""

import sys
import threading
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
from tools.thread_context import propagate_context_to_thread


@pytest.fixture(autouse=True)
def _reset_clients():
    reset_honcho_client()
    yield
    reset_honcho_client()


def _install_fake_honcho_sdk(monkeypatch):
    class _FakeHoncho:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_mod = types.ModuleType("honcho")
    fake_mod.Honcho = _FakeHoncho
    monkeypatch.setitem(sys.modules, "honcho", fake_mod)


def _cfg(workspace: str, api_key: str) -> HonchoClientConfig:
    return HonchoClientConfig(
        api_key=api_key,
        base_url="https://api.honcho.dev",
        timeout=30.0,
        workspace_id=workspace,
        environment="production",
    )


def _key_from_thread(target_wrapper):
    captured = {}

    def worker():
        get_honcho_client(_cfg("ws_w", "key_w"))
        captured["key"] = honcho_client._honcho_client_cache_key()

    t = threading.Thread(target=target_wrapper(worker))
    t.start()
    t.join()
    return captured["key"]


def test_wrapped_thread_files_client_under_active_profile(monkeypatch, tmp_path):
    _install_fake_honcho_sdk(monkeypatch)
    prof_w = tmp_path / "work"
    prof_w.mkdir()

    token = set_hermes_home_override(str(prof_w))
    try:
        key = _key_from_thread(propagate_context_to_thread)
    finally:
        reset_hermes_home_override(token)

    assert key == str(prof_w.resolve())
    # The active profile's slot holds the client; nothing else was built.
    assert honcho_client._honcho_client_slots[key].peek() is not None


def test_bare_thread_reverts_to_launch_profile(monkeypatch, tmp_path):
    """Negative control: without the wrap the key ignores the override."""
    _install_fake_honcho_sdk(monkeypatch)
    prof_w = tmp_path / "work"
    prof_w.mkdir()

    token = set_hermes_home_override(str(prof_w))
    try:
        bare_key = _key_from_thread(lambda fn: fn)
    finally:
        reset_hermes_home_override(token)

    # A bare thread's empty context resolves away from the active profile —
    # exactly the leak the wrap prevents.
    assert bare_key != str(prof_w.resolve())

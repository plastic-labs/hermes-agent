"""Provider shutdown joins every plugin thread within one budget (#37632, #33485, #60616)."""

import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho import session as session_module
from plugins.memory.honcho.client import (
    HonchoClientConfig,
    close_honcho_clients,
    join_plugin_threads,
    spawn_context_thread,
)
from plugins.memory.honcho.client_cache import _client_slots, _client_slots_lock
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager
from plugins.plugin_utils import SingletonSlot


def _hybrid_config(**overrides) -> SimpleNamespace:
    cfg = SimpleNamespace(
        enabled=True, api_key="test-key", base_url=None, recall_mode="hybrid", init_on_session_start=False,
        injection_frequency="every-turn", context_cadence=1, dialectic_cadence=1, query_rewrite=False,
        first_turn_base_wait=3.0, first_turn_dialectic_wait=2.0, dialectic_depth=1, dialectic_depth_levels=None,
        reasoning_heuristic=True, reasoning_level_cap="high", context_tokens=None, message_max_chars=25000,
        session_strategy="per-directory", timeout=None, save_messages=True,
    )
    cfg.resolve_session_name = lambda **kwargs: "test-session"
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


@pytest.fixture
def async_manager(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(session_module, "get_honcho_client", lambda *a, **k: fake)
    cfg = HonchoClientConfig(write_frequency="async", api_key="test-key", enabled=True)
    mgr = HonchoSessionManager(honcho=fake, config=cfg)
    mgr.fake_client = fake
    yield mgr
    mgr.shutdown()


class _Owner:
    pass


class TestThreadRegistry:
    def test_join_covers_only_the_given_owners(self):
        mine, theirs = _Owner(), _Owner()
        release = threading.Event()
        own = spawn_context_thread(lambda: release.wait(timeout=2), name="own", owner=mine)
        other = spawn_context_thread(lambda: release.wait(timeout=2), name="other", owner=theirs)
        own.start()
        other.start()
        try:
            started = time.monotonic()
            assert join_plugin_threads((theirs, None), timeout=0.05) == ["other"]
            release.set()
            assert join_plugin_threads((mine,), timeout=2) == []
            assert time.monotonic() - started < 1.5
        finally:
            release.set()
            own.join(timeout=1)
            other.join(timeout=1)

    def test_join_reports_threads_that_outlive_the_budget(self):
        owner = _Owner()
        release = threading.Event()
        t = spawn_context_thread(lambda: release.wait(timeout=2), name="honcho-slow", owner=owner)
        t.start()
        try:
            assert join_plugin_threads((owner,), timeout=0.05) == ["honcho-slow"]
        finally:
            release.set()
            t.join(timeout=1)


class TestProviderShutdown:
    def _provider(self, manager, cfg=None):
        provider = HonchoMemoryProvider()
        provider._manager = manager
        provider._config = cfg or manager._config
        provider._session_key = "test-session"
        provider._session_initialized = True
        return provider

    def test_shutdown_waits_for_an_in_flight_context_prefetch(self, async_manager):
        provider = self._provider(async_manager)
        started, release, done = threading.Event(), threading.Event(), threading.Event()

        def slow_prefetch(session_key, user_message=None):
            started.set()
            release.wait(timeout=2)
            return {"representation": "ready"}

        async_manager.get_prefetch_context = slow_prefetch
        async_manager.prefetch_context("test-session", "query")
        assert started.wait(timeout=1)

        def shut_down():
            try:
                provider.shutdown()
            finally:
                done.set()

        shutdown_thread = threading.Thread(target=shut_down, daemon=True)
        shutdown_thread.start()
        try:
            assert not done.wait(timeout=0.05)
            release.set()
            assert done.wait(timeout=2)
        finally:
            release.set()
            shutdown_thread.join(timeout=2)
        assert not any(t.name == "honcho-context-prefetch" and t.is_alive() for t in threading.enumerate())

    def test_shutdown_joins_a_stalled_init_thread_within_the_budget(self, monkeypatch, caplog):
        monkeypatch.setattr(HonchoMemoryProvider, "_SHUTDOWN_JOIN_FLOOR", 0.2)
        cfg = _hybrid_config(timeout=0.1)
        release = threading.Event()
        entered = threading.Event()

        class StalledManager:
            def __init__(self, *args, **kwargs):
                pass

            def get_or_create(self, session_key):
                entered.set()
                release.wait(timeout=5)
                return SimpleNamespace(messages=[])

            def migrate_memory_files(self, *a, **k):
                pass

        monkeypatch.setattr("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", lambda: cfg)
        monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", lambda cfg: object())
        monkeypatch.setattr("plugins.memory.honcho.session.HonchoSessionManager", StalledManager)
        provider = HonchoMemoryProvider()
        provider.initialize("session-1", platform="cli")
        try:
            assert entered.wait(timeout=1)
            started = time.monotonic()
            with caplog.at_level(logging.WARNING, logger="plugins.memory.honcho"):
                provider.shutdown()
            assert 0.15 <= time.monotonic() - started < 1.0
            assert "honcho-session-init" in caplog.text
            assert "timed out after 0.2s" in caplog.text
        finally:
            release.set()
            provider._init_thread.join(timeout=2)

    def test_shutdown_joins_the_prewarm_dialectic_thread(self, async_manager):
        provider = self._provider(async_manager)
        release = threading.Event()
        async_manager.dialectic_query = lambda *a, **k: release.wait(timeout=2) and "answer"
        provider._spawn_dialectic("who?", thread_name="honcho-prewarm-dialectic", fired_at=0, log_label="prewarm")
        done = threading.Event()

        def shut_down():
            provider.shutdown()
            done.set()

        threading.Thread(target=shut_down, daemon=True).start()
        assert not done.wait(timeout=0.05)
        release.set()
        assert done.wait(timeout=2)
        assert not provider._prefetch_thread.is_alive()

    def test_shutdown_does_not_close_the_shared_client(self, async_manager):
        provider = self._provider(async_manager)
        provider.shutdown()
        async_manager.fake_client._http.close.assert_not_called()


class TestShutdownBudget:
    def test_explicit_timeout_widens_the_join(self):
        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(api_key="k", timeout=60.0)
        assert provider._shutdown_join_budget() == 60.0

    def test_short_timeout_keeps_the_floor(self):
        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(api_key="k", timeout=2.0)
        assert provider._shutdown_join_budget() == 5.0

    def test_config_without_timeout_keeps_the_floor(self):
        provider = HonchoMemoryProvider()
        provider._config = SimpleNamespace()
        assert provider._shutdown_join_budget() == 5.0


class TestManagerAfterShutdown:
    def test_prefetch_is_skipped_after_shutdown(self, async_manager):
        calls = []
        async_manager.get_prefetch_context = lambda *a, **k: calls.append(a) or {}
        async_manager.shutdown()
        async_manager.prefetch_context("test-session", "query")
        time.sleep(0.05)
        assert calls == []

    def test_save_after_shutdown_flushes_inline_without_a_writer(self, async_manager):
        async_manager.shutdown()
        session = HonchoSession(key="k", user_peer_id="u", assistant_peer_id="a", honcho_session_id="k")
        session.add_message("user", "late")
        flushed = []
        async_manager._flush_session = lambda s: flushed.append(s) or True
        async_manager.save(session)
        assert flushed == [session]
        assert async_manager._async_thread is None


def test_close_honcho_clients_closes_every_pool_and_drops_the_slots():
    client = MagicMock()
    slot = SingletonSlot()
    slot.get(lambda: client)
    key = ("test", "close-me")
    with _client_slots_lock:
        _client_slots[key] = slot
    try:
        close_honcho_clients()
    finally:
        with _client_slots_lock:
            _client_slots.pop(key, None)
    client._http.close.assert_called_once_with()
    assert key not in _client_slots

"""A process that resumes a session must inject what a long-lived process would have.

In one long-lived process, turn N's injection comes from the refresh queued at the end of
turn N-1. ``hermes chat --resume <id> --query ...`` starts a new process at turn N, where that
refresh never ran and later turns do not wait, so before this fix it injected nothing.
"""

import time
from unittest.mock import MagicMock, patch

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


def _resumed_provider(context_queries: list[str]) -> HonchoMemoryProvider:
    """A provider whose session already holds the previous turn, with a Honcho that answers slowly."""
    manager = MagicMock()
    manager.get_or_create.return_value.messages = [
        {"role": "user", "content": "Add the ShapeIndex encoder."},
        {"role": "assistant", "content": "Done."},
    ]
    manager.pop_context_result.return_value = {}

    def _context(_key, query=None, **_kwargs):
        context_queries.append(query)
        time.sleep(0.3)
        return {"representation": f"conclusions recalled for: {query}"}

    def _dialectic(*_args, **_kwargs):
        time.sleep(0.3)
        return "The user wants every encoder round-trip tested."

    manager.get_prefetch_context.side_effect = _context
    manager.dialectic_query.side_effect = _dialectic

    cfg = HonchoClientConfig(api_key="test-key", enabled=True, recall_mode="hybrid", dialectic_depth=1)
    provider = HonchoMemoryProvider()
    with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
         patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
         patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=manager), \
         patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
        provider.initialize(session_id="resumed-session")
    if provider._init_thread is not None:
        provider._init_thread.join(timeout=5.0)
    return provider


def test_resumed_turn_injects_the_previous_turns_refresh():
    context_queries: list[str] = []
    provider = _resumed_provider(context_queries)

    provider.on_turn_start(2, "Now decode it back.")
    injected = provider.prefetch("Now decode it back.")

    assert "conclusions recalled for: Add the ShapeIndex encoder." in injected
    assert "The user wants every encoder round-trip tested." in injected
    assert context_queries[0] == "Add the ShapeIndex encoder."

    # The catch-up stands in for one missed refresh; the next turn is an ordinary later turn.
    provider.on_turn_start(3, "Cover the edge cases.")
    provider.prefetch("Cover the edge cases.")
    assert context_queries.count("Add the ShapeIndex encoder.") == 1

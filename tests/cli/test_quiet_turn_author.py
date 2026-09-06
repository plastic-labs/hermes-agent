"""``hermes chat -Q`` passes the dispatcher's HERMES_TURN_AUTHOR to ``run_conversation`` as ``turn_author``.

A bot-to-bot delivery runs the recipient's turn as a ``-Q`` subprocess with that variable set.
A human's ``-Q`` run has it unset and the turn stays unattributed.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import cli
from agent.turn_author import TURN_AUTHOR_ENV


@pytest.fixture(autouse=True)
def _plain_one_shot_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)


def _fake_cli(recorded):
    def run_conversation(**kwargs):
        recorded.append(kwargs)
        return {"final_response": "ok"}

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="s-1")
    return SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1")


def _run(monkeypatch, env_value):
    if env_value is None:
        monkeypatch.delenv(TURN_AUTHOR_ENV, raising=False)
    else:
        monkeypatch.setenv(TURN_AUTHOR_ENV, env_value)
    recorded = []
    with pytest.raises(SystemExit) as exc:
        cli._run_quiet_single_query(_fake_cli(recorded), "hello")
    assert exc.value.code == 0
    assert len(recorded) == 1
    assert recorded[0]["user_message"] == "hello"
    return recorded[0]


def test_quiet_one_shot_passes_turn_author_from_env(monkeypatch, capsys):
    author = {"id": "bot:coder", "name": "coder", "is_bot": True}
    kwargs = _run(monkeypatch, json.dumps(author))
    assert kwargs["turn_author"] == author
    assert capsys.readouterr().out.strip() == "ok"


def test_quiet_one_shot_consumes_the_variable_before_the_turn(monkeypatch):
    """Tool subprocesses spawned during the turn must not see the dispatcher's author."""
    author = {"id": "bot:coder", "name": "coder", "is_bot": True}
    seen = {}

    def run_conversation(**kwargs):
        seen["env"] = os.environ.get(TURN_AUTHOR_ENV)
        return {"final_response": "ok"}

    monkeypatch.setenv(TURN_AUTHOR_ENV, json.dumps(author))
    fake = _fake_cli([])
    fake.agent.run_conversation = run_conversation
    with pytest.raises(SystemExit):
        cli._run_quiet_single_query(fake, "hello")
    assert seen["env"] is None
    assert TURN_AUTHOR_ENV not in os.environ


def test_quiet_one_shot_without_env_keeps_the_old_call_shape(monkeypatch):
    assert "turn_author" not in _run(monkeypatch, None)


def test_quiet_one_shot_junk_env_keeps_the_old_call_shape(monkeypatch):
    assert "turn_author" not in _run(monkeypatch, "not json")


def test_quiet_one_shot_skips_the_keyword_for_an_agent_that_cannot_take_it(monkeypatch):
    """A wrapper with an older run_conversation signature must not crash the turn."""
    recorded = []

    def run_conversation(user_message, conversation_history=None):
        recorded.append(user_message)
        return {"final_response": "ok"}

    fake = _fake_cli([])
    fake.agent.run_conversation = run_conversation
    monkeypatch.setenv(TURN_AUTHOR_ENV, json.dumps({"id": "bot:coder", "is_bot": True}))
    with pytest.raises(SystemExit) as exc:
        cli._run_quiet_single_query(fake, "hello")
    assert exc.value.code == 0
    assert recorded == ["hello"]

"""Tests for agent.turn_author: the per-turn author carried from a dispatcher to memory hooks."""

import json

import pytest

from agent.turn_author import (
    TURN_AUTHOR_ENV,
    scope_for,
    parse_turn_author,
    take_turn_author_from_env,
    turn_author_env,
    turn_author_from_env,
)


class TestParseTurnAuthor:
    def test_dict_is_normalized(self):
        out = parse_turn_author({"id": "  bot:alpha ", "name": "Alpha", "is_bot": True, "extra": 1})
        assert out == {"id": "bot:alpha", "name": "Alpha", "is_bot": True}

    def test_json_string_is_parsed(self):
        raw = json.dumps({"id": "bot:alpha", "name": "Alpha", "is_bot": 1})
        assert parse_turn_author(raw) == {"id": "bot:alpha", "name": "Alpha", "is_bot": True}

    def test_author_without_id_or_name_is_none(self):
        assert parse_turn_author({}) is None
        assert parse_turn_author({"is_bot": True}) is None

    def test_missing_fields_default(self):
        assert parse_turn_author({"name": "Alpha"}) == {"id": None, "name": "Alpha", "is_bot": False}

    @pytest.mark.parametrize("flag", [True, 1, "true", "1", " YES "])
    def test_bot_flag_accepts_booleans_and_truthy_strings(self, flag):
        assert parse_turn_author({"id": "bot:alpha", "is_bot": flag})["is_bot"] is True

    @pytest.mark.parametrize("flag", [False, 0, None, "false", "0", "no", "", "bot", [True], {"a": 1}, 1.0])
    def test_bot_flag_rejects_everything_else(self, flag):
        assert parse_turn_author({"id": "bot:alpha", "is_bot": flag})["is_bot"] is False

    @pytest.mark.parametrize("raw", [None, 42, [], ["bot:alpha"], "not json", '"a string"', "[1, 2]", b"\xff"])
    def test_junk_returns_none(self, raw):
        assert parse_turn_author(raw) is None

    def test_non_string_fields_become_none(self):
        assert parse_turn_author({"id": 7, "name": "Alpha", "is_bot": "yes"}) == {
            "id": None, "name": "Alpha", "is_bot": True,
        }

    def test_empty_and_whitespace_become_none(self):
        assert parse_turn_author({"id": "", "name": "   "}) is None
        assert parse_turn_author({"id": "", "name": " Alpha "}) == {"id": None, "name": "Alpha", "is_bot": False}

    def test_control_characters_are_stripped(self):
        out = parse_turn_author({"id": "bot:\x00al\x1bpha\n", "name": "Al\tpha\r"})
        assert out["id"] == "bot:alpha"
        assert out["name"] == "Alpha"

    def test_format_characters_and_nbsp_survive(self):
        family = "\U0001F468\u200D\U0001F469\u200D\U0001F467"
        out = parse_turn_author({"id": "bot:alpha", "name": f"{family} Al\u00a0pha\u00a0"})
        assert out["name"] == f"{family} Al\u00a0pha"

    def test_oversize_fields_are_capped(self):
        out = parse_turn_author({"id": "x" * 500, "name": "y" * 201})
        assert len(out["id"]) == 200
        assert len(out["name"]) == 200


class TestEnvCarrier:
    def test_round_trip_through_env(self):
        author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
        env = turn_author_env(author)
        assert set(env) == {TURN_AUTHOR_ENV}
        assert turn_author_from_env(env) == author

    def test_env_json_is_compact(self):
        assert turn_author_env({"id": "a", "is_bot": True})[TURN_AUTHOR_ENV] == '{"id":"a","is_bot":true}'

    def test_absent_env_is_none(self):
        assert turn_author_from_env({}) is None

    def test_garbage_env_is_none(self):
        assert turn_author_from_env({TURN_AUTHOR_ENV: "{not json"}) is None

    def test_take_removes_the_variable(self):
        author = {"id": "bot:alpha", "name": "Alpha", "is_bot": True}
        env = dict(turn_author_env(author), OTHER="kept")
        assert take_turn_author_from_env(env) == author
        assert env == {"OTHER": "kept"}
        assert take_turn_author_from_env(env) is None


class TestMemoryScopeFor:
    def test_bot_with_id_gets_a2a_scope(self):
        assert scope_for({"id": "bot:alpha", "name": "Alpha", "is_bot": True}) == "a2a:bot:alpha"

    def test_human_has_no_scope(self):
        assert scope_for({"id": "user-1", "name": "Eri", "is_bot": False}) is None

    def test_bot_without_id_has_no_scope(self):
        assert scope_for({"id": None, "name": "Alpha", "is_bot": True}) is None

    @pytest.mark.parametrize("author", [None, {}, "bot:alpha", 3])
    def test_non_author_has_no_scope(self, author):
        assert scope_for(author) is None

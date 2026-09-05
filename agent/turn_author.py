"""Who wrote the user side of a turn, carried from the dispatcher to the recipient's memory hooks.

The dispatcher sets ``HERMES_TURN_AUTHOR`` on the recipient's one-shot subprocess only. A cached
gateway agent sees several authors over its lifetime, so the author is read per turn, never per agent.
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Any, Dict, Mapping, MutableMapping, Optional

TURN_AUTHOR_ENV = "HERMES_TURN_AUTHOR"

_MAX_FIELD_LEN = 200


_DROPPED_CATEGORIES = frozenset({"Cc", "Cs", "Cn", "Co"})
_TRUTHY = frozenset({"true", "1", "yes"})


def _clean_text(value: Any) -> Optional[str]:
    """Strip whitespace, control and unassigned characters, then cap the length. None when nothing is left.

    Format characters and non-breaking spaces stay so emoji sequences and display names survive."""
    if not isinstance(value, str):
        return None
    text = "".join(ch for ch in value if unicodedata.category(ch) not in _DROPPED_CATEGORIES).strip()
    if not text:
        return None
    return text[:_MAX_FIELD_LEN]


def _bot_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return isinstance(value, (bool, int)) and bool(value)


def parse_turn_author(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize a dict or JSON string into ``{"id", "name", "is_bot"}``. Never raises.

    None for anything else and for an author with neither id nor name. The id is whatever the
    transport knows the sender by: ``bot:<profile>`` on bot-mode deliveries, the platform user id
    on gateway platforms."""
    try:
        if isinstance(raw, (str, bytes)):
            raw = json.loads(raw)
        if not isinstance(raw, Mapping):
            return None
        author = {
            "id": _clean_text(raw.get("id")),
            "name": _clean_text(raw.get("name")),
            "is_bot": _bot_flag(raw.get("is_bot")),
        }
        if author["id"] is None and author["name"] is None:
            return None
        return author
    except Exception:
        return None


def turn_author_from_env(environ: Mapping[str, str] = os.environ) -> Optional[Dict[str, Any]]:
    """The author the dispatcher placed in ``HERMES_TURN_AUTHOR``, or None."""
    return parse_turn_author(environ.get(TURN_AUTHOR_ENV))


def take_turn_author_from_env(environ: MutableMapping[str, str] = os.environ) -> Optional[Dict[str, Any]]:
    """Read and remove ``HERMES_TURN_AUTHOR`` so subprocesses started during the turn do not inherit it."""
    return parse_turn_author(environ.pop(TURN_AUTHOR_ENV, None))


def scope_for(turn_author: Optional[Dict[str, Any]]) -> Optional[str]:
    """``a2a:<bot id>`` for a bot author with an id. None otherwise, so human turns keep the default scope."""
    if not isinstance(turn_author, dict) or not turn_author.get("is_bot"):
        return None
    author_id = turn_author.get("id")
    return f"a2a:{author_id}" if author_id else None


def turn_author_env(author: Dict[str, Any]) -> Dict[str, str]:
    """The environment entry a dispatcher merges into a child's env."""
    return {TURN_AUTHOR_ENV: json.dumps(author, separators=(",", ":"))}

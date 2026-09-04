"""Who wrote the user side of a turn, carried from the dispatcher to the recipient's memory hooks.

The dispatcher sets ``HERMES_TURN_AUTHOR`` on the recipient's one-shot subprocess only. A cached
gateway agent sees several authors over its lifetime, so the author is read per turn, never per agent.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

TURN_AUTHOR_ENV = "HERMES_TURN_AUTHOR"

_MAX_FIELD_LEN = 200


def _clean_text(value: Any) -> Optional[str]:
    """Strip whitespace and control characters, cap the length; None when nothing is left."""
    if not isinstance(value, str):
        return None
    text = "".join(ch for ch in value if ch.isprintable()).strip()
    if not text:
        return None
    return text[:_MAX_FIELD_LEN]


def parse_turn_author(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize a dict or JSON string into ``{"id", "name", "is_bot"}``; None for anything else. Never raises."""
    try:
        if isinstance(raw, (str, bytes)):
            raw = json.loads(raw)
        if not isinstance(raw, Mapping):
            return None
        return {
            "id": _clean_text(raw.get("id")),
            "name": _clean_text(raw.get("name")),
            "is_bot": bool(raw.get("is_bot")),
        }
    except Exception:
        return None


def turn_author_from_env(environ: Mapping[str, str] = os.environ) -> Optional[Dict[str, Any]]:
    """The author the dispatcher placed in ``HERMES_TURN_AUTHOR``, or None."""
    return parse_turn_author(environ.get(TURN_AUTHOR_ENV))


def scope_for(turn_author: Optional[Dict[str, Any]]) -> Optional[str]:
    """``a2a:<bot id>`` for a bot author with an id; None otherwise. Human turns keep the default scope."""
    if not isinstance(turn_author, dict) or not turn_author.get("is_bot"):
        return None
    author_id = turn_author.get("id")
    return f"a2a:{author_id}" if author_id else None


def turn_author_env(author: Dict[str, Any]) -> Dict[str, str]:
    """The environment entry a dispatcher merges into a child's env."""
    return {TURN_AUTHOR_ENV: json.dumps(author, separators=(",", ":"))}

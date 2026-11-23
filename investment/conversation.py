"""Bounded, caller-owned short-term conversation context; no persistent state."""

import json


MAX_EXCHANGES = 6
MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_CHARS = 12000


def _serialize(history: list[dict[str, str]]) -> str:
    return json.dumps(history, ensure_ascii=False, separators=(",", ":"))


def _clip(message: str) -> str:
    suffix = "\n[truncated]"
    if len(message) <= MAX_MESSAGE_CHARS:
        return message
    return message[:MAX_MESSAGE_CHARS - len(suffix)] + suffix


def bounded_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Copy the newest complete exchanges, trimming messages and oldest pairs."""
    result = [
        {"user": _clip(turn["user"]), "assistant": _clip(turn["assistant"])}
        for turn in (history or [])[-MAX_EXCHANGES:]
    ]
    while len(result) > 1 and len(_serialize(result)) > MAX_HISTORY_CHARS:
        result.pop(0)
    # JSON escapes can expand control characters beyond their message length.
    # Keep the newest exchange and shorten its larger field until it fits.
    while result and len(_serialize(result)) > MAX_HISTORY_CHARS:
        turn = result[0]
        field = max(turn, key=lambda key: len(json.dumps(turn[key], ensure_ascii=False)))
        turn[field] = turn[field][:len(turn[field]) // 2] + "\n[truncated]"
    return result


def add_exchange(history: list[dict[str, str]], question: str, answer: str) -> list[dict[str, str]]:
    return bounded_history([*history, {"user": question, "assistant": answer}])


def format_history(history: list[dict[str, str]] | None) -> str:
    bounded = bounded_history(history)
    return _serialize(bounded) if bounded else ""

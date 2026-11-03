"""Locale lookup: hint/fault keys -> text. Missing keys fall back to English, then the key."""
from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

import yaml


@cache
def _table(lang: str) -> dict[str, str]:
    path = resources.files("physiosight.i18n") / f"{lang}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown language {lang!r}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def available() -> list[str]:
    return sorted(p.name.removesuffix(".yaml") for p in resources.files("physiosight.i18n").iterdir()
                  if p.name.endswith(".yaml"))


def text(key: str, lang: str = "en", **values: Any) -> str:
    template = _table(lang).get(key) or _table("en").get(key) or key
    try:
        return template.format(**values)
    except (KeyError, ValueError, IndexError):
        return template

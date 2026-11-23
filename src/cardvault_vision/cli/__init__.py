"""Command-line interface.

Entry point: ``vision`` (installed as a console script) or ``python -m
cardvault_vision``.

Command groups follow the roadmap:

    vision dataset    validate | inspect | generate | statistics | split | bootstrap
    vision synthetic  generate | preview | validate | benchmark | profiles
    vision train      classifier | name-ocr | number-ocr
    vision test       classifier | name-ocr | number-ocr
    vision evaluate   evaluate | confusion | hard-examples
    vision benchmark  benchmark
    vision export     export | inspect-model | validate-export | registry
    vision layouts    layouts
    vision doctor     doctor
"""

from __future__ import annotations

from .main import main

__all__ = ["main"]

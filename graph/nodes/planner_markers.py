"""Compatibility shim: planner marker parser re-exported from planner.py.

Tests originally targeted a split-out `planner_markers` module. Until the
parser is actually extracted, this module forwards the public helper.
"""
from __future__ import annotations

from graph.nodes.planner import parse_planner_output

__all__ = ["parse_planner_output"]

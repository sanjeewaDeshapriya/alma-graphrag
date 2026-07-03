"""Test configuration.

Puts the project root on sys.path and forces the query parser into
deterministic regex-only mode (no LLM calls), so the suite runs offline and
produces identical results regardless of which API keys are in .env.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.crag.query_parser as query_parser

query_parser._client = None

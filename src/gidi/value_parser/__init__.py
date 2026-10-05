"""Deterministic value-span parser (experiment value-span-v8-deterministic-parser)."""

from gidi.value_parser.parser import Candidate, ValueSpan, find_candidates, parse_value

# Pinned by the ``value_parser.version`` of a bundle's config.json (gidi-finance-v3). Bump it for
# any change that can alter ``parse_value`` output; a bundle frozen with another version refuses
# to load rather than answer differently from what it was released with.
VERSION = "1"

__all__ = ["Candidate", "VERSION", "ValueSpan", "find_candidates", "parse_value"]

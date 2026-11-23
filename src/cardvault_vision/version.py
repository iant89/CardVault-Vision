"""Single source of truth for the package version."""

from __future__ import annotations

__version__ = "0.1.0"

#: Version of the on-disk annotation/manifest/export schemas. Bump the major
#: number whenever a change would invalidate previously written artifacts.
SCHEMA_VERSION = "1.0"

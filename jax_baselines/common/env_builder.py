"""Compatibility shim for the repo-local :mod:`env_builder` adapter.

Temporary migration debt: callers should move to the core env contract or the
repo-local adapter surface in a later cleanup slice.
"""

import sys

from env_builder import env_builder as _impl

sys.modules[__name__] = _impl

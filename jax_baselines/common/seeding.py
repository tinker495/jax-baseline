"""Compatibility shim for :mod:`env_builder.seeding`."""

import sys

from env_builder import seeding as _impl

sys.modules[__name__] = _impl

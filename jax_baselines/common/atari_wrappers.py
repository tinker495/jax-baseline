"""Compatibility shim for :mod:`env_builder.atari_wrappers`."""

import sys

from env_builder import atari_wrappers as _impl

sys.modules[__name__] = _impl

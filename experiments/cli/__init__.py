"""Experiment adapter command-line entry points.

Each module exposes ``main(argv=None)`` wired to a console script in
``pyproject.toml`` (``qnet``, ``dpg``, ...), so experiments run from
any working directory via ``uv run <family> ...``.
"""

"""Command-line entry points for jax_baselines runners.

Each module exposes ``main(argv=None)`` wired to a console script in
``pyproject.toml`` (``qnet``, ``dpg``, ...), so experiments run from
any working directory via ``uv run <family> ...``.
"""

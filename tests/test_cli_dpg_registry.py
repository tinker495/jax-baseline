"""Tests for the CLI family runner registry (``experiments/cli/_run.py``).

Covers the local (env-based) families that ride ``run_family``: dpg, pg, qnet.
Pinned behaviors:
- Each ``*_RUNNER.algos`` registers exactly its family's algorithms.
- Each algo's ``build()`` returns only kwargs accepted by the agent constructor
  (the base classes have explicit params and no ``**kwargs``, so a misnamed
  kwarg would raise ``TypeError`` at construction — this is a real check).
- ``resolve_maker`` resolves every flax builder and raises ``SystemExit`` for
  unsupported combos (CrossQ+haiku, BBF+haiku).
- dpg: simba/simbav2 variants, the TD7 isolation invariant, flag renames.
- qnet: ``--hl_gauss`` selects the HL_GAUSS_* class via ``AlgoSpec.resolve_cls``.
- ``run_family`` wires env/maker/policy_kwargs into the agent and calls
  ``learn()`` + ``test()`` without constructing any real env or JAX model.
"""

from __future__ import annotations

import importlib
import inspect
from argparse import ArgumentParser

import jax.numpy as jnp
import numpy as np
import optax
import pytest

# ---------------------------------------------------------------------------
# Runners under test
# ---------------------------------------------------------------------------

EXPECTED_ALGOS = {
    "dpg": {"DDPG", "TD3", "SAC", "CrossQ", "TQC", "TD7"},
    "pg": {"A2C", "PPO", "TPPO", "SPO"},
    "qnet": {"DQN", "C51", "QRDQN", "IQN", "FQF", "SPR", "BBF"},
}
LOCAL_FAMILIES = sorted(EXPECTED_ALGOS)


def _runner(name: str):
    if name == "dpg":
        from experiments.cli.dpg import DPG_RUNNER

        return DPG_RUNNER
    if name == "pg":
        from experiments.cli.pg import PG_RUNNER

        return PG_RUNNER
    if name == "qnet":
        from experiments.cli.qnet import QNET_RUNNER

        return QNET_RUNNER
    raise AssertionError(name)


def _parse(runner, argv: list[str]):
    p = ArgumentParser()
    runner.add_args(p)
    return p.parse_args(argv)


def _accepted_kwargs(cls: type) -> set[str]:
    """Union of POSITIONAL_OR_KEYWORD and KEYWORD_ONLY params over the MRO."""
    kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    names: set[str] = set()
    for klass in cls.__mro__:
        try:
            sig = inspect.signature(klass.__init__)
        except (ValueError, TypeError):
            continue
        for name, param in sig.parameters.items():
            if name != "self" and param.kind in kinds:
                names.add(name)
    return names


# ---------------------------------------------------------------------------
# Generic checks across every local family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", LOCAL_FAMILIES)
def test_algos_registered(family: str):
    assert set(_runner(family).algos) == EXPECTED_ALGOS[family]


@pytest.mark.parametrize("family", LOCAL_FAMILIES)
def test_build_kwargs_are_accepted(family: str):
    runner = _runner(family)
    for algo, spec in runner.algos.items():
        for extra in ([], ["--hl_gauss"] if family == "qnet" else []):
            args = _parse(runner, ["--algo", algo, *extra])
            built = spec.build(args)
            cls = spec.resolve_cls(args)
            stray = set(built) - _accepted_kwargs(cls)
            assert not stray, f"{family}/{algo} build() not in {cls.__name__}: {stray}"


@pytest.mark.parametrize("family", LOCAL_FAMILIES)
def test_resolver_resolves_flax_base(family: str):
    from experiments.cli._run import resolve_maker

    runner = _runner(family)
    for algo, spec in runner.algos.items():
        args = _parse(runner, ["--algo", algo, "--model_lib", "flax"])
        assert callable(resolve_maker(runner, spec, args))


# ---------------------------------------------------------------------------
# dpg-specific
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [["--simba"], ["--simbav2"]])
def test_dpg_simba_variants_resolve(flags: list[str]):
    from experiments.cli._run import resolve_maker
    from experiments.cli.dpg import DPG_RUNNER

    for algo, spec in DPG_RUNNER.algos.items():
        args = _parse(DPG_RUNNER, ["--algo", algo, "--model_lib", "flax", *flags])
        assert callable(resolve_maker(DPG_RUNNER, spec, args))


def test_dpg_crossq_haiku_is_unsupported_clean_error():
    from experiments.cli._run import resolve_maker
    from experiments.cli.dpg import DPG_RUNNER

    args = _parse(DPG_RUNNER, ["--algo", "CrossQ", "--model_lib", "haiku"])
    with pytest.raises(SystemExit):
        resolve_maker(DPG_RUNNER, DPG_RUNNER.algos["CrossQ"], args)


def test_dpg_build_threads_renamed_args():
    from experiments.cli.dpg import DPG_RUNNER

    args = _parse(DPG_RUNNER, ["--algo", "TQC", "--mixture", "wang", "--batch", "64", "--per"])
    built = DPG_RUNNER.algos["TQC"].build(args)
    assert built["mixture_type"] == "wang"
    assert built["batch_size"] == 64
    assert built["prioritized_replay"] is True
    assert built["simba"] is False


def test_dpg_td7_build_omits_forced_internal_defaults():
    """TD7.__init__ forces n_step=1, target_network_update_tau=0,
    prioritized_replay=True and use_checkpointing=True (caller kwargs win), so
    its build() must NOT pass those keys, or the forced values get overridden.
    """
    from experiments.cli.dpg import DPG_RUNNER

    built = DPG_RUNNER.algos["TD7"].build(_parse(DPG_RUNNER, ["--algo", "TD7"]))
    for forced in (
        "prioritized_replay",
        "use_checkpointing",
        "n_step",
        "target_network_update_tau",
    ):
        assert forced not in built, f"TD7 build() must not pass {forced}: TD7 forces it internally"


# ---------------------------------------------------------------------------
# qnet-specific
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "algo,base,hl",
    [
        ("C51", "C51", "HL_GAUSS_C51"),
        ("SPR", "SPR", "HL_GAUSS_SPR"),
        ("BBF", "BBF", "HL_GAUSS_BBF"),
    ],
)
def test_qnet_hl_gauss_selects_class(algo: str, base: str, hl: str):
    from experiments.cli.qnet import QNET_RUNNER

    spec = QNET_RUNNER.algos[algo]
    assert spec.resolve_cls(_parse(QNET_RUNNER, ["--algo", algo])).__name__ == base
    assert spec.resolve_cls(_parse(QNET_RUNNER, ["--algo", algo, "--hl_gauss"])).__name__ == hl


def test_qnet_bbf_optimizer_policy_preserves_plain_bbf_cli_optimizer(monkeypatch):
    import experiments.cli.qnet as qnet_cli

    calls = []

    def fake_make_optimizer_factory(optimizer_name, **kwargs):
        calls.append((optimizer_name, kwargs))
        return lambda learning_rate: (optimizer_name, kwargs, learning_rate)

    monkeypatch.setattr(qnet_cli, "make_optimizer_factory", fake_make_optimizer_factory)

    args = _parse(qnet_cli.QNET_RUNNER, ["--algo", "BBF", "--optimizer", "sgd"])
    built = qnet_cli.QNET_RUNNER.algos["BBF"].build(args)

    assert calls == [("sgd", {"weight_decay": 0.1})]
    assert built["optimizer_factory"](0.25) == ("sgd", {"weight_decay": 0.1}, 0.25)


def test_qnet_hl_gauss_bbf_optimizer_policy_matches_historical_adamw_default():
    from experiments.cli.qnet import QNET_RUNNER

    args = _parse(
        QNET_RUNNER,
        ["--algo", "BBF", "--hl_gauss", "--optimizer", "sgd"],
    )
    built = QNET_RUNNER.algos["BBF"].build(args)
    optimizer = built["optimizer_factory"](0.25)
    reference = optax.adamw(learning_rate=0.25, weight_decay=0.1)
    params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
    grads = {"w": jnp.array([0.5, -0.25], dtype=jnp.float32)}

    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    expected, _ = reference.update(grads, reference.init(params), params)

    np.testing.assert_allclose(updates["w"], expected["w"], rtol=1e-6)


def test_qnet_bbf_haiku_is_unsupported_clean_error():
    from experiments.cli._run import resolve_maker
    from experiments.cli.qnet import QNET_RUNNER

    args = _parse(QNET_RUNNER, ["--algo", "BBF", "--model_lib", "haiku"])
    with pytest.raises(SystemExit):
        resolve_maker(QNET_RUNNER, QNET_RUNNER.algos["BBF"], args)


# ---------------------------------------------------------------------------
# run_family wiring (no real env / JAX model)
# ---------------------------------------------------------------------------


def test_run_family_wires_agent_without_env_or_model(monkeypatch):
    from dataclasses import replace

    import experiments.cli._run as run_mod
    from experiments.cli.dpg import DPG_RUNNER
    from experiments.runtime_adapters import TensorboardLogger

    captured: dict = {}

    class FakeAgent:
        def __init__(self, env_builder, maker, **kwargs):
            captured.update(env=env_builder, maker=maker, kwargs=kwargs)

        def learn(
            self,
            steps,
            experiment_name=None,
            eval_num=100,
            logger_factory=None,
            progress_factory=None,
            record_test_fn=None,
        ):
            captured.update(
                steps=steps,
                exp=experiment_name,
                eval_num=eval_num,
                logger_factory=logger_factory,
                progress_factory=progress_factory,
                record_test_fn=record_test_fn,
            )

        def test(self):
            captured["tested"] = True

    fake_spec = replace(DPG_RUNNER.algos["DDPG"], cls=FakeAgent)
    fake_runner = replace(
        DPG_RUNNER,
        algos={**DPG_RUNNER.algos, "DDPG": fake_spec},
        build_env=lambda a: ("ENVB", {"pk": 1}),
    )
    monkeypatch.setattr(run_mod, "resolve_maker", lambda r, s, a: "MAKER")

    run_mod.run_family(fake_runner, ["--algo", "DDPG", "--steps", "7"])

    assert captured["steps"] == 7
    assert captured["eval_num"] == 100
    assert captured["logger_factory"] is TensorboardLogger
    assert captured["progress_factory"] is run_mod.make_progress
    assert captured["record_test_fn"] is run_mod.record_and_test
    assert captured["maker"] == "MAKER"
    assert captured["env"] == "ENVB"
    assert captured["kwargs"]["policy_kwargs"] == {"pk": 1}
    assert captured["tested"] is True


@pytest.mark.parametrize(
    ("module_name", "runner_name", "runner_func_name"),
    [
        ("qnet", "QNET_RUNNER", "run_family"),
        ("dpg", "DPG_RUNNER", "run_family"),
        ("pg", "PG_RUNNER", "run_family"),
        ("impala", "IMPALA_RUNNER", "run_distributed_family"),
        ("apex_qnet", "APEX_QNET_RUNNER", "run_distributed_family"),
        ("apex_dpg", "APEX_DPG_RUNNER", "run_distributed_family"),
    ],
)
def test_console_main_returns_zero_after_success(
    monkeypatch,
    module_name,
    runner_name,
    runner_func_name,
):
    mod = importlib.import_module(f"experiments.cli.{module_name}")
    captured = {}

    def fake_runner(runner, argv=None):
        captured["runner"] = runner
        captured["argv"] = argv
        return object()

    monkeypatch.setattr(mod, runner_func_name, fake_runner)

    assert mod.main(["--sentinel"]) == 0
    assert captured == {"runner": getattr(mod, runner_name), "argv": ["--sentinel"]}


# ---------------------------------------------------------------------------
# Distributed families (ray-based): impala / apex_dpg / apex_qnet
# ---------------------------------------------------------------------------

EXPECTED_DIST_ALGOS = {
    "impala": {"A2C", "PPO", "TPPO", "SPO"},
    "apex_dpg": {"DDPG", "TD3"},
    "apex_qnet": {"DQN", "C51", "QRDQN", "IQN"},
}
DIST_FAMILIES = sorted(EXPECTED_DIST_ALGOS)


def _dist_runner(name: str):
    if name == "impala":
        from experiments.cli.impala import IMPALA_RUNNER

        return IMPALA_RUNNER
    if name == "apex_dpg":
        from experiments.cli.apex_dpg import APEX_DPG_RUNNER

        return APEX_DPG_RUNNER
    if name == "apex_qnet":
        from experiments.cli.apex_qnet import APEX_QNET_RUNNER

        return APEX_QNET_RUNNER
    raise AssertionError(name)


@pytest.mark.parametrize("family", DIST_FAMILIES)
def test_dist_algos_registered(family: str):
    assert set(_dist_runner(family).algos) == EXPECTED_DIST_ALGOS[family]


@pytest.mark.parametrize("family", DIST_FAMILIES)
def test_dist_build_kwargs_are_accepted(family: str):
    runner = _dist_runner(family)
    for algo, spec in runner.algos.items():
        args = _parse(runner, ["--algo", algo])
        built = spec.build(args)
        cls = spec.resolve_cls(args)
        stray = set(built) - _accepted_kwargs(cls)
        assert not stray, f"{family}/{algo} build() not in {cls.__name__}: {stray}"


@pytest.mark.parametrize("family", DIST_FAMILIES)
def test_dist_resolver_resolves_flax_base(family: str):
    from experiments.cli._run import resolve_maker

    runner = _dist_runner(family)
    for algo, spec in runner.algos.items():
        args = _parse(runner, ["--algo", algo, "--model_lib", "flax"])
        assert callable(resolve_maker(runner, spec, args))


@pytest.mark.parametrize("family", DIST_FAMILIES)
def test_dist_policy_kwargs_uses_shared_normal_embedding(family: str):
    """All distributed families share the Atari ``normal`` embedding policy.

    Pins the consolidation of three byte-identical ``policy_kwargs`` helpers onto
    ``_run.default_policy_kwargs`` so the embedding mode cannot silently drift per
    family.
    """
    from experiments.cli._run import default_policy_kwargs

    runner = _dist_runner(family)
    assert runner.policy_kwargs is default_policy_kwargs
    args = _parse(runner, ["--algo", sorted(runner.algos)[0], "--node", "128", "--hidden_n", "3"])
    assert runner.policy_kwargs(args) == {
        "node": 128,
        "hidden_n": 3,
        "embedding_mode": "normal",
    }


def test_run_distributed_family_wires_agent(monkeypatch):
    import types
    from dataclasses import replace

    import experiments.cli._run as run_mod
    from experiments.cli.apex_dpg import APEX_DPG_RUNNER
    from experiments.runtime_adapters import TensorboardLogger

    captured: dict = {}

    class FakeAgent:
        def __init__(self, workers, maker, manager, **kwargs):
            captured.update(workers=workers, maker=maker, manager=manager, kwargs=kwargs)

        def learn(self, steps, experiment_name=None, logger_factory=None, progress_factory=None):
            captured.update(
                steps=steps,
                experiment_name=experiment_name,
                logger_factory=logger_factory,
                progress_factory=progress_factory,
            )

    fake_spec = replace(APEX_DPG_RUNNER.algos["DDPG"], cls=FakeAgent)
    fake_runner = replace(
        APEX_DPG_RUNNER,
        algos={**APEX_DPG_RUNNER.algos, "DDPG": fake_spec},
        make_workers=lambda a: ["W0", "W1"],
        policy_kwargs=lambda a: {"pk": 1},
    )
    monkeypatch.setattr(run_mod, "resolve_maker", lambda r, s, a: "MAKER")
    monkeypatch.setattr("ray.init", lambda **k: None)
    monkeypatch.setattr(
        "multiprocessing.get_context",
        lambda *a: types.SimpleNamespace(Manager=lambda: "MGR"),
    )

    run_mod.run_distributed_family(fake_runner, ["--algo", "DDPG", "--steps", "11"])

    assert captured["workers"] == ["W0", "W1"]
    assert captured["maker"] == "MAKER"
    assert captured["manager"] == "MGR"
    assert captured["steps"] == 11
    assert captured["experiment_name"] == "APEX_DPG"  # threaded from --experiment_name default
    assert captured["logger_factory"] is TensorboardLogger
    assert captured["progress_factory"] is run_mod.make_progress
    assert captured["kwargs"]["policy_kwargs"] == {"pk": 1}

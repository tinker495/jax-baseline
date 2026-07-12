"""Issue #7 behavioral seam regressions."""

from __future__ import annotations

import ast
import inspect
from argparse import ArgumentParser
from pathlib import Path

import jax
import numpy as np
import pytest


def _parse(runner, argv: list[str]):
    parser = ArgumentParser()
    runner.add_args(parser)
    return parser.parse_args(argv)


def _local_specs():
    from experiments.cli.dpg import DPG_RUNNER
    from experiments.cli.qnet import QNET_RUNNER

    for runner in (QNET_RUNNER, DPG_RUNNER):
        for algo, spec in runner.algos.items():
            variants = [[]]
            if runner is QNET_RUNNER and algo in {"C51", "SPR", "BBF"}:
                variants.append(["--hl_gauss"])
            for extra in variants:
                args = _parse(runner, ["--algo", algo, *extra])
                yield spec.resolve_cls(args)


def _distributed_specs():
    from experiments.cli.apex_dpg import APEX_DPG_RUNNER
    from experiments.cli.apex_qnet import APEX_QNET_RUNNER
    from experiments.cli.impala import IMPALA_RUNNER

    for runner in (APEX_QNET_RUNNER, APEX_DPG_RUNNER, IMPALA_RUNNER):
        for algo, spec in runner.algos.items():
            args = _parse(runner, ["--algo", algo])
            yield runner, algo, spec, args, spec.resolve_cls(args)


@pytest.mark.parametrize("cls", list(_local_specs()), ids=lambda cls: cls.__name__)
def test_local_learn_contract_accepts_driver_factories(cls):
    inspect.signature(cls.learn).bind_partial(
        cls.__new__(cls),
        0,
        logger_factory=None,
        progress_factory=None,
        record_test_fn=None,
    )


@pytest.mark.parametrize(
    "_runner,_algo,_spec,_args,cls",
    list(_distributed_specs()),
    ids=lambda item: item.__name__ if isinstance(item, type) else str(item),
)
def test_distributed_learn_contract_accepts_driver_factories(_runner, _algo, _spec, _args, cls):
    inspect.signature(cls.learn).bind_partial(
        cls.__new__(cls),
        0,
        callback=None,
        reset_num_timesteps=True,
        replay_wrapper=None,
        logger_factory=None,
        progress_factory=None,
    )


@pytest.mark.parametrize("cls", list(_local_specs()), ids=lambda cls: cls.__name__)
def test_local_learn_inherits_family_driver_factories(cls):
    assert "learn" not in cls.__dict__
    source = inspect.getsource(cls.learn)
    assert "TrainingSession().run(" in source
    assert "logger_factory" in inspect.signature(cls.learn).parameters
    assert "progress_factory" in inspect.signature(cls.learn).parameters
    assert "record_test_fn" in inspect.signature(cls.learn).parameters


@pytest.mark.parametrize(
    "_runner,_algo,_spec,_args,cls",
    list(_distributed_specs()),
    ids=lambda item: item.__name__ if isinstance(item, type) else str(item),
)
def test_distributed_learn_inherits_family_driver_factories(_runner, _algo, _spec, _args, cls):
    assert "learn" not in cls.__dict__
    source = inspect.getsource(cls.learn)
    # The resolved family entrypoint must still consume the injected factories
    # rather than hardcoding the distributed runtime.
    assert "logger_factory" in inspect.signature(cls.learn).parameters
    assert "progress_factory" in inspect.signature(cls.learn).parameters
    assert "experiment_name" in inspect.signature(cls.learn).parameters
    assert "create_logger_server(" in source
    assert "logger_factory" in source
    assert "progress_factory" in source
    assert "experiment_name" in source


@pytest.mark.parametrize(
    "runner,algo,spec,args,cls",
    list(_distributed_specs()),
    ids=lambda item: item.__name__ if isinstance(item, type) else str(item),
)
def test_distributed_constructor_kwargs_are_explicit_and_reject_unknown(
    runner, algo, spec, args, cls
):
    signature = inspect.signature(cls)
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    ), f"{cls.__name__} must not absorb unknown constructor kwargs"

    built = spec.build(args)
    signature.bind_partial(
        [], lambda *a, **k: None, object(), policy_kwargs=runner.policy_kwargs(args), **built
    )
    with pytest.raises(TypeError):
        signature.bind_partial([], lambda *a, **k: None, object(), typo_unknown=True)


@pytest.mark.parametrize("algo", ["DDPG", "TD3"])
def test_apex_dpg_constructors_initialize_model_setup_dependencies(monkeypatch, algo):
    from experiments.cli.apex_dpg import APEX_DPG_RUNNER

    class _Worker:
        def get_info(self):
            return {
                "observation_space": [[3]],
                "action_size": [1],
                "action_type": "continuous",
                "env_type": "single",
                "env_id": "FakeContinuous-v0",
                "worker_num": 1,
                "core_env_type": "SingleEnv",
            }

    class _Runtime:
        def replay_manager(self):
            return None

        def worker_info(self, worker):
            return worker.get_info()

    class _Replay:
        def get_buffer_info(self):
            return object()

    def _multi_replay_factory(_need):
        return _Replay()

    def _worker_replay_factory(*_args, **_kwargs):
        return object()

    def _optimizer_factory(_learning_rate):
        class _Optimizer:
            def init(self, params):
                return ("opt_state", params)

        return _Optimizer()

    def _model_builder_maker(*_args, **_kwargs):
        def _builder(*_builder_args, **_builder_kwargs):
            return "preproc", "actor", "critic", {"params": 1}

        return _builder

    args = _parse(APEX_DPG_RUNNER, ["--algo", algo])
    spec = APEX_DPG_RUNNER.algos[algo]
    built = spec.build(args)
    built.update(
        {
            "multi_replay_factory": _multi_replay_factory,
            "worker_replay_factory": _worker_replay_factory,
            "optimizer_factory": _optimizer_factory,
        }
    )

    agent = spec.resolve_cls(args)([_Worker()], _model_builder_maker, _Runtime(), **built)

    assert agent.actor_builder is not None
    if algo == "TD3":
        assert agent.action_noise_clamp == 0.5
        assert agent.target_action_noise == pytest.approx(agent.action_noise * 1.5)


def test_apex_c51_worker_priority_uses_n_step_gamma():
    source = Path("jax_baselines/C51/apex_c51.py").read_text()
    tree = ast.parse(source)
    get_actor_builder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_actor_builder"
    )
    gamma_assignment = next(
        node
        for node in get_actor_builder.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "gamma" for target in node.targets)
    )
    assert isinstance(gamma_assignment.value, ast.Attribute)
    assert gamma_assignment.value.attr == "_gamma"


def test_flax_ac_continuous_actor_initializes_log_std_param():
    from model_builder.flax.ac.ac_builder import model_builder_maker

    builder = model_builder_maker(
        [[3]],
        [1],
        "continuous",
        {"node": 8, "hidden_n": 1, "embedding_mode": "normal"},
    )
    _preproc, actor, critic, params = builder(jax.random.PRNGKey(0))

    obs = [np.zeros((1, 3), dtype=np.float32)]
    mu, log_std = actor(params, None, obs)
    value = critic(params, None, obs)

    assert mu.shape[-1] == 1
    assert log_std.shape == (1, 1)
    assert value.shape[-1] == 1

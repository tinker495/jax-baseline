from argparse import ArgumentParser

import pytest
import yaml

from experiments.cli import exp
from experiments.cli.pg import PG_RUNNER
from jax_baselines.A2C import base_class as ac_base
from jax_baselines.A2C.base_class import Actor_Critic_Policy_Gradient_Family


def _parse(argv):
    parser = ArgumentParser()
    PG_RUNNER.add_args(parser)
    return parser.parse_args(argv)


def _build(algo, extra=None):
    args = _parse(["--algo", algo, *(extra or [])])
    return PG_RUNNER.algos[algo].build(args)


def test_pg_cli_preserves_ppo2_control_defaults():
    a2c = _build("A2C")
    ppo = _build("PPO")
    tppo = _build("TPPO")

    for built in (a2c, ppo, tppo):
        assert built["optimizer_eps"] == pytest.approx(1e-2 / 256.0)
        assert built["max_grad_norm"] is None
        assert built["lr_annealing"] is False

    assert ppo["ppo_eps"] == pytest.approx(0.2)
    assert ppo["value_clip"] == pytest.approx(2.0)
    assert tppo["value_clip"] == pytest.approx(2.0)
    assert "ppo_eps" not in tppo
    assert "ppo_eps" not in a2c
    assert "value_clip" not in a2c


@pytest.mark.parametrize(
    "algo,expected_clip_keys",
    [
        ("A2C", set()),
        ("PPO", {"ppo_eps", "value_clip"}),
        ("TPPO", {"value_clip"}),
        ("SPO", {"ppo_eps", "value_clip"}),
    ],
)
def test_pg_cli_routes_ppo_clip_kwargs_by_constructor(algo, expected_clip_keys):
    built = _build(
        algo,
        [
            "--optimizer_eps",
            "1e-5",
            "--max_grad_norm",
            "0.5",
            "--lr_annealing",
            "--ppo_eps",
            "0.1",
            "--value_clip",
            "0.1",
        ],
    )

    assert built["optimizer_eps"] == pytest.approx(1e-5)
    assert built["max_grad_norm"] == pytest.approx(0.5)
    assert built["lr_annealing"] is True
    assert {key for key in ("ppo_eps", "value_clip") if key in built} == expected_clip_keys
    if "ppo_eps" in expected_clip_keys:
        assert built["ppo_eps"] == pytest.approx(0.1)
    if "value_clip" in expected_clip_keys:
        assert built["value_clip"] == pytest.approx(0.1)


def test_pg_breakout_config_emits_ppo2_atari_flags_parseable_by_pg_cli():
    with open("experiments/configs/pg_breakout.yaml") as handle:
        config = yaml.safe_load(handle)

    commands = list(exp._iter_commands(config))
    by_algo = {}
    for command in commands:
        assert command[0] == "pg"
        args = _parse(command[1:])
        by_algo[args.algo] = (args, command)

    assert set(by_algo) == {"SPO", "TPPO", "PPO"}
    for args, command in by_algo.values():
        assert args.gamma == pytest.approx(0.99)
        assert args.learning_rate == pytest.approx(0.00025)
        assert args.optimizer == "adam"
        assert args.optimizer_eps == pytest.approx(1e-5)
        assert args.max_grad_norm == pytest.approx(0.5)
        assert args.lr_annealing is True
        assert args.ent_coef == pytest.approx(0.01)
        assert args.val_coef == pytest.approx(0.5)
        assert args.epoch_num == 4
        assert args.value_clip == pytest.approx(0.1)
        assert args.gae_normalize is True
        assert args.gae_normalize_scope == "minibatch"
        assert "--value_clip" in command
        assert "--lr_annealing" in command

    assert by_algo["PPO"][0].ppo_eps == pytest.approx(0.1)
    assert by_algo["SPO"][0].ppo_eps == pytest.approx(0.1)
    assert "--ppo_eps" in by_algo["PPO"][1]
    assert "--ppo_eps" in by_algo["SPO"][1]
    assert "--ppo_eps" not in by_algo["TPPO"][1]


def test_ac_base_optimizer_threads_eps_and_global_grad_clip(monkeypatch):
    calls = []

    def fake_select_optimizer(optim_str, lr, eps, grad_max):
        calls.append({"optim_str": optim_str, "lr": lr, "eps": eps, "grad_max": grad_max})
        return object()

    monkeypatch.setattr(ac_base, "select_optimizer", fake_select_optimizer)
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.optimizer_name = "adam"
    agent.optimizer_eps = 1e-5
    agent.max_grad_norm = 0.5

    optimizer = agent._make_optimizer(0.00025)

    assert optimizer is not None
    assert calls == [
        {
            "optim_str": "adam",
            "lr": 0.00025,
            "eps": pytest.approx(1e-5),
            "grad_max": pytest.approx(0.5),
        }
    ]


def test_prepare_run_rebuilds_optimizer_with_linear_lr_schedule(monkeypatch):
    calls = []

    class FakeOptimizer:
        def init(self, params):
            calls.append({"init_params": params})
            return {"state": "reset"}

    def fake_select_optimizer(optim_str, lr, eps, grad_max):
        calls.append({"optim_str": optim_str, "lr": lr, "eps": eps, "grad_max": grad_max})
        return FakeOptimizer()

    monkeypatch.setattr(ac_base, "select_optimizer", fake_select_optimizer)
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.optimizer_name = "adam"
    agent.learning_rate = 0.00025
    agent.optimizer_eps = 1e-5
    agent.max_grad_norm = 0.5
    agent.lr_annealing = True
    agent.batch_size = 10
    agent.worker_size = 2
    agent.minibatch_size = 5
    agent.epoch_num = 3
    agent.params = {"w": 1.0}
    agent.opt_state = {"state": "old"}

    agent.prepare_run(100)

    schedule = calls[0]["lr"]
    assert agent._lr_annealing_transition_steps(100) == 60
    assert callable(schedule)
    assert float(schedule(0)) == pytest.approx(0.00025)
    assert float(schedule(30)) == pytest.approx(0.000125)
    assert float(schedule(60)) == pytest.approx(0.0)
    assert calls[0]["eps"] == pytest.approx(1e-5)
    assert calls[0]["grad_max"] == pytest.approx(0.5)
    assert calls[1] == {"init_params": {"w": 1.0}}
    assert agent.opt_state == {"state": "reset"}


def test_prepare_run_skips_lr_annealing_until_params_exist(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ac_base,
        "select_optimizer",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.lr_annealing = True
    agent.params = None

    agent.prepare_run(100)

    assert calls == []


def test_pg_breakout_lr_annealing_transition_counts_optimizer_updates():
    with open("experiments/configs/pg_breakout.yaml") as handle:
        config = yaml.safe_load(handle)
    ppo_command = [
        command
        for command in exp._iter_commands(config)
        if "--algo" in command and command[command.index("--algo") + 1] == "PPO"
    ][0]
    args = _parse(ppo_command[1:])
    built = PG_RUNNER.algos["PPO"].build(args)
    agent = Actor_Critic_Policy_Gradient_Family.__new__(Actor_Critic_Policy_Gradient_Family)
    agent.batch_size = built["batch_size"]
    agent.worker_size = args.worker
    agent.minibatch_size = built["minibatch_size"]
    agent.epoch_num = built["epoch_num"]

    assert agent._lr_annealing_transition_steps(int(float(args.steps))) == 468736

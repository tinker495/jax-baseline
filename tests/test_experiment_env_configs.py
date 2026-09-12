import tomllib
from pathlib import Path

import pytest
import yaml

from experiments.cli.exp import _iter_commands


def test_simulator_extra_is_optional_and_old_stacks_are_removed():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert extras["mjlab"] == ["mjlab>=1.6,<1.7; python_version < '3.14'"]
    assert "mjlab" not in extras["all"]
    assert "mjx" not in extras and "isaaclab" not in extras


@pytest.mark.parametrize(
    ("filename", "runner", "algorithms"),
    [("pg_mjlab_go1.yaml", "pg", ["PPO", "SPO"]), ("dpg_mjlab_go1.yaml", "dpg", ["TQC"])],
)
def test_adapter_configs_preserve_actor_and_critic_observations(filename, runner, algorithms):
    path = Path("experiments/configs/mjlab") / filename
    config = yaml.safe_load(path.read_text())
    expected = {
        "env": "Mjlab-Velocity-Flat-Unitree-Go1",
        "env_backend": "mjlab",
        "env_device": "cuda:0",
    }
    assert config["category"] == "mjlab"
    assert config["runner"] == runner
    assert config["variants"] == [{"algo": algo, "enabled": True} for algo in algorithms]
    assert config["base"] | expected == config["base"]
    assert config["base"]["obs_rms_norm"] is True
    assert "env_episode_length" not in config["base"]
    assert "env_observation_key" not in config["base"]

    command = next(_iter_commands(config, config_dir=path.parent))
    assert command[0] == runner
    assert "--obs_rms_norm" in command
    assert "--env_observation_key" not in command
    for key, value in expected.items():
        index = command.index(f"--{key}")
        assert command[index + 1] == value

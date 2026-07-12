import yaml

from experiments.cli import exp


def test_set_overrides_parse_yaml_scalars(tmp_path, capsys):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "family": "qnet",
                "base": {"clip_rewards": True, "batch_size": 32},
                "variants": [{}],
            }
        )
    )

    assert (
        exp.main(
            [
                str(config_path),
                "--dry-run",
                "--set",
                "clip_rewards=false",
                "--set",
                "batch_size=64",
                "--set",
                "dueling_model=true",
            ]
        )
        == 0
    )

    assert capsys.readouterr().out.strip() == "[1/1] qnet --batch_size 64 --dueling_model"

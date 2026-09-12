import yaml

from experiments.cli import exp


def test_set_overrides_parse_yaml_scalars(tmp_path, capsys):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "category": "atari",
                "runner": "qnet",
                "base": {"double": True, "batch": 32},
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
                "double=false",
                "--set",
                "batch=64",
                "--set",
                "dueling=true",
            ]
        )
        == 0
    )

    assert capsys.readouterr().out.strip() == ("category: atari\n[1/1] qnet --batch 64 --dueling")

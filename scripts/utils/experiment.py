from pathlib import Path
import json
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_experiment_output_dir(config):
    phase = config["experiment"]["phase"]
    name = config["experiment"]["name"]

    return (
        PROJECT_ROOT
        / "outputs"
        / phase
        / name
    )


def prepare_experiment_directory(config):

    output_dir = get_experiment_output_dir(
        config
    )

    directories = [
        output_dir,
        output_dir / "checkpoints",
        output_dir / "logs",
        output_dir / "config",
        output_dir / "reconstructions",
        output_dir / "evaluation",
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    save_config(
        config,
        output_dir,
    )

    return output_dir


def save_config(config, output_dir):

    config_dir = (
        output_dir / "config"
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    yaml_path = (
        config_dir / "config.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    json_path = (
        config_dir / "config.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )
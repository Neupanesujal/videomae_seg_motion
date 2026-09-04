from pathlib import Path
import copy
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "base"
    / "pretraining.yaml"
)

EXPERIMENT_CONFIG_DIR = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
)


def load_yaml(path):
    path = Path(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, override):
    """
    Recursively merge override configuration
    into the base configuration.
    """

    result = copy.deepcopy(base)

    for key, value in override.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )

        else:
            result[key] = copy.deepcopy(value)

    return result


def load_experiment_config(experiment_name):
    """
    Load the common base configuration and
    merge the experiment-specific configuration.
    """

    base_config = load_yaml(
        BASE_CONFIG
    )

    experiment_path = (
        EXPERIMENT_CONFIG_DIR
        / f"{experiment_name}.yaml"
    )

    if not experiment_path.exists():
        raise FileNotFoundError(
            f"Experiment configuration not found:\n"
            f"{experiment_path}"
        )

    experiment_config = load_yaml(
        experiment_path
    )

    config = deep_merge(
        base_config,
        experiment_config,
    )

    return config


def resolve_path(path):
    """
    Convert a project-relative path into
    an absolute path.
    """

    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path
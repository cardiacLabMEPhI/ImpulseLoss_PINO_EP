"""Run configured P2P and rollout evaluations from evaluation_config.yaml."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "evaluation_config.yaml"


def normalise_types(config):
    """Return evaluation methods and data modes from the flexible config format."""
    configured = config.get("eval_types", ["p2p", "rollout"])
    if isinstance(configured, str):
        configured = [item.strip() for item in configured.split(",") if item.strip()]
    configured = [str(item).lower() for item in configured]

    methods = [item for item in configured if item in {"p2p", "rollout"}]
    data_modes = [item for item in configured if item in {"full", "sample"}]

    # Current project configs use full/sample for data selection while both
    # evaluation methods are expected. A config with p2p/rollout selects methods.
    if not methods:
        methods = ["p2p", "rollout"]
    if not data_modes:
        data_modes = list(config.get("data_eval_types", ["full", "sample"]))

    invalid = set(configured) - {"p2p", "rollout", "full", "sample"}
    if invalid:
        raise ValueError(f"Unknown evaluation types in evaluation_config.yaml: {sorted(invalid)}")
    if not data_modes or any(item not in {"full", "sample"} for item in data_modes):
        raise ValueError("data_eval_types must contain only 'full' and/or 'sample'")
    return methods, data_modes


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}

    methods, data_modes = normalise_types(base_config)
    print(f"Evaluation methods: {methods}")
    print(f"Evaluation data modes: {data_modes}")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="pino_eval_",
        dir=PROJECT_ROOT,
        delete=False,
        encoding="utf-8",
    ) as runtime_file:
        runtime_config_path = Path(runtime_file.name)

    try:
        for method in methods:
            script = PROJECT_ROOT / (
                "Evaluation_P2P.py" if method == "p2p" else "Evaluation_Rollout.py"
            )
            for data_mode in data_modes:
                runtime_config = dict(base_config)
                runtime_config["eval_data_type"] = data_mode
                runtime_config["eval_method"] = method

                # Keep the result tree explicit even when this file is run
                # independently of PINO_Train.py.
                configured_run_dir = runtime_config.get("run_dir")
                output_root = Path(configured_run_dir) / "eval" if configured_run_dir else Path("eval_results")
                if not output_root.is_absolute():
                    output_root = PROJECT_ROOT / output_root
                stage = runtime_config.get("eval_stage", "standalone")
                runtime_config["eval_output_dir"] = str(
                    output_root / method / str(stage) / data_mode
                )

                with runtime_config_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(runtime_config, handle, sort_keys=False)

                print(f"\n=== {method.upper()} / {data_mode} ===")
                result = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=PROJECT_ROOT,
                    env={**os.environ, "PINO_EVAL_CONFIG": str(runtime_config_path)},
                )
                if result.returncode != 0:
                    raise SystemExit(result.returncode)
    finally:
        runtime_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

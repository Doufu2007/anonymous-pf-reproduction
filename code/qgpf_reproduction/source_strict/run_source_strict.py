"""Fail-closed entry point for a source-faithful QGPF reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_strict_common import MissingSourceEvidence, load_json, write_failure


REQUIRED_FIELDS = (
    "ratio_0_7_meaning",
    "noise_matrix_convention",
    "observation_noise_variance",
    "replacement_retention_rule",
    "recompute_weights_after_optimization",
    "trajectory_reuse_across_99_runs",
    "random_number_consumption_order",
    "author_code_or_raw_data",
    "execution_mode",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("source_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("runs-strict"))
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        if config.get("particle_count") != 400 or config.get("bit_length") != 10 or config.get("generation_count") != 4:
            raise MissingSourceEvidence("strict QGPF constants must remain N=400, m=10, t=4")
        require = [k for k in REQUIRED_FIELDS if config.get(k) in (None, "UNRESOLVED", "REQUEST_AUTHOR")]
        if require:
            raise MissingSourceEvidence("unresolved source fields: " + ", ".join(require))
        raise MissingSourceEvidence(
            "strict contract is closed, but the source trace and independent code review must be attached before execution"
        )
    except (MissingSourceEvidence, ValueError, json.JSONDecodeError) as exc:
        failure = args.out_dir / "failure_artifacts" / "strict_blocked.json"
        write_failure(failure, method="QGPF", message=str(exc), config_path=args.config)
        print(json.dumps({"status": "BLOCKED", "method": "QGPF", "failure": str(failure), "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

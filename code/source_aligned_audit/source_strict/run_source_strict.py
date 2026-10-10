"""Fail-closed entry point for a full 180-step EC-QPF reproduction."""

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
    "iteration_max",
    "table_II_parameters",
    "imu_velocity_heading_step_generator",
    "imu_noise_parameters",
    "initial_particle_distribution",
    "beta_to_ellipse_scale_relation",
    "fitness_scalar_V_definition",
    "empty_subset_policy",
    "tie_break_rule",
    "duplication_allocation",
    "random_seeds_or_source_trajectories",
    "cmcr_pf_implementation",
    "sr_pf_implementation",
    "execution_mode",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("source_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("runs-strict"))
    args = parser.parse_args()
    try:
        config = load_json(args.config)
        if config.get("particle_count") != 500 or config.get("swarm_size") != 500 or config.get("tracking_steps") != 180:
            raise MissingSourceEvidence("strict EC-QPF constants must remain N=np=500 and 180 tracking steps")
        unresolved = [k for k in REQUIRED_FIELDS if config.get(k) in (None, "UNRESOLVED", "REQUEST_AUTHOR")]
        if unresolved:
            raise MissingSourceEvidence("unresolved source fields: " + ", ".join(unresolved))
        if config.get("fitness_contract") != "source_literal_eq22":
            raise MissingSourceEvidence("strict runner requires the source-literal Eq. (22) contract")
        raise MissingSourceEvidence(
            "strict contract is closed, but the 180-step trace and independent code review must be attached before execution"
        )
    except (MissingSourceEvidence, ValueError, json.JSONDecodeError) as exc:
        failure = args.out_dir / "failure_artifacts" / "strict_blocked.json"
        write_failure(failure, method="EC-QPF", message=str(exc), config_path=args.config)
        print(json.dumps({"status": "BLOCKED", "method": "EC-QPF", "failure": str(failure), "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

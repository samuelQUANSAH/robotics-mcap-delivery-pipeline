import argparse
import json
from pathlib import Path

from .mcap_engine import load_client_configs, run_client_delivery_pipeline
from .models import FaultInjectionConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configurable Robotics Multimodal MCAP Transformation & Client Delivery CLI"
    )
    parser.add_argument(
        "--client",
        default="frontier_manipulation_lab",
        help="Target frontier lab client ID from configs/clients.yaml",
    )
    parser.add_argument("--frames", type=int, default=120, help="Number of multimodal frames to process")
    parser.add_argument("--hand-threshold", type=float, default=None, help="Override min_hand_in_frame_confidence")
    parser.add_argument("--max-drift-ms", type=float, default=None, help="Override max_sensor_sync_drift_ms")
    parser.add_argument("--inject-occlusion", action="store_true", help="Simulate hand occlusion burst")
    parser.add_argument("--inject-drift", action="store_true", help="Simulate IMU/RGB timestamp drift spike")
    parser.add_argument("--inject-nan", action="store_true", help="Simulate NaN joint encoder spike")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for .mcap container")
    args = parser.parse_args()

    clients = load_client_configs()
    if args.client not in clients:
        raise SystemExit(f"Unknown client '{args.client}'. Available: {', '.join(clients.keys())}")

    result = run_client_delivery_pipeline(
        client_id=args.client,
        num_frames=args.frames,
        faults=FaultInjectionConfig(
            hand_occlusion_burst=args.inject_occlusion,
            sensor_clock_drift=args.inject_drift,
            nan_joint_spike=args.inject_nan,
        ),
        override_hand_confidence=args.hand_threshold,
        override_sync_drift_ms=args.max_drift_ms,
        output_dir=args.output_dir,
    )

    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()

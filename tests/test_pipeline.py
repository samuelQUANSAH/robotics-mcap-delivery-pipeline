from pathlib import Path

from src.pipeline.mcap_engine import MCAP_MAGIC, load_client_configs, run_client_delivery_pipeline
from src.pipeline.models import FaultInjectionConfig


def test_load_client_configs() -> None:
    clients = load_client_configs()
    assert "frontier_manipulation_lab" in clients
    assert "embodied_navigation_lab" in clients
    assert "vla_foundation_lab" in clients


def test_clean_pipeline_run_and_mcap_magic(tmp_path: Path) -> None:
    result = run_client_delivery_pipeline(
        client_id="frontier_manipulation_lab",
        num_frames=60,
        faults=FaultInjectionConfig(),
        output_dir=tmp_path,
    )
    assert result.overall_status == "DELIVERED"
    assert result.delivered_frames == 60
    assert result.dropped_frames == 0
    assert result.mcap_magic_verified is True

    mcap_file = Path(result.mcap_file_path)
    assert mcap_file.is_file()
    raw = mcap_file.read_bytes()
    assert raw.startswith(MCAP_MAGIC)
    assert raw.endswith(MCAP_MAGIC)
    assert "ego_hand_cam.rgb_frame" in result.sample_transformed_frame["channels"]


def test_hand_occlusion_filtering_and_nan_repair(tmp_path: Path) -> None:
    result = run_client_delivery_pipeline(
        client_id="frontier_manipulation_lab",
        num_frames=80,
        faults=FaultInjectionConfig(
            hand_occlusion_burst=True,
            nan_joint_spike=True,
        ),
        override_hand_confidence=0.85,
        output_dir=tmp_path,
    )
    assert result.overall_status == "DELIVERED_WITH_FILTERS"
    # Frames 18..27 (10 frames) have hand confidence < 0.85 and should be dropped
    assert result.dropped_frames == 10
    assert result.delivered_frames == 70
    # Frames 62..64 (3 frames) have NaN joint spikes and should be repaired via zero-order hold
    assert result.repaired_frames == 3


def test_per_client_schema_difference(tmp_path: Path) -> None:
    res_a = run_client_delivery_pipeline(
        client_id="frontier_manipulation_lab", num_frames=30, output_dir=tmp_path
    )
    res_b = run_client_delivery_pipeline(
        client_id="embodied_navigation_lab", num_frames=30, output_dir=tmp_path
    )
    assert "ego_hand_cam.rgb_frame" in res_a.sample_transformed_frame["channels"]
    assert "/sensors/front_stereo/image_rect_color" in res_b.sample_transformed_frame["channels"]

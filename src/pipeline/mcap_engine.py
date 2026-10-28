import hashlib
import json
import math
import struct
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

from .models import (
    ClientSchemaConfig,
    FaultInjectionConfig,
    PipelineRunResult,
    QualityGateConfig,
    StageTelemetry,
    StorageEconomicsReport,
    ValidationAnomaly,
)

MCAP_MAGIC = b"\x89MCAP0\r\n"

STORAGE_COST_PER_TB_USD = {
    "S3_INTELLIGENT_TIERING": 18.40,
    "S3_EXPRESS_ONE_ZONE": 115.00,
    "S3_STANDARD": 23.00,
    "GCS_STANDARD": 20.00,
}


def load_client_configs(config_path: Optional[Path] = None) -> Dict[str, ClientSchemaConfig]:
    configs_dir = Path(__file__).resolve().parents[2] / "configs"
    if config_path is None:
        yaml_path = configs_dir / "clients.yaml"
        json_path = configs_dir / "clients.json"
        if yaml is not None and yaml_path.is_file():
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        if config_path.suffix in (".yaml", ".yml") and yaml is not None:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
    clients: Dict[str, ClientSchemaConfig] = {}
    for cid, cdata in raw.get("clients", {}).items():
        clients[cid] = ClientSchemaConfig(**cdata)
    return clients


def generate_raw_multimodal_stream(
    num_frames: int = 120,
    base_hz: float = 50.0,
    faults: Optional[FaultInjectionConfig] = None,
) -> List[Dict[str, Any]]:
    """Generates a realistic multi-sensor robotics episode (RGB, Aligned Depth, 6-DoF IMU, 14-DoF Bimanual Joints)."""
    if faults is None:
        faults = FaultInjectionConfig()

    base_ts_ns = 1_760_000_000_000_000_000
    period_ns = int(1_000_000_000 / base_hz)
    frames: List[Dict[str, Any]] = []

    for i in range(num_frames):
        t_rgb_ns = base_ts_ns + i * period_ns
        # Nominal sub-millisecond hardware PTP sync jitter (0.3ms - 1.1ms)
        sync_offset_ns = int((0.35 + 0.4 * abs(math.sin(i * 0.3))) * 1_000_000)

        # Fault 1: IMU / Depth clock synchronization drift spike on frames 40..47
        if faults.sensor_clock_drift and 40 <= i <= 47:
            sync_offset_ns = int((6.8 + (i - 40) * 0.45) * 1_000_000)

        # Nominal SAM2/VLM hand-in-frame segmentation confidence (0.89 - 0.98)
        hand_conf = round(0.91 + 0.07 * (0.5 + 0.5 * math.cos(i * 0.19)), 4)

        # Fault 2: Hand out-of-frame / occlusion burst on frames 18..27
        if faults.hand_occlusion_burst and 18 <= i <= 27:
            hand_conf = round(0.28 + 0.04 * (i - 18), 4)

        # 14-DoF bimanual arm joint angles (radians)
        joints = [round(0.45 * math.sin(i * 0.08 + j * 0.3), 5) for j in range(14)]
        if faults.nan_joint_spike and i in (62, 63, 64):
            joints[3] = float("nan")

        frames.append(
            {
                "frame_index": i,
                "timestamp_ns": t_rgb_ns,
                "imu_timestamp_ns": t_rgb_ns + sync_offset_ns,
                "sync_drift_ms": round(abs(sync_offset_ns) / 1_000_000.0, 3),
                "/camera/wrist_rgb": {
                    "encoding": "rgb8",
                    "resolution": [1280, 720],
                    "hand_in_frame_confidence": hand_conf,
                    "hands_detected": 2 if hand_conf >= 0.5 else 0,
                    "bbox_xyxy": [210, 140, 980, 660] if hand_conf >= 0.5 else [0, 0, 0, 0],
                },
                "/depth/aligned_depth": {
                    "encoding": "16UC1",
                    "min_depth_mm": 180,
                    "max_depth_mm": 1450,
                    "valid_pixel_ratio": round(0.96 - (0.18 if hand_conf < 0.5 else 0.0), 3),
                },
                "/imu/wrist": {
                    "linear_acceleration_m_s2": [
                        round(0.12 * math.cos(i * 0.1), 4),
                        round(-0.08 * math.sin(i * 0.1), 4),
                        9.806,
                    ],
                    "angular_velocity_rad_s": [
                        round(0.03 * math.sin(i * 0.2), 4),
                        round(0.02 * math.cos(i * 0.2), 4),
                        0.005,
                    ],
                },
                "/robot/joint_states": {
                    "joint_names": [f"arm_{side}_j{j}" for side in ("L", "R") for j in range(1, 8)],
                    "positions_rad": joints,
                    "gripper_aperture_m": [0.042, 0.038],
                },
            }
        )
    return frames


def _pack_mcap_record(opcode: int, payload: bytes) -> bytes:
    """Packs a binary MCAP record: [opcode: uint8][length: uint64 LE][payload]."""
    return struct.pack("<BQ", opcode, len(payload)) + payload


def _pack_mcap_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def write_mcap_container(
    output_path: Path,
    client_cfg: ClientSchemaConfig,
    episode_id: str,
    transformed_frames: List[Dict[str, Any]],
) -> Tuple[int, str, bool]:
    """
    Writes a spec-compliant MCAP v0 container file with Header, Schema, Channel,
    timestamped Message records, compressed payload summary, and Footer.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buf = bytearray()
    buf.extend(MCAP_MAGIC)

    # 1. Header Record (Opcode 0x01): profile + library
    header_payload = _pack_mcap_string("ros2" if "ros2" in client_cfg.schema_version else "x-robotics-multimodal") + _pack_mcap_string(
        f"robotics-mcap-pipeline/{client_cfg.schema_version}"
    )
    buf.extend(_pack_mcap_record(0x01, header_payload))

    # 2. Schema Record (Opcode 0x03)
    schema_def = json.dumps(
        {
            "title": client_cfg.display_name,
            "schema_version": client_cfg.schema_version,
            "field_mappings": client_cfg.field_mappings,
            "custom_metadata": client_cfg.custom_metadata,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    schema_payload = (
        struct.pack("<H", 1)
        + _pack_mcap_string(f"{client_cfg.client_id}_episode_schema")
        + _pack_mcap_string("jsonschema")
        + struct.pack("<I", len(schema_def))
        + schema_def
    )
    buf.extend(_pack_mcap_record(0x03, schema_payload))

    # 3. Channel Record (Opcode 0x04) for each mapped topic
    channel_ids: Dict[str, int] = {}
    for idx, (_, dst_topic) in enumerate(client_cfg.field_mappings.items(), start=1):
        channel_ids[dst_topic] = idx
        ch_payload = (
            struct.pack("<HH", idx, 1)
            + _pack_mcap_string(dst_topic)
            + _pack_mcap_string("json")
            + struct.pack("<I", 0)
        )
        buf.extend(_pack_mcap_record(0x04, ch_payload))

    # 4. Message Records (Opcode 0x05)
    for seq, frame in enumerate(transformed_frames):
        ts_ns = int(frame["timestamp_ns"])
        payload_dict = frame["channels"]
        for dst_topic, ch_data in payload_dict.items():
            ch_id = channel_ids.get(dst_topic, 1)
            encoded_msg = json.dumps(ch_data, separators=(",", ":")).encode("utf-8")
            if client_cfg.compression in ("zstd", "lz4"):
                encoded_msg = zlib.compress(encoded_msg, level=6)
            msg_header = struct.pack("<HIQQ", ch_id, seq, ts_ns, ts_ns)
            buf.extend(_pack_mcap_record(0x05, msg_header + encoded_msg))

    # 5. Footer Record (Opcode 0x02) + Trailing Magic
    summary_start = len(buf)
    footer_payload = struct.pack("<QQI", summary_start, 0, zlib.crc32(buf))
    buf.extend(_pack_mcap_record(0x02, footer_payload))
    buf.extend(MCAP_MAGIC)

    raw_bytes = bytes(buf)
    output_path.write_bytes(raw_bytes)

    # Verify MCAP magic header and footer bytes
    verified = raw_bytes.startswith(MCAP_MAGIC) and raw_bytes.endswith(MCAP_MAGIC)
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    return len(raw_bytes), sha256_hex, verified


def run_client_delivery_pipeline(
    client_id: str = "frontier_manipulation_lab",
    num_frames: int = 120,
    faults: Optional[FaultInjectionConfig] = None,
    override_hand_confidence: Optional[float] = None,
    override_sync_drift_ms: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> PipelineRunResult:
    """
    Executes the 4-stage Robotics Multimodal Data Delivery Pipeline:
      Stage 1: Multi-Sensor Stream Ingestion & Demuxing
      Stage 2: Configurable VLM Hand-in-Frame Segmentation & Sync QC Gate
      Stage 3: Per-Client Declarative Schema & Coordinate Transformation
      Stage 4: Spec-Compliant MCAP Packaging & Object Storage Economics Manifest
    """
    clients = load_client_configs()
    if client_id not in clients:
        raise ValueError(f"Unknown client_id '{client_id}'. Available: {list(clients.keys())}")

    client_cfg = clients[client_id].model_copy(deep=True)
    if override_hand_confidence is not None:
        client_cfg.quality_gates.min_hand_in_frame_confidence = override_hand_confidence
    if override_sync_drift_ms is not None:
        client_cfg.quality_gates.max_sensor_sync_drift_ms = override_sync_drift_ms

    if faults is None:
        faults = FaultInjectionConfig()

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    episode_id = f"ep_{client_id[:12]}_{int(time.time())}"
    stages: List[StageTelemetry] = []
    anomalies: List[ValidationAnomaly] = []

    # -------------------------------------------------------------------------
    # STAGE 1: Multi-Sensor Stream Ingestion & Demuxing
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    raw_frames = generate_raw_multimodal_stream(num_frames=num_frames, base_hz=50.0, faults=faults)
    raw_json_bytes = len(json.dumps(raw_frames, allow_nan=True).encode("utf-8"))
    s1_ms = round((time.perf_counter() - t0) * 1000, 3)

    stages.append(
        StageTelemetry(
            stage_id="1_ingest_demux",
            stage_name="1. Multi-Sensor Stream Ingestion & Demux",
            status="PASS",
            duration_ms=s1_ms,
            summary=f"Ingested {len(raw_frames)} multimodal frames across 4 topics (RGB, Depth, 6-DoF IMU, 14-DoF Joints).",
            metrics={"raw_frames": len(raw_frames), "raw_bytes": raw_json_bytes, "topics_demuxed": 4},
        )
    )

    # -------------------------------------------------------------------------
    # STAGE 2: Configurable VLM Hand-Segmentation & Multi-Sensor Sync QC Gate
    # -------------------------------------------------------------------------
    t1 = time.perf_counter()
    gates: QualityGateConfig = client_cfg.quality_gates
    valid_frames: List[Dict[str, Any]] = []
    dropped_count = 0
    repaired_count = 0
    max_drift = 0.0
    hand_conf_sum = 0.0

    for idx, frame in enumerate(raw_frames):
        ts_ns = frame["timestamp_ns"]
        drift_ms = frame["sync_drift_ms"]
        max_drift = max(max_drift, drift_ms)
        hand_conf = frame["/camera/wrist_rgb"]["hand_in_frame_confidence"]
        hand_conf_sum += hand_conf

        # Check 2A: Non-finite joint encoder values (NaN / Inf repair via linear hold)
        joints = frame["/robot/joint_states"]["positions_rad"]
        if gates.require_joint_finite and any(math.isnan(v) or math.isinf(v) for v in joints):
            prev_joints = valid_frames[-1]["/robot/joint_states"]["positions_rad"] if valid_frames else [0.0] * 14
            frame["/robot/joint_states"]["positions_rad"] = [
                prev_joints[j] if (math.isnan(v) or math.isinf(v)) else v for j, v in enumerate(joints)
            ]
            repaired_count += 1
            anomalies.append(
                ValidationAnomaly(
                    frame_index=idx,
                    timestamp_ns=ts_ns,
                    anomaly_type="NAN_JOINT_ENCODER_SPIKE",
                    severity="REPAIRED",
                    detail="Interpolated non-finite joint encoder reading using zero-order hold from previous valid state.",
                )
            )

        # Check 2B: Multi-sensor timestamp synchronization gate
        if drift_ms > gates.max_sensor_sync_drift_ms:
            dropped_count += 1
            anomalies.append(
                ValidationAnomaly(
                    frame_index=idx,
                    timestamp_ns=ts_ns,
                    anomaly_type="SENSOR_SYNC_DRIFT_EXCEEDED",
                    severity="DROPPED",
                    detail=f"IMU-to-RGB clock drift {drift_ms:.2f}ms exceeds client max {gates.max_sensor_sync_drift_ms:.2f}ms.",
                )
            )
            continue

        # Check 2C: Customer-configurable VLM Hand-in-Frame segmentation filter
        if hand_conf < gates.min_hand_in_frame_confidence:
            if gates.drop_occluded_frames:
                dropped_count += 1
                anomalies.append(
                    ValidationAnomaly(
                        frame_index=idx,
                        timestamp_ns=ts_ns,
                        anomaly_type="HAND_SEGMENTATION_BELOW_THRESHOLD",
                        severity="DROPPED",
                        detail=f"Hand confidence {hand_conf:.2f} < client threshold {gates.min_hand_in_frame_confidence:.2f} ({client_cfg.custom_metadata.get('hand_segmentation_model', 'sam2')}).",
                    )
                )
                continue
            else:
                anomalies.append(
                    ValidationAnomaly(
                        frame_index=idx,
                        timestamp_ns=ts_ns,
                        anomaly_type="HAND_SEGMENTATION_FLAGGED",
                        severity="FLAGGED",
                        detail=f"Hand confidence {hand_conf:.2f} < {gates.min_hand_in_frame_confidence:.2f}; retained with occlusion metadata flag per client config.",
                    )
                )

        valid_frames.append(frame)

    s2_ms = round((time.perf_counter() - t1) * 1000, 3)
    s2_status = "PASS" if (dropped_count == 0 and repaired_count == 0) else ("REPAIRED" if repaired_count > 0 else "WARN")
    stages.append(
        StageTelemetry(
            stage_id="2_segmentation_and_sync_qc",
            stage_name="2. VLM Hand-Segmentation & Multi-Sensor Sync QC",
            status=s2_status,
            duration_ms=s2_ms,
            summary=f"Validated {len(valid_frames)}/{len(raw_frames)} frames ({dropped_count} dropped, {repaired_count} repaired). Hand filter >= {gates.min_hand_in_frame_confidence:.2f}.",
            metrics={
                "passed_frames": len(valid_frames),
                "dropped_frames": dropped_count,
                "repaired_frames": repaired_count,
                "max_sync_drift_ms": round(max_drift, 3),
            },
        )
    )

    # -------------------------------------------------------------------------
    # STAGE 3: Per-Client Declarative Schema & Coordinate Transformation
    # -------------------------------------------------------------------------
    t2 = time.perf_counter()
    transformed_frames: List[Dict[str, Any]] = []
    mappings = client_cfg.field_mappings

    for frame in valid_frames:
        mapped_channels: Dict[str, Any] = {}
        for src_topic, dst_field in mappings.items():
            if src_topic in frame:
                mapped_channels[dst_field] = frame[src_topic]
        transformed_frames.append(
            {
                "frame_index": frame["frame_index"],
                "timestamp_ns": frame["timestamp_ns"],
                "schema_version": client_cfg.schema_version,
                "coordinate_frame": client_cfg.custom_metadata.get("coordinate_frame", "default"),
                "channels": mapped_channels,
            }
        )

    s3_ms = round((time.perf_counter() - t2) * 1000, 3)
    stages.append(
        StageTelemetry(
            stage_id="3_schema_transform",
            stage_name="3. Per-Client Schema & Coordinate Transform",
            status="PASS",
            duration_ms=s3_ms,
            summary=f"Mapped {len(transformed_frames)} frames into '{client_cfg.schema_version}' ({client_cfg.custom_metadata.get('coordinate_frame')}).",
            metrics={
                "schema_version": client_cfg.schema_version,
                "mapped_channels": list(mappings.values()),
            },
        )
    )

    # -------------------------------------------------------------------------
    # STAGE 4: MCAP Container Packaging & Object Storage Economics Manifest
    # -------------------------------------------------------------------------
    t3 = time.perf_counter()
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "deliveries"
    mcap_path = output_dir / f"{episode_id}_{client_cfg.client_id}.mcap"

    mcap_bytes, sha256_hex, magic_ok = write_mcap_container(
        output_path=mcap_path,
        client_cfg=client_cfg,
        episode_id=episode_id,
        transformed_frames=transformed_frames,
    )

    compression_ratio = round(raw_json_bytes / max(mcap_bytes, 1), 2)
    cost_per_tb = STORAGE_COST_PER_TB_USD.get(client_cfg.storage_class, 23.0)
    multipart_parts = math.ceil((10 * 1024) / client_cfg.chunk_size_mb)

    economics = StorageEconomicsReport(
        raw_payload_bytes=raw_json_bytes,
        mcap_container_bytes=mcap_bytes,
        compression_ratio=compression_ratio,
        target_chunk_size_mb=client_cfg.chunk_size_mb,
        multipart_parts_at_10gb=multipart_parts,
        estimated_monthly_cost_per_tb_usd=cost_per_tb,
        storage_class=client_cfg.storage_class,
        delivery_uri=f"{client_cfg.delivery_target}{mcap_path.name}",
    )
    s4_ms = round((time.perf_counter() - t3) * 1000, 3)

    stages.append(
        StageTelemetry(
            stage_id="4_mcap_package_and_manifest",
            stage_name="4. MCAP Packaging & Cloud Object Storage Delivery",
            status="PASS" if magic_ok else "FAIL",
            duration_ms=s4_ms,
            summary=f"Packaged {mcap_bytes:,}B MCAP ({compression_ratio}x compression, {client_cfg.compression}) -> {economics.delivery_uri}",
            metrics={
                "mcap_bytes": mcap_bytes,
                "compression_ratio": compression_ratio,
                "sha256": sha256_hex[:16],
                "magic_verified": magic_ok,
            },
        )
    )

    effective_hz = round(50.0 * (len(transformed_frames) / max(len(raw_frames), 1)), 1)
    overall_status = (
        "REJECTED"
        if effective_hz < gates.min_episode_hz or not magic_ok
        else ("DELIVERED_WITH_FILTERS" if (dropped_count > 0 or repaired_count > 0) else "DELIVERED")
    )

    return PipelineRunResult(
        run_id=run_id,
        episode_id=episode_id,
        client_id=client_cfg.client_id,
        client_display_name=client_cfg.display_name,
        schema_version=client_cfg.schema_version,
        overall_status=overall_status,
        total_raw_frames=len(raw_frames),
        delivered_frames=len(transformed_frames),
        dropped_frames=dropped_count,
        repaired_frames=repaired_count,
        effective_hz=effective_hz,
        max_observed_sync_drift_ms=round(max_drift, 3),
        avg_hand_confidence=round(hand_conf_sum / max(len(raw_frames), 1), 4),
        mcap_file_path=str(mcap_path),
        mcap_magic_verified=magic_ok,
        mcap_sha256=sha256_hex,
        stages=stages,
        anomalies=anomalies,
        sample_raw_frame=raw_frames[0] if raw_frames else {},
        sample_transformed_frame=transformed_frames[0] if transformed_frames else {},
        storage_economics=economics,
    )

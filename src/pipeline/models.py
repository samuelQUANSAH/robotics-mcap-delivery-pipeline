from dataclasses import asdict, dataclass, field
import copy
from typing import Any, Dict, List


class _SerializableModel:
    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)

    def model_copy(self, deep: bool = True) -> Any:
        return copy.deepcopy(self) if deep else copy.copy(self)


@dataclass
class QualityGateConfig(_SerializableModel):
    min_hand_in_frame_confidence: float = 0.85
    max_sensor_sync_drift_ms: float = 5.0
    min_episode_hz: float = 30.0
    drop_occluded_frames: bool = True
    require_joint_finite: bool = True


@dataclass
class ClientSchemaConfig(_SerializableModel):
    client_id: str
    display_name: str
    schema_version: str
    delivery_target: str
    field_mappings: Dict[str, str]
    quality_gates: QualityGateConfig
    container_format: str = "mcap"
    compression: str = "zstd"
    chunk_size_mb: int = 64
    storage_class: str = "S3_INTELLIGENT_TIERING"
    custom_metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.quality_gates, dict):
            self.quality_gates = QualityGateConfig(**self.quality_gates)


@dataclass
class FaultInjectionConfig(_SerializableModel):
    hand_occlusion_burst: bool = False
    sensor_clock_drift: bool = False
    nan_joint_spike: bool = False


@dataclass
class StageTelemetry(_SerializableModel):
    stage_id: str
    stage_name: str
    status: str  # "PASS", "WARN", "REPAIRED", "FAIL"
    duration_ms: float
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationAnomaly(_SerializableModel):
    frame_index: int
    timestamp_ns: int
    anomaly_type: str
    severity: str  # "DROPPED", "REPAIRED", "FLAGGED"
    detail: str


@dataclass
class StorageEconomicsReport(_SerializableModel):
    raw_payload_bytes: int
    mcap_container_bytes: int
    compression_ratio: float
    target_chunk_size_mb: int
    multipart_parts_at_10gb: int
    estimated_monthly_cost_per_tb_usd: float
    storage_class: str
    delivery_uri: str


@dataclass
class PipelineRunResult(_SerializableModel):
    run_id: str
    episode_id: str
    client_id: str
    client_display_name: str
    schema_version: str
    overall_status: str  # "DELIVERED", "DELIVERED_WITH_FILTERS", "REJECTED"
    total_raw_frames: int
    delivered_frames: int
    dropped_frames: int
    repaired_frames: int
    effective_hz: float
    max_observed_sync_drift_ms: float
    avg_hand_confidence: float
    mcap_file_path: str
    mcap_magic_verified: bool
    mcap_sha256: str
    stages: List[StageTelemetry]
    anomalies: List[ValidationAnomaly]
    sample_raw_frame: Dict[str, Any]
    sample_transformed_frame: Dict[str, Any]
    storage_economics: StorageEconomicsReport

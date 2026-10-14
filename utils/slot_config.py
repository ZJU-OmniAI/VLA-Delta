"""Persistent, opt-in structured supervision configuration."""

from dataclasses import asdict, dataclass, field
import math

TASK_WEIGHTS = dict(zip((f"Q{i}" for i in range(1, 10)), (.20, .20, .04, .04, .04, .20, .12, .12, .04)))


def _structured_slots_v1(hidden_size, num_slot_queries):
    from model.structured_slot_head import StructuredSlotHead
    return StructuredSlotHead(hidden_size, num_slot_queries)


SLOT_AUX_REGISTRY = {"none": None, "structured_slots_v1": _structured_slots_v1}


@dataclass(frozen=True)
class SlotConfig:
    slot_aux_type: str = "none"
    slot_loss_weight: float = 0.0
    slot_task_weights: dict = field(default_factory=lambda: dict(TASK_WEIGHTS))
    slot_schema_version: str = "future_difference_training_v5"
    slot_smooth_l1_beta: float = 1.0
    slot_class_max_weight_ratio: float = 10.0
    slot_q5_std_floor: float = 1e-3
    slot_label_smoothing: float = .02
    slot_progress_monotonic_weight: float = .1
    slot_bbox_giou_weight: float = .1
    slot_q5_consistency_weight: float = .05
    slot_presence_weight: float = .5
    slot_obstacle_bbox_weight: float = .25
    slot_risk_weight: float = .25
    slot_init_seed: int = 42
    stage2_aux_sampling: str = "slot_valid"

    @property
    def enabled(self):
        return self.slot_aux_type != "none"

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.slot_aux_type not in SLOT_AUX_REGISTRY:
            raise NotImplementedError(self.slot_aux_type)
        if self.slot_schema_version != "future_difference_training_v5":
            raise ValueError("unsupported slot_schema_version")
        if type(self.slot_init_seed) is not int:
            raise ValueError("slot_init_seed must be an integer")
        if set(self.slot_task_weights) != set(TASK_WEIGHTS):
            raise ValueError("slot_task_weights must specify exactly Q1-Q9")
        for name, value in self.to_dict().items():
            if name != "slot_init_seed" and isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid {name}")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in self.slot_task_weights.values()):
            raise ValueError("invalid slot_task_weights")
        if not self.enabled and self.slot_loss_weight != 0:
            raise ValueError("disabled Slot requires zero slot_loss_weight")
        if self.enabled and not any(self.slot_task_weights.values()):
            raise ValueError("enabled Slot needs a positive task weight")
        if self.slot_smooth_l1_beta <= 0 or self.slot_q5_std_floor <= 0 or self.slot_class_max_weight_ratio < 1:
            raise ValueError("invalid Slot numerical bounds")
        if not 0 <= self.slot_label_smoothing <= 1:
            raise ValueError("invalid label smoothing")
        if self.stage2_aux_sampling not in {"slot_valid", "any_aux_valid"}:
            raise ValueError("invalid stage2_aux_sampling")
        return self


def resolve_query_layout(total, flow_count, *, slot_enabled, query_enabled):
    if flow_count is None:
        if slot_enabled:
            raise ValueError("first Slot initialization requires explicit num_flow_queries (zero allowed)")
        return None
    if type(flow_count) is not int or not 0 <= flow_count <= total:
        raise ValueError("invalid num_flow_queries role boundary")
    if slot_enabled and (not query_enabled or total - flow_count < 7):
        raise ValueError("Slot requires enabled Difference Query and at least seven Slot queries")
    from model.structured_slot_head import slot_groups
    n = total - flow_count
    return {"num_difference_queries": total, "num_flow_queries": flow_count, "num_slot_queries": n,
            "groups": {k: list(v) for k, v in slot_groups(n).items()} if n >= 7 else {}}

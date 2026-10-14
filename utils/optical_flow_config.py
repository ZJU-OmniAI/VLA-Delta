"""Central configuration and objective selection for optional auxiliary training."""

from dataclasses import asdict, dataclass
import math


STAGES = {"stage1_ar": "vlm", "stage2_aux": "aux", "stage3_joint": "vlm_and_action",
          "downstream_flow": "action"}
from utils.slot_config import SLOT_AUX_REGISTRY


@dataclass(frozen=True)
class OpticalFlowConfig:
    optical_flow_aux_type: str = "none"
    num_flow_queries: int | None = None
    optical_flow_loss_weight: float = 0.0
    flow_delta_frames: int = 10
    flow_target_resolution: int = 56
    flow_grid_size: int = 14
    flow_head_hidden_dim: int = 256
    flow_head_num_heads: int = 8
    flow_head_num_layers: int = 2
    flow_cell_min_valid_fraction: float = 0.5
    flow_motion_loss_weight: float = 1.0
    flow_motion_threshold: float = 0.01
    flow_loss_epsilon: float = 1e-3
    flow_init_seed: int = 42

    @property
    def enabled(self):
        return self.optical_flow_aux_type != "none"

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.optical_flow_aux_type not in {"none", "dense_regression_v1"}:
            raise NotImplementedError(self.optical_flow_aux_type)
        for name in ("optical_flow_loss_weight", "flow_motion_loss_weight", "flow_motion_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.enabled:
            if self.optical_flow_loss_weight != 0:
                raise ValueError("disabled OF requires optical_flow_loss_weight=0")
            return self
        if isinstance(self.num_flow_queries, bool) or not isinstance(self.num_flow_queries, int) or self.num_flow_queries < 1:
            raise ValueError("OF requires explicit positive num_flow_queries")
        if self.optical_flow_loss_weight <= 0:
            raise ValueError("OF requires positive optical_flow_loss_weight")
        for name in ("flow_delta_frames", "flow_head_hidden_dim", "flow_head_num_layers", "flow_grid_size", "flow_target_resolution", "flow_head_num_heads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")
        if self.optical_flow_aux_type == "dense_regression_v1":
            if (self.flow_grid_size, self.flow_target_resolution, self.flow_head_num_heads) != (14, 56, 8):
                raise ValueError("dense_regression_v1 requires grid=14, resolution=56, heads=8")
            if self.flow_head_hidden_dim % 8:
                raise ValueError("flow_head_hidden_dim must be divisible by 8")
        if not 0 < self.flow_cell_min_valid_fraction <= 1:
            raise ValueError("flow_cell_min_valid_fraction must be in (0,1]")
        if not math.isfinite(self.flow_loss_epsilon) or self.flow_loss_epsilon <= 0:
            raise ValueError("flow_loss_epsilon must be finite and positive")
        return self


def resolve_stage(training_stage, loss_type=None, *, flow=None, slot_aux_type="none", slot=None):
    flow = (flow or OpticalFlowConfig()).validate()
    if slot is not None:
        from utils.aux_objectives import validate_stage_objectives
        validate_stage_objectives(training_stage, slot, flow)
        slot_aux_type = slot.slot_aux_type
    if slot_aux_type not in SLOT_AUX_REGISTRY:
        raise NotImplementedError("unregistered Slot auxiliary type: " + slot_aux_type)
    if training_stage is None:
        if flow.enabled or slot_aux_type != "none":
            raise ValueError("auxiliary training requires an explicit training_stage")
        if loss_type not in {None, "vlm", "action", "vlm_and_action"}:
            raise ValueError("loss_type must be vlm, action or vlm_and_action")
        return loss_type or "vlm_and_action"
    if training_stage not in STAGES:
        raise ValueError(f"unknown training_stage {training_stage!r}")
    expected = STAGES[training_stage]
    if loss_type is not None and loss_type != expected:
        raise ValueError(f"explicit loss_type={loss_type} conflicts with {training_stage}")
    if training_stage == "stage1_ar" and (flow.enabled or slot_aux_type != "none"):
        raise ValueError("stage1_ar only permits AR")
    if training_stage == "stage2_aux" and not flow.enabled and slot_aux_type == "none":
        raise ValueError("stage2_aux requires enabled Slot or Flow with positive weight")
    if training_stage == "downstream_flow" and (not flow.enabled or slot_aux_type != "none"):
        raise ValueError("downstream_flow requires Optical Flow and forbids Slot")
    return expected


def stage_description(stage, flow, slot_aux_type="none"):
    if stage == "downstream_flow":
        return "downstream FM+OF / AR and Slot disabled"
    if slot_aux_type != "none":
        return {"stage2_aux": "Slot+OF" if flow.enabled else "Slot-only",
                "stage3_joint": "AR+Slot+OF+FM" if flow.enabled else "AR+Slot+FM"}.get(stage)
    return {"stage1_ar": "AR-only", "stage2_aux": "stage2 OF-only / Slot disabled",
            "stage3_joint": ("AR+FM+OF provisional joint" if flow.enabled else "AR+FM provisional joint")}.get(stage)


def stage_loss_metadata(stage, *, ar=False, flow=False, fm=False, slot=False, slot_enabled=False):
    if stage is None:
        return {}
    return {"training_stage": stage, "provisional": stage == "stage3_joint" and not slot_enabled,
            "slot_loss_computed": bool(slot), "ar_loss_computed": bool(ar),
            "flow_loss_computed": bool(flow), "fm_loss_computed": bool(fm)}

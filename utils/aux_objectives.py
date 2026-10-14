"""Effective auxiliary coefficients shared by startup and window activity checks."""


def slot_objective_components(config):
    """Return positively weighted loss components, without changing reductions."""
    components = {}
    for task, weight in config.slot_task_weights.items():
        terms = {"base": 1.}
        if task == "Q9":
            terms = {"presence": config.slot_presence_weight,
                     "bbox": config.slot_obstacle_bbox_weight, "risk": config.slot_risk_weight}
        # Q1-Q8 each retain a unit-coefficient base term; their optional
        # consistency, monotonicity and GIoU terms cannot disable that term.
        components[task] = tuple(name for name, coefficient in terms.items()
            if config.enabled and config.slot_loss_weight > 0 and weight > 0 and coefficient > 0)
    return components


def validate_stage_objectives(stage, slot, flow):
    slot.validate()
    flow.validate()
    if stage != "stage2_aux":
        return
    if any(slot_objective_components(slot).values()) or (flow.enabled and flow.optical_flow_loss_weight > 0):
        return
    raise ValueError(
        f"stage2_aux has no positive effective objective: Slot={slot.slot_aux_type}, "
        f"slot_loss_weight={slot.slot_loss_weight}, slot_task_weights={slot.slot_task_weights}, "
        f"Q9 weights (presence={slot.slot_presence_weight}, bbox={slot.slot_obstacle_bbox_weight}, "
        f"risk={slot.slot_risk_weight}); Flow={flow.optical_flow_aux_type}, "
        f"optical_flow_loss_weight={flow.optical_flow_loss_weight}")

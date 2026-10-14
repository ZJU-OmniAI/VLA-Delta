"""Mask-aware normalized-extent targets and differentiable sample loss sums."""

import torch
from torch.nn import functional as F


def prepare_flow_targets(batch, config, device):
    available = batch.get("flow_supervision_available")
    batch_size = batch["input_ids"].shape[0]
    targets, masks, indices = [], [], []
    for index in range(batch_size):
        if available is None or not bool(available[index]):
            continue
        nominal = batch.get("flow_nominal_delta_frames")
        expected_delta = int(nominal[index]) if nominal is not None else config.flow_delta_frames
        if expected_delta <= 0:
            raise ValueError("invalid per-source flow nominal delta")
        if int(batch["flow_actual_delta_frames"][index]) != expected_delta or int(batch["flow_label_source"][index]) != 1:
            continue
        target = batch["flow_target"][index].to(device=device, dtype=torch.float32)
        mask = batch["flow_valid_mask"][index].to(device=device, dtype=torch.float32)
        if target.shape != (2, 224, 224) or mask.shape != (1, 224, 224):
            raise ValueError("flow target/mask must be [2,224,224]/[1,224,224]")
        if not torch.isfinite(target).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("invalid flow target/mask")
        fraction = F.interpolate(mask[None], size=(56, 56), mode="area")[0]
        valid = fraction >= config.flow_cell_min_valid_fraction
        if not valid.any():
            continue
        down = F.interpolate((target * mask)[None], size=(56, 56), mode="area")[0]
        targets.append(down / fraction.clamp_min(1e-12))
        masks.append(valid[0])
        indices.append(index)
    if not indices:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, 2, 56, 56, device=device), torch.empty(0, 56, 56, dtype=torch.bool, device=device))
    return torch.tensor(indices, device=device), torch.stack(targets), torch.stack(masks)


def optical_flow_loss(prediction, batch, config):
    indices, target, valid = prepare_flow_targets(batch, config, prediction.device)
    zero = prediction.float().sum() * 0.0
    names = ("epe", "motion_epe", "zero_epe", "valid_fraction", "motion_fraction", "eligible_pixels")
    stats = {name: zero.detach().clone() for name in names}
    count = prediction.new_tensor(len(indices), dtype=torch.float32)
    loss_sum = zero
    if len(indices):
        residual = prediction[indices].float() - target
        epe = residual.square().sum(1).sqrt()
        robust = (residual.square().sum(1) + config.flow_loss_epsilon ** 2).sqrt() - config.flow_loss_epsilon
        magnitude = target.square().sum(1).sqrt()
        motion = valid & (magnitude > config.flow_motion_threshold)
        def means(value, mask):
            return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)
        loss_sum = (means(robust, valid) + config.flow_motion_loss_weight * means(robust, motion)).sum() + zero
        stats.update(epe=means(epe, valid).sum().detach(), motion_epe=means(epe, motion).sum().detach(),
                     zero_epe=means(magnitude, valid).sum().detach(),
                     valid_fraction=valid.float().mean((1, 2)).sum(), motion_fraction=motion.float().mean((1, 2)).sum(),
                     eligible_pixels=valid.sum().float())
    return {"optical_flow_loss_sum": loss_sum, "optical_flow_loss_count": count,
            "optical_flow_loss": loss_sum / count.clamp_min(1),
            **{"flow_" + key + "_sum": value for key, value in stats.items()},
            "flow_batch_samples": prediction.new_tensor(prediction.shape[0], dtype=torch.float32)}


def flow_supervision_indices(batch, config, device):
    """Use the loss branch's exact eligibility rule for window normalization."""
    return prepare_flow_targets(batch, config, device)[0]

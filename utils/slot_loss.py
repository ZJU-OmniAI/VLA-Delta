"""FP32 per-task sample means and additive statistics for distributed windows."""

import torch
from torch.nn import functional as F

from utils.slot_labels import empty_slot_labels, task_validity, VOCABULARIES


def giou(boxes, targets):
    intersection = (torch.minimum(boxes[..., 2:], targets[..., 2:]) - torch.maximum(boxes[..., :2], targets[..., :2])).clamp_min(0).prod(-1)
    area = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
    target_area = (targets[..., 2:] - targets[..., :2]).clamp_min(0).prod(-1)
    union = area + target_area - intersection
    enclosing = (torch.maximum(boxes[..., 2:], targets[..., 2:]) - torch.minimum(boxes[..., :2], targets[..., :2])).clamp_min(0).prod(-1)
    return intersection / union.clamp_min(1e-7) - (enclosing - union) / enclosing.clamp_min(1e-7)


def masked_sample_mean(values, mask):
    values, mask = values.flatten(1), mask.flatten(1)
    return torch.where(mask, values, 0).sum(-1) / mask.sum(-1).clamp_min(1)


def slot_loss(predictions, batch, config, stats):
    device = predictions["Q1"].device
    with torch.autocast(device_type=device.type, enabled=False):
        return _slot_loss_fp32({k: v.float() for k, v in predictions.items()}, batch, config, stats)


def _slot_loss_fp32(p, batch, config, stats):
    size, device = p["Q1"].shape[0], p["Q1"].device
    zero = sum(value.sum() * 0 for value in p.values())
    labels = {k: batch.get(k, v.unsqueeze(0).expand(size, *v.shape)).to(device)
              for k, v in empty_slot_labels().items()}
    valid = task_validity(labels, device=device, batch_size=size)
    losses, metrics = {}, {}

    def target(q):
        return labels[f"slot_{q}"].float()

    def mask(q):
        return labels[f"slot_{q}_mask"].bool()

    def smooth(x, y):
        return F.smooth_l1_loss(x, y, beta=config.slot_smooth_l1_beta, reduction="none")

    def metric(name, values, valid_mask):
        metrics[f"slot_metric_{name}_sum"] = torch.where(valid_mask, values, 0).sum().detach()
        metrics[f"slot_metric_{name}_count"] = valid_mask.sum().float().detach()

    def bbox_loss(box, truth):
        return smooth(box, truth).mean(-1) + config.slot_bbox_giou_weight * (1 - giou(box, truth))

    losses["Q1"] = masked_sample_mean(smooth(p["Q1"], target("Q1")), mask("Q1"))
    losses["Q1"] += config.slot_progress_monotonic_weight * F.relu(p["Q1"][:, 0] - p["Q1"][:, 1]) * mask("Q1").all(-1)
    for q in VOCABULARIES:
        width = len(VOCABULARIES[q])
        weights = torch.tensor(stats["classes"][q]["weights"], device=device, dtype=torch.float32)
        truth = labels[f"slot_{q}"].long()
        ce = F.cross_entropy(p[q].reshape(-1, width), truth.reshape(-1), weight=weights,
                             label_smoothing=config.slot_label_smoothing, reduction="none").reshape(size, -1)
        losses[q] = masked_sample_mean(ce, mask(q).reshape(size, -1))
        actual = truth[mask(q)]
        predicted = p[q].argmax(-1)[mask(q)]
        metrics[f"slot_metric_{q}_confusion"] = torch.bincount(actual * width + predicted, minlength=width * width).reshape(width, width).float()
    for q in ("Q3", "Q4"):
        losses[q] = masked_sample_mean(bbox_loss(p[q], target(q)), mask(q))
        metric(q + "_bbox_l1", (p[q] - target(q)).abs().mean(-1), mask(q))
        metric(q + "_giou", giou(p[q], target(q)), mask(q))
    if stats.get("version") == 2:
        route = batch["slot_dataset_index"].to(device=device, dtype=torch.long)
        if route.shape != (size,) or (route < 0).any() or (route >= len(stats["dataset_order"])).any():
            raise ValueError("invalid Slot statistics dataset route")
        mean = torch.tensor([stats["datasets"][key]["q5_mean"] for key in stats["dataset_order"]], device=device)[route]
        std = torch.tensor([stats["datasets"][key]["q5_std"] for key in stats["dataset_order"]], device=device)[route]
    else:
        mean = torch.tensor(stats["q5_mean"], device=device, dtype=torch.float32)
        std = torch.tensor(stats["q5_std"], device=device, dtype=torch.float32)
    std = std.clamp_min(config.slot_q5_std_floor)
    # Q5 predicts normalized displacement; consistency and diagnostics use meters.
    meters = p["Q5"] * std + mean
    losses["Q5"] = masked_sample_mean(smooth(p["Q5"], (target("Q5") - mean) / std).mean(-1), mask("Q5"))
    residual = meters[:, 2] - (meters[:, 1] - meters[:, 0])
    losses["Q5"] += config.slot_q5_consistency_weight * smooth(residual, torch.zeros_like(residual)).mean(-1) * mask("Q5").all(-1)
    metric("Q5_squared_m", (meters - target("Q5")).square(), mask("Q5").unsqueeze(-1).expand_as(meters))
    metric("Q5_consistency_m", residual.abs().mean(-1), mask("Q5").all(-1))
    losses["Q6"] = masked_sample_mean(smooth(p["Q6"], target("Q6")).mean(-1), mask("Q6"))
    metric("Q6_point_l1", (p["Q6"] - target("Q6")).abs().mean(-1), mask("Q6"))
    obstacles = p["Q9"]
    losses["Q9"] = config.slot_presence_weight * masked_sample_mean(
        F.binary_cross_entropy_with_logits(obstacles[..., 5], target("Q9_presence"), reduction="none"), mask("Q9_presence"))
    losses["Q9"] += config.slot_obstacle_bbox_weight * masked_sample_mean(bbox_loss(obstacles[..., :4], target("Q9_bbox")), mask("Q9_bbox"))
    losses["Q9"] += config.slot_risk_weight * masked_sample_mean(smooth(obstacles[..., 4], target("Q9_risk")), mask("Q9_risk"))
    positive, actual, supervised = obstacles[..., 5] >= 0, target("Q9_presence").bool(), mask("Q9_presence")
    metrics["slot_metric_Q9_presence_counts"] = torch.stack(((positive & actual & supervised).sum(),
        (positive & ~actual & supervised).sum(), (~positive & actual & supervised).sum(), supervised.sum())).float()
    metric("Q9_bbox_l1", (obstacles[..., :4] - target("Q9_bbox")).abs().mean(-1), mask("Q9_bbox"))
    metric("Q9_risk_l1", (obstacles[..., 4] - target("Q9_risk")).abs(), mask("Q9_risk"))
    out, raw = {}, zero
    for q in config.slot_task_weights:
        numerator = torch.where(valid[q], losses[q], 0).sum() + zero
        count = valid[q].sum().float()
        out[f"slot_{q}_loss_sum"], out[f"slot_{q}_count"] = numerator, count
        raw = raw + config.slot_task_weights[q] * numerator / count.clamp_min(1)
    out.update(metrics)
    out.update(slot_loss_raw=raw, slot_loss_weighted=raw * config.slot_loss_weight,
               slot_batch_samples=torch.tensor(float(size), device=device),
               slot_valid_samples=torch.stack(list(valid.values())).any(0).sum().float())
    return out


def finalize_slot_metrics(reduced, counts, config):
    result, raw = {}, torch.zeros((), device=counts.device)
    for i, q in enumerate(config.slot_task_weights):
        count = counts[i]
        result[f"slot_{q}_valid_count"] = count
        result[f"slot_{q}_available"] = bool(count > 0)
        if count > 0:
            value = reduced[f"slot_{q}_loss_sum"] / count
            weighted = value * config.slot_task_weights[q] * config.slot_loss_weight
            result.update({f"slot_{q}_loss_raw": value, f"slot_{q}_loss_weighted": weighted})
            raw = raw + value * config.slot_task_weights[q]
    result["slot_sample_coverage"] = reduced["slot_valid_samples"] / reduced["slot_batch_samples"].clamp_min(1)
    result["slot_loss_computed"] = bool(counts.sum() > 0)
    if counts.sum() > 0:
        result.update(slot_loss_raw=raw, slot_loss_weighted=raw * config.slot_loss_weight)
    for key, value in reduced.items():
        if not key.startswith("slot_metric_"):
            continue
        name = key[len("slot_metric_"):]
        if name.endswith("_confusion"):
            prefix = name.removesuffix("_confusion")
            if value.sum() > 0:
                tp = value.diag()
                result[f"slot_{prefix}_accuracy"] = tp.sum() / value.sum()
                result[f"slot_{prefix}_macro_f1"] = (2 * tp / (value.sum(0) + value.sum(1)).clamp_min(1)).mean()
        elif name == "Q9_presence_counts":
            tp, fp, fn, count = value
            result["slot_Q9_presence_available"] = bool(count > 0)
            if count > 0:
                result.update(slot_Q9_presence_precision=tp / (tp + fp).clamp_min(1),
                              slot_Q9_presence_recall=tp / (tp + fn).clamp_min(1),
                              slot_Q9_presence_f1=2 * tp / (2 * tp + fp + fn).clamp_min(1))
        elif name.endswith("_sum"):
            count = reduced[key.removesuffix("_sum") + "_count"]
            result["slot_" + name.removesuffix("_sum") + "_available"] = bool(count > 0)
            if count > 0:
                metric_name = name.removesuffix("_sum")
                result["slot_" + ("Q5_rmse_m" if metric_name == "Q5_squared_m" else metric_name)] = (value / count).sqrt() if metric_name == "Q5_squared_m" else value / count
    return result

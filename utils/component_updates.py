"""Opt-in component updates at the native AdamW boundary, including ZeRO-2."""

import copy
import json
from pathlib import Path

import torch


def component_activity(activity, *, detach_action=False):
    ar = bool(activity.get("ar_active_token_count", 0) > 0)
    fm = bool(activity.get("fm_active_element_count", 0) > 0)
    slot = bool(activity.get("slot_active_supervision_count", 0) > 0)
    flow = bool(activity.get("flow_active_sample_count", 0) > 0)
    shared = ar or slot or flow or (fm and not detach_action)
    return dict(vlm=shared, difference_query=shared, action_expert=fm,
                slot_aux=slot, optical_flow_aux=flow)


def _snapshot(group, optimizer):
    return [(p.detach().cpu().clone(), copy.deepcopy({
        k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
        for k, v in optimizer.state.get(p, {}).items()})) for p in group["params"]]


def _delta(before, after):
    if set(before) != set(after):
        return float("inf")
    total = 0.
    for key, old in before.items():
        new = after[key]
        if isinstance(old, torch.Tensor):
            total += float((new.detach().cpu().double() - old.double()).square().sum())
        elif old != new:
            return float("inf")
    return total ** .5


class ComponentUpdateGuard:
    """Remove inactive gradients in public native step hooks; never edit moments."""

    def __init__(self, optimizer, scheduler, *, diagnostics=False, log_path=None, diagnostics_mode=None):
        if type(optimizer) is not torch.optim.AdamW:
            raise RuntimeError("component updates require native torch AdamW")
        self.optimizer, self.scheduler = optimizer, scheduler
        self.groups = list(optimizer.param_groups)
        self.names = [g.get("component") for g in self.groups]
        if any(n not in component_activity({}) for n in self.names) or len(set(self.names)) != len(self.names):
            raise RuntimeError("component updates require distinct named component groups")
        if diagnostics_mode not in (None, "full", "inactive"):
            raise ValueError("component diagnostics must be full or inactive")
        self.diagnostics = diagnostics if diagnostics_mode is None else diagnostics_mode == "full"
        self.native_gradient_diagnostics = diagnostics_mode != "inactive"
        self.log_path = Path(log_path) if log_path is not None else None
        self.pending = False
        self.handles = [optimizer.register_step_pre_hook(self._before_step),
                        optimizer.register_step_post_hook(self._after_step)]

    def begin(self, activity, *, detach_action=False, global_step=None):
        if self.pending:
            raise RuntimeError("unfinished component update window")
        if self.log_path is not None and (type(global_step) is not int or global_step < 0):
            raise ValueError("component diagnostic logs require the current global update count")
        self.active = component_activity(activity, detach_action=detach_action)
        self.calls = dict.fromkeys(self.names, 0)
        self.metrics = {}
        self.global_step = global_step
        self.scheduler_before = self.scheduler.state_dict()["last_epoch"]
        self.old_lrs = [g["lr"] for g in self.groups]
        scheduled = self.scheduler.get_last_lr()
        if len(scheduled) != len(self.groups):
            raise RuntimeError("component scheduler group count changed")
        self.before = {}
        for i, group in enumerate(self.groups):
            name = group["component"]
            if self.active[name]:
                group["lr"] = scheduled[i]
            if self.diagnostics or not self.active[name]:
                self.before[name] = _snapshot(group, self.optimizer)
            self.metrics[name + "_active_supervision"] = self.active[name]
            self.metrics[name + "_lr_used"] = float(group["lr"])
        self.pending = True

    def _before_step(self, optimizer, args, kwargs):
        if not self.pending:
            raise RuntimeError("native optimizer step outside a supervised window")
        self.removed = []
        # ZeRO-2 calls the native optimizer once for each FP32 partition group.
        for group in optimizer.param_groups:
            name = group.get("component")
            if name not in self.calls:
                raise RuntimeError("unknown native optimizer component")
            self.calls[name] += 1
            if self.calls[name] != 1:
                raise RuntimeError("component optimizer updated more than once in one window")
            norm = 0.
            for parameter in group["params"]:
                if self.native_gradient_diagnostics and parameter.grad is not None:
                    norm += float(parameter.grad.detach().double().square().sum())
                if not self.active[name]:
                    self.removed.append((parameter, parameter.grad))
                    parameter.grad = None
            if self.native_gradient_diagnostics:
                self.metrics[name + "_native_grad_norm"] = norm ** .5

    def _after_step(self, optimizer, args, kwargs):
        for parameter, gradient in self.removed:
            parameter.grad = gradient
        self.removed = []

    def finish(self, *, applied):
        if not self.pending:
            raise RuntimeError("component window was not initialized")
        scheduler_after = self.scheduler.state_dict()["last_epoch"]
        if scheduler_after != self.scheduler_before + int(applied):
            raise RuntimeError("global scheduler did not advance exactly once per successful update")
        self.metrics.update(scheduler_step_before=self.scheduler_before, scheduler_step_after=scheduler_after)
        for i, group in enumerate(self.groups):
            name = group["component"]
            if self.calls[name] != int(applied):
                raise RuntimeError(f"component update outcome is unknown: {name}")
            changed = applied and self.active[name]
            self.metrics[name + "_update_applied"] = changed
            if name in self.before:
                after = _snapshot(group, self.optimizer)
                old = self.before[name]
                parameter_delta = sum(float((b[0].double() - a[0].double()).square().sum())
                                      for a, b in zip(old, after)) ** .5
                # Newly initialized state is recorded separately from its finite norm.
                created = sum(set(b[1]) != set(a[1]) for a, b in zip(old, after))
                state_delta = sum(_delta(a[1], {k: b[1][k] for k in a[1]}) ** 2
                                  for a, b in zip(old, after)) ** .5
                if not changed and (parameter_delta != 0 or state_delta != 0 or created):
                    raise RuntimeError(f"inactive/skipped component changed: {name}")
                self.metrics.update({name + "_parameter_delta": parameter_delta,
                    name + "_optimizer_state_delta": state_delta, name + "_optimizer_states_created": created})
            steps = [float(self.optimizer.state.get(p, {}).get("step", 0)) for p in group["params"]]
            self.metrics[name + "_optimizer_step_min"] = min(steps)
            self.metrics[name + "_optimizer_step_max"] = max(steps)
            if not self.active[name] or not applied:
                group["lr"] = self.old_lrs[i]
            self.metrics[name + "_lr_after"] = float(group["lr"])
        self.pending = False
        self.before.clear()
        if self.log_path is not None:
            with self.log_path.open("a") as stream:
                json.dump({"global_step_before": self.global_step,
                    "global_step_after": self.global_step + int(applied),
                    "global_update_applied": applied, **self.metrics}, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
        return self.metrics


def attach_component_guard(model, optimizer, scheduler, accelerator, *, diagnostics=False,
                           diagnostics_directory=None, diagnostics_mode=None):
    from accelerate.utils import DistributedType
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        zero = model.optimizer
        if not getattr(zero, "partition_gradients", False) or not hasattr(zero, "single_partition_of_fp32_groups"):
            raise RuntimeError("component updates require ZeRO-2")
        native = zero.optimizer
    else:
        native = getattr(optimizer, "optimizer", optimizer)
    path = (Path(diagnostics_directory) / f"component_updates_rank{accelerator.process_index}.jsonl"
            if diagnostics_directory is not None else None)
    guard = ComponentUpdateGuard(native, scheduler, diagnostics=diagnostics, log_path=path,
                                 diagnostics_mode=diagnostics_mode)
    accelerator.unwrap_model(model).component_update_guard = guard
    return guard

"""Explicit successful-update limits, independent of scheduler length."""



def validate_update_limits(options):
    stop = getattr(options, "save_and_exit_after_updates", None)
    skips = getattr(options, "max_consecutive_skipped_windows", None)
    component_diagnostics = getattr(options, "component_update_diagnostics", None)
    if component_diagnostics is not None and not getattr(options, "component_optimizer_groups", False):
        raise ValueError("component diagnostics require component optimizer groups")
    if getattr(options, "fast_resume_data_skip", False) and not (getattr(options, "training_stage", None)
            or getattr(options, "downstream_runtime_state", False)):
        raise ValueError("fast data resume requires an explicit three-stage dataset contract")
    if stop is not None and (type(stop) is not int or stop < 1):
        raise ValueError("save_and_exit_after_updates must be positive")
    if skips is not None and (type(skips) is not int or skips < 1):
        raise ValueError("max_consecutive_skipped_windows must be positive")
    if getattr(options, "component_optimizer_groups", False) and not options.training_stage:
        raise ValueError("component optimizer groups require an explicit stage")


def execution_limit(options, scheduler_steps):
    validate_update_limits(options)
    limits = [scheduler_steps]
    if getattr(options, "save_and_exit_after_updates", None) is not None:
        limits.append(options.save_and_exit_after_updates)
    return min(limits)


def check_next_update(completed, limit):
    if completed >= limit:
        raise RuntimeError(f"optimizer update budget exhausted: {completed}/{limit}")


def consecutive_skips(previous, applied, limit):
    count = 0 if applied else previous + 1
    if limit is not None and count >= limit:
        raise RuntimeError(f"stopped after {count} consecutive windows without a successful optimizer update")
    return count

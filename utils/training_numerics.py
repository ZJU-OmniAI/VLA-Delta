import math

import torch


def assert_all_finite(accelerator, value, name: str, step: int) -> None:
    if isinstance(value, torch.Tensor):
        local_finite = torch.isfinite(value.detach()).all()
    else:
        local_finite = torch.tensor(
            math.isfinite(float(value)), device=accelerator.device
        )

    finite_count = accelerator.reduce(
        # Offloaded optimizers may report CPU norms; collectives must use the
        # accelerator device (CUDA for NCCL). Move only the scalar validity flag.
        local_finite.to(device=accelerator.device, dtype=torch.long), reduction="sum"
    )
    if int(finite_count.item()) != accelerator.num_processes:
        raise FloatingPointError(
            f"Non-finite {name} detected across training ranks at step {step}."
        )

import torch


def get_default_device() -> torch.device:
    """Return the first available accelerator: CUDA, then Intel XPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    return torch.device("cpu")

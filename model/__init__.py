"""Model exports; independent auxiliary heads do not load the full VLM stack."""

__all__ = ["ZR0Model", "FlowmatchingActionHeadConfig"]


def __getattr__(name):
    if name == "ZR0Model":
        from .reasoning_vla_model import ZR0Model

        globals()[name] = ZR0Model
        return ZR0Model
    if name == "FlowmatchingActionHeadConfig":
        from .flow_matching_action_head import FlowmatchingActionHeadConfig

        globals()[name] = FlowmatchingActionHeadConfig
        return FlowmatchingActionHeadConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

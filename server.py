import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "lerobot"))
from utils.cli_options import parse_server_options
from utils.websocket_server_policy import WebsocketPolicyServer
import random
import numpy as np
import torch
import json

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_option(args=None):
    return parse_server_options(args)

def deploy():
    opt = parse_option()
    set_all_seeds(opt.seed)
    from utils.runtime_config import configure_datasets
    configure_datasets(opt.dataset_registry)
    from policies.reasoning_vla_policy import ZR0Policy

    assert opt.inference_mode in ["direct_action", "subtask_then_action"]
    # settings
    kwargs = {
        "dataset_entry": opt.dataset_entry,
        "ckpt_dir": opt.ckpt_dir,
        "stats_key": getattr(opt, "stats_key", None),
        "stats_file": opt.stats_file,
        "inference_mode": opt.inference_mode,
        "window_size": opt.window_size,
        "num_denoised_steps": opt.num_denoised_steps,
        "max_pad_state_and_action_length": opt.max_pad_state_and_action_length,
        "device": opt.device,
        "compile_model": opt.compile_model,
        "use_difference_query": opt.use_difference_query,
        "num_difference_queries": opt.num_difference_queries,
        "vlm_attention_backend": opt.vlm_attention_backend,
        "action_conditioning_norm": getattr(opt, "action_conditioning_norm", "final_rmsnorm"),
        "allow_legacy_checkpoint_without_manifest": (
            opt.allow_legacy_checkpoint_without_manifest
        ),
        "allow_legacy_checkpoint_without_observation_contract": getattr(
            opt,
            "allow_legacy_checkpoint_without_observation_contract",
            False,
        ),
    }
    print(json.dumps(kwargs, indent=2, ensure_ascii=False))
    policy = ZR0Policy(**kwargs)
    
    print("Start serving.")
    host = opt.host
    port = opt.port
    from utils.dataset_manifest import dataset_spec_to_manifest
    from pathlib import Path
    metadata = {"project": "VLA-∆", "checkpoint": str(Path(opt.ckpt_dir).resolve()),
        "dataset_entry": opt.dataset_entry, "action_dimension": policy.action_dim,
        "action_horizon": policy.model.action_expert_config.action_horizon,
        "num_denoised_steps": opt.num_denoised_steps, "seed": opt.seed,
        "dataset": dataset_spec_to_manifest(policy.dataset_spec, "action")}
    processor = policy.model.backbone.processor.image_processor
    metadata["preprocessing"] = {"final_image_size": [224, 224], "resize": "PIL RGB bicubic; square",
        "crop": False, "pad": False, "image_mean": processor.image_mean,
        "image_std": processor.image_std, "rescale_factor": processor.rescale_factor,
        "do_rescale": getattr(processor, "do_rescale", True),
        "do_normalize": getattr(processor, "do_normalize", True),
        "image_processor_resize": False}
    record = Path(opt.ckpt_dir) / "portable_run.json"
    if record.is_file():
        saved = json.loads(record.read_text())
        metadata["training_run"] = {"commit": saved.get("commit"), "wandb": saved.get("wandb"),
            "stage": saved["options"].get("training_stage"), "seed": saved["options"].get("seed")}
    server = WebsocketPolicyServer(policy, host, port, metadata=metadata)
    server.serve_forever()

if __name__ == "__main__":
    deploy()

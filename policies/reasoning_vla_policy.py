import torch
import numpy as np
import time

from typing import Dict
from torchvision.transforms import ToPILImage, ToTensor
from transformers import AutoProcessor
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from model import ZR0Model
from utils.load_training_dataset import prepare_qwen_vl_inputs_cpu, prepare_action_expert_inputs_cpu, custom_collate_fn
from utils.obs_buffer import ObservationBuffer
from utils.constants import DATASET2FEATURE
from utils.dataset_spec import (
    denormalize_actions,
    resolve_dataset_adapter_name,
    resolve_dataset_spec,
    resolve_inference_dataset_configs,
    resolve_objective_requirements,
)
from utils.vision_text import prepare_future_difference_direct_inputs
from utils.dataset import prepare_stage05_direct_inputs, STAGE05_CAMERA_LABELS
from utils.dataset_manifest import validate_policy_dataset_manifest
from policies.base_policy import BasePolicy

class ZR0Policy(BasePolicy):
    def __init__(
        self,
        dataset_entry,
        ckpt_dir,
        inference_mode,
        window_size,
        num_denoised_steps=5,
        max_pad_state_and_action_length=64,
        device="cuda:0",
        use_difference_query=None,
        num_difference_queries=None,
        vlm_attention_backend=None,
        allow_legacy_checkpoint_without_manifest=False,
        allow_legacy_checkpoint_without_observation_contract=False,
        stats_key=None,
        action_conditioning_norm="final_rmsnorm",
        compile_model=True,
        stats_file=None,
    ):
        super().__init__()
        # env params
        self.dataset_entry = dataset_entry
        self.ckpt_dir = ckpt_dir
        self.window_size = window_size
        self.num_denoised_steps = num_denoised_steps
        self.max_pad_state_and_action_length = max_pad_state_and_action_length
        self.device = device
        self.inference_mode = inference_mode
        self.prompt_suffix = "" if inference_mode == "direct_action" else "Sub task:"

        # load relevant configurations from the constant pool
        if dataset_entry not in DATASET2FEATURE:
            raise ValueError(f"unknown dataset entry {dataset_entry!r}")
        dataset_config = resolve_inference_dataset_configs(
            {dataset_entry: DATASET2FEATURE[dataset_entry]}, ckpt_dir,
            stats_file=stats_file,
        )[dataset_entry]
        self.dataset_path = dataset_config["dataset_path"]
        adapter = resolve_dataset_adapter_name(dataset_config)
        self.dataset_adapter = adapter
        if adapter == "stage05_mixed_pretraining":
            configured_stats_key = dataset_config.get("stats_key")
            if not stats_key or stats_key != configured_stats_key:
                raise ValueError(
                    f"{dataset_entry}: explicit stats_key must equal {configured_stats_key!r}"
                )
        if (
            adapter == "lerobot_v3_future_difference"
            and inference_mode != "direct_action"
        ):
            raise ValueError(
                f"{dataset_entry}: adapter {adapter!r} does not support inference_mode "
                f"{inference_mode!r}; only 'direct_action' is supported"
            )
        if adapter == "lerobot_v3_future_difference" and window_size != 1:
            raise ValueError(
                f"{dataset_entry}: v3 direct-action policy requires window_size=1 "
                "for current images"
            )

        # load VLM's processor
        self.processor = AutoProcessor.from_pretrained(ckpt_dir)
        # load dataset's metadata
        self.dataset_meta = None
        if adapter == "lerobot_v2":
            self.dataset_meta = LeRobotDatasetMetadata(
                repo_id=self.dataset_path.split("/")[-1],
                root=self.dataset_path,
                load_stats=stats_file is None,
            )
        # load model
        self.model = ZR0Model.from_pretrained(
            ckpt_dir,
            for_action_inference=True,
            checkpoint_load_purpose="inference",
            use_difference_query=use_difference_query,
            num_difference_queries=num_difference_queries,
            vlm_attention_backend=vlm_attention_backend,
            action_conditioning_norm=action_conditioning_norm,
        ).to(device).to(torch.bfloat16)
        self.model.eval()

        inference_requirements = resolve_objective_requirements(
            "action",
            adapter=adapter,
            target_text_field=dataset_config.get("target_text_field"),
            dataset_type=str(dataset_config.get("dataset_type", "vla")),
            dataset_entry=dataset_entry,
        )
        self.dataset_spec = resolve_dataset_spec(
            dataset_entry,
            dataset_config,
            action_horizon=self.model.action_expert_config.action_horizon,
            window_size=window_size,
            requirements=inference_requirements,
            v2_metadata=self.dataset_meta,
            checkpoint_directory=ckpt_dir,
        )
        validate_policy_dataset_manifest(
            self.dataset_spec,
            ckpt_dir,
            allow_legacy_missing=allow_legacy_checkpoint_without_manifest,
            external_stats=stats_file is not None,
            allow_legacy_missing_observation_contract=(
                allow_legacy_checkpoint_without_observation_contract
            ),
        )
        model_action_dim = getattr(
            self.model.action_expert_config, "action_dim", max_pad_state_and_action_length
        )
        model_state_dim = getattr(
            self.model.action_expert_config, "state_dim", max_pad_state_and_action_length
        )
        if (
            model_action_dim != max_pad_state_and_action_length
            or model_state_dim != max_pad_state_and_action_length
        ):
            raise ValueError(
                f"{dataset_entry}: policy padding dimension "
                f"{max_pad_state_and_action_length} does not match Action Expert "
                f"action/state dimensions {model_action_dim}/{model_state_dim}"
            )
        self.camera_keys = list(self.dataset_spec.camera_keys)
        self.grounding_camera_keys = list(self.dataset_spec.grounding_camera_keys)
        self.state_dim = self.dataset_spec.state_dim
        self.action_dim = self.dataset_spec.action_dim
        self.normalization_stats = self.dataset_spec.normalization_stats
        self.use_quantile = self.dataset_spec.normalization in {
            "quantile",
            "quantile_min_max_q01_q99",
        }
        # use `torch.compile` to speed up inference
        if compile_model:
            self.model = torch.compile(self.model, mode="default")
        
        # initialize buffer
        self.ob_buffer = ObservationBuffer(max_recent_observations = window_size)
        self.to_tensor = ToTensor()
        self.global_inference_steps = 0

    def _prepare_vl_inputs(self, data_sample: dict):
        if self.dataset_adapter == "stage05_mixed_pretraining":
            images = []
            for label, camera_key in zip(STAGE05_CAMERA_LABELS, self.camera_keys):
                value = data_sample.get(camera_key)
                if value is None:
                    if label == "wrist camera":
                        continue
                    raise ValueError(f"{self.dataset_entry}: main camera is required")
                if value.ndim == 4:
                    value = value[-1]
                images.append((label, ToPILImage()(value)))
            return prepare_stage05_direct_inputs(
                task=data_sample["task"], images=images, processor=self.processor
            )
        if self.dataset_adapter != "lerobot_v3_future_difference":
            return prepare_qwen_vl_inputs_cpu(
                data=data_sample,
                camera_keys=self.camera_keys,
                grounding_camera_keys=self.grounding_camera_keys,
                processor=self.processor,
                process_mode="eval",
                prompt_suffix=self.prompt_suffix,
                fast_tokenizer=None,
            )

        images = []
        for camera_key in self.camera_keys:
            value = data_sample[camera_key]
            if value.ndim == 4:
                if value.shape[0] != 1:
                    raise ValueError(
                        f"{self.dataset_entry}: v3 direct-action expects one current "
                        f"image for {camera_key!r}, got {value.shape[0]}"
                    )
                value = value[0]
            if value.ndim != 3:
                raise ValueError(
                    f"{self.dataset_entry}: camera {camera_key!r} must have 3 dimensions"
                )
            images.append((camera_key, ToPILImage()(value)))
        return prepare_future_difference_direct_inputs(
            task=data_sample["task"],
            images=images,
            processor=self.processor,
        )
        
    def infer(self, obs: Dict) -> Dict:
        task = obs.get('task')
        from utils.episode_input import validate_input_instruction
        task = validate_input_instruction(task, getattr(self.dataset_spec, "episode_input_contract", None))
        state = obs.get('observation.state')
        n_action_steps = obs.get('n_action_steps')
        if (
            isinstance(n_action_steps, bool)
            or not isinstance(n_action_steps, int)
            or not 0 < n_action_steps <= self.model.action_expert_config.action_horizon
        ):
            raise ValueError(
                f"{self.dataset_entry}: n_action_steps must be in "
                f"[1, {self.model.action_expert_config.action_horizon}]"
            )
        print("task:", task)
        if isinstance(state, np.ndarray):
            print("state:", state.tolist())
        else:
            print("state:", state)
        print("n_action_steps:", n_action_steps)

        # convert image from numpy to tensor
        multi_view_images = dict()
        for camera_index, cam_key in enumerate(self.camera_keys):
            image = obs.get(cam_key)
            if image is None:
                if self.dataset_adapter == "stage05_mixed_pretraining" and camera_index == 1:
                    continue
                raise ValueError(f"{self.dataset_entry}: observation is missing camera {cam_key!r}")
            multi_view_images[cam_key] = self.to_tensor(np.array(image, dtype=np.uint8))

        # manage historical observation buffer
        self.ob_buffer.add_observation(multi_view_images)
        
        # get the latest observations from the buffer
        available_camera_keys = list(multi_view_images)
        observations = self.ob_buffer.get_inference_time_observations(available_camera_keys)

        state = torch.as_tensor(state, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape != (1, self.state_dim) or not torch.isfinite(state).all():
            raise ValueError(
                f"{self.dataset_entry}: observation.state must have shape "
                f"[1, {self.state_dim}] with finite values"
            )
        
        data_sample = {
            "task": task, "observation.state": state
        } # "embodiment_id": self.embodiment_id, 
        for key, value in observations.items():
            data_sample[key] = value
        
        action_expert_inputs = prepare_action_expert_inputs_cpu(
            data_sample, self.normalization_stats, self.max_pad_state_and_action_length, self.use_quantile,
            self.model.action_expert_config.action_horizon, self.action_dim
        )
        vl_inputs = self._prepare_vl_inputs(data_sample)
        sub_task_flag = 0 if self.inference_mode == "direct_action" else 1

        # prepare batch input
        vla_input = custom_collate_fn([{**action_expert_inputs, **vl_inputs, "sub_task_flag": torch.tensor(sub_task_flag)}])
        for key, value in vla_input.items():
            vla_input[key] = value.to(self.device)

        # model inference
        with torch.no_grad():
            st = time.time()
            if self.inference_mode == "direct_action":
                vla_outputs = self.model.get_action_direct(vla_input, self.num_denoised_steps)
            elif self.inference_mode == "subtask_then_action":
                vla_outputs = self.model.get_action_subtask(vla_input, self.num_denoised_steps)
            print(f"vla forward takes {time.time()-st} seconds.")

        '''
        1. bs=1
        2. only retain the next `n_action_steps` steps in the chunk
        3. slice out the first few `action_dim` (the remaining dim are padded with 0 during training)
        '''
        action_chunk = vla_outputs["action_pred"][0][:n_action_steps]
        action_chunk = denormalize_actions(action_chunk, self.dataset_spec)
        action_chunk = action_chunk.tolist()

        print("action_chunk:")
        for action in action_chunk:
            print(action)

        self.global_inference_steps += 1
        print("-"*30)

        return {"actions": action_chunk}

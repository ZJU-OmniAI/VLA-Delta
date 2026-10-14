import torch
import copy
import hashlib
import io
import json
import math
import random
from pathlib import Path
try:
    import orjson
except ImportError:  # The standard-library fallback keeps parquet VQA loading usable.
    orjson = None

from PIL import Image
from datasets import load_dataset
from torchvision.transforms import ToPILImage
try:
    from transformers import AutoProcessor
except ImportError:
    class AutoProcessor:  # type: ignore[no-redef]
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("transformers is required to construct training datasets")
from utils.normalization import min_max_norm
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    class LeRobotDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("lerobot is required for the lerobot_v2 dataset adapter")

from torch.utils.data import ConcatDataset, DataLoader, Sampler
from utils.constants import DATASET2FEATURE
from utils.vision_text import (
    process_vision_info,
)
from utils.dataset_manifest import build_resolved_dataset_manifest
from utils.dataset_spec import (
    resolve_dataset_adapter_name,
    ObjectiveRequirements,
    ResolvedDatasetSpec,
    resolve_dataset_spec,
    resolve_objective_requirements,
    qwen_image_input_contract,
)
from utils.training_tokenization import (
    DatasetIntegrityError,
    TOKENIZATION_METRIC_SCHEMA,
    extract_final_assistant_target,
    is_transient_io_error,
    run_with_same_sample_retries,
    token_metric_validity_key,
    tokenize_chat_with_complete_assistant,
)

def pad_2d_to_max_length(tensor_2d, max_pad_length=64, pad_value=0.0):
    """
    Pads each row of a 2D tensor to max_length with pad_value and returns the corresponding mask.
    
    Args:
        tensor_2d (torch.Tensor): shape (N, L), L <= max_length
        max_length (int): pad length
        pad_value (int or float): value for padding
    
    Returns:
        padded_tensor (torch.Tensor): shape (N, max_length)
        mask (torch.Tensor): shape (N, max_length), 1=real, 0=pad
    """
    N, L = tensor_2d.shape
    if L >= max_pad_length:
        return tensor_2d[:, :max_pad_length], torch.ones((N, max_pad_length), dtype=torch.long, device=tensor_2d.device)
    pad_len = max_pad_length - L
    pad_tensor = torch.full((N, pad_len), pad_value, dtype=tensor_2d.dtype, device=tensor_2d.device)
    padded_tensor = torch.cat([tensor_2d, pad_tensor], dim=1)
    mask = torch.cat([
        torch.ones((N, L), dtype=torch.long, device=tensor_2d.device),
        torch.zeros((N, pad_len), dtype=torch.long, device=tensor_2d.device)
    ], dim=1)
    return padded_tensor, mask

def _v2_action_temporal_valid(data, *, identity="lerobot_v2 sample"):
    actions = data.get("action")
    action_is_pad = data.get("action_is_pad")
    if not isinstance(actions, torch.Tensor) or actions.ndim != 2:
        raise DatasetIntegrityError(f"{identity}: action must have shape [H, D]")
    if (
        not isinstance(action_is_pad, torch.Tensor)
        or action_is_pad.dtype != torch.bool
        or action_is_pad.shape != (actions.shape[0],)
    ):
        raise DatasetIntegrityError(
            f"{identity}: action_is_pad must be a bool tensor with shape "
            f"[{actions.shape[0]}]"
        )
    temporal_valid = ~action_is_pad
    if not temporal_valid.any():
        raise DatasetIntegrityError(f"{identity}: action horizon has no valid timestep")
    return temporal_valid


def prepare_action_expert_inputs_cpu(
    data,
    stats,
    max_pad_length,
    use_quantile,
    action_horizon=None,
    action_dim=None,
    *,
    dataset_entry="lerobot_v2",
    sample_id="sample=unknown",
):
    action_expert_inputs = {}
    # a scalar (shape is [])
    # if not isinstance(data["embodiment_id"], torch.Tensor):
    #     data["embodiment_id"] = torch.tensor(data["embodiment_id"])
    # action_expert_inputs["embodiment_id"] = data["embodiment_id"].to(torch.long)

    # 1. per-dim min-max norm; 2. dim-size padding + return mask
    data["norm_state_wo_pad"] = min_max_norm(data["observation.state"], stats["observation.state"], use_quantile) # (1, state_dim)
    padded_states, state_masks = pad_2d_to_max_length(data["norm_state_wo_pad"], max_pad_length)
    action_expert_inputs["observation.state"] = padded_states  # (1, max_pad_length)
    action_expert_inputs["state_mask"] = state_masks           # (1, max_pad_length)

    if "action" in data:
        identity = f"dataset_entry={dataset_entry} {sample_id}".strip()
        temporal_valid = _v2_action_temporal_valid(data, identity=identity)
        data["norm_action_wo_pad"] = min_max_norm(data["action"], stats["action"], use_quantile) # (action_horizon, action_dim)
        padded_actions, action_masks = pad_2d_to_max_length(data["norm_action_wo_pad"], max_pad_length)
        action_masks = action_masks.to(torch.bool) & temporal_valid[:, None]
        padded_actions = padded_actions.masked_fill(~action_masks, 0)
        action_expert_inputs["action"] = padded_actions    # (action_horizon, max_pad_length)
        action_expert_inputs["action_mask"] = action_masks # (action_horizon, max_pad_length)
        action_expert_inputs["action_supervision_available"] = torch.tensor(True)
        data["action_temporal_valid"] = temporal_valid
    else:
        infer_action_mask = torch.zeros((action_horizon, max_pad_length), dtype=torch.long)
        infer_action_mask[:, :action_dim] = 1
        action_expert_inputs["infer_action_mask"] = infer_action_mask # (action_horizon, max_pad_length)

    return action_expert_inputs

def convert_fast_tokens_to_vlm_action_seq(fast_tokens: list[int]) -> str:
    """
    convert fast action tokens to VLM's (special) action tokens.
    E.g., action tokens [0, 343, 745] are converted to the string "<robot_action_0><robot_action_343><robot_action_745>"
    """
    return ''.join([f"<robot_action_{token}>" for token in fast_tokens])


def _sample_scalar(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else fallback
    if isinstance(value, (int, str)):
        return value
    return fallback


def v2_sample_identity(data: dict, sample_index: int) -> str:
    episode = _sample_scalar(data.get("episode_index"), "unknown")
    frame = _sample_scalar(
        data.get("frame_index", data.get("index")), "unknown"
    )
    return f"episode={episode} frame={frame} sample={sample_index}"

def tokenize_vision_language_inputs(
    msg,
    process_mode,
    processor,
    max_length=1200,
    *,
    has_target=None,
    dataset_entry="legacy",
    sample_id="sample=unknown",
    target=None,
):
    if has_target is None:
        has_target = process_mode == "train"
    image_inputs, video_inputs = process_vision_info(msg, image_patch_size=16) # image_patch_size, 14 for Qwen2.5-VL and 16 for Qwen3-VL
    if process_mode == "train":
        processor.tokenizer.padding_side = "right"
        if has_target and target is None:
            try:
                target = extract_final_assistant_target(msg)
            except DatasetIntegrityError as error:
                raise DatasetIntegrityError(
                    f"dataset_entry={dataset_entry} {sample_id}: {error}"
                ) from error
        return tokenize_chat_with_complete_assistant(
            msg,
            processor,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
            max_length=max_length,
            dataset_entry=dataset_entry,
            sample_id=sample_id,
            target=target if has_target else None,
        )

    text = processor.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=not has_target
    )
    vl_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        do_resize=False,
        return_tensors="pt",
    )
    
    # remove the batch dim of input_ids and attention_mask
    vl_inputs["input_ids"] = vl_inputs["input_ids"][0] # (seq_len,)
    vl_inputs["attention_mask"] = vl_inputs["attention_mask"][0] # (seq_len,)
    # pixel_values and image_grid_thw do not contain the batch dim
    vl_inputs["pixel_values"] = vl_inputs["pixel_values"] # (pixel_values_width, pixel_values_height)
    vl_inputs["image_grid_thw"] = vl_inputs["image_grid_thw"] # (image_grid_thw_width, image_grid_thw_height)

    return vl_inputs

def prepare_qwen_vl_inputs_cpu(
    data: dict,
    camera_keys: list[str],
    grounding_camera_keys: list[str],
    processor: AutoProcessor,
    process_mode: str,
    prompt_suffix: str,
    fast_tokenizer: AutoProcessor,
    max_length: int = 1200,
    requirements: ObjectiveRequirements | None = None,
    target_text_field: str | None = None,
    dataset_entry: str = "lerobot_v2",
    sample_id: str = "sample=unknown",
):
    identity = f"dataset_entry={dataset_entry} {sample_id}".strip()
    to_pil = ToPILImage()
    vision_contract = qwen_image_input_contract()
    resized_height = vision_contract["image_height"]
    resized_width = vision_contract["image_width"]

    # store contextual images (history + current)
    context_images = []
    if len(data[camera_keys[0]].shape) == 4:
        # for each camera, the lerobot dataset gives several historical observations and one current observations
        # For example, torch.Size([4, 3, 480, 640]) means that it contains 3 historical obs and 1 current obs
        obs_num = data[camera_keys[0]].shape[0]
        for obs_idx in range(obs_num):
            for camera_key in camera_keys:
                cam_key = camera_key if obs_idx == obs_num - 1 else f"historical-observation-{obs_idx}.{camera_key}"
                context_images.append([cam_key, to_pil(data[camera_key][obs_idx])])
    elif len(data[camera_keys[0]].shape) == 3:
        # for each camera, the lerobot dataset only gives one current observation
        for camera_key in camera_keys:
            context_images.append([camera_key, to_pil(data[camera_key])])
    else:
        raise DatasetIntegrityError(
            f"{identity}: expected image shape with 3 or 4 dimensions for "
            f"data['{camera_keys[0]}'], "
            f"but got shape with {len(data[camera_keys[0]].shape)} dimensions: {data[camera_keys[0]].shape}."
        )

    prompt = {"role": "user", "content": []}
    for cam_key, img in context_images:
        prompt["content"].append({"type": "text", "text": cam_key})
        prompt["content"].append({"type": "image", "image": img, "resized_height": resized_height, "resized_width": resized_width})

    prompt["content"].append({"type": "text", "text": "<TASK> " + data["task"].strip() + " <\\TASK>\n" + prompt_suffix})
    msg = [prompt]

    has_target = process_mode == "train" if requirements is None else requirements.requires_target
    output_seq = None
    if has_target:
        if target_text_field is not None:
            output_seq = data.get(target_text_field)
            if not isinstance(output_seq, str) or not output_seq.strip():
                raise DatasetIntegrityError(
                    f"{identity}: target_text_field {target_text_field!r} must "
                    "contain a non-empty string"
                )
        else:
            if fast_tokenizer is None:
                raise DatasetIntegrityError(
                    f"{identity}: legacy v2 target construction requires FAST tokenizer"
                )
            temporal_valid = data.get("action_temporal_valid")
            if temporal_valid is None:
                temporal_valid = _v2_action_temporal_valid(data, identity=identity)
            discrete_action_tokens = fast_tokenizer(
                data["norm_action_wo_pad"][temporal_valid]
            )
            discrete_action_tokens_seq = convert_fast_tokens_to_vlm_action_seq(discrete_action_tokens[0])

            ecot_str = data["embodied_cot"]
            try:
                ecot_json = json.loads(ecot_str)
            except (TypeError, json.JSONDecodeError) as error:
                raise DatasetIntegrityError(
                    f"{identity}: embodied_cot must be valid JSON"
                ) from error
            if prompt_suffix == "":
                ecot_json["Discrete Action Tokens"] = discrete_action_tokens_seq
                output_seq = json.dumps(ecot_json, indent=2, ensure_ascii=False)
            elif prompt_suffix == "Sub task:":
                if "To-do Actions" in ecot_json:
                    output_seq = "<SUB_TASK>" + ecot_json["To-do Actions"][0] + "</SUB_TASK>"
                else:
                    output_seq = "<SUB_TASK>done</SUB_TASK>"
        msg.append({"role": "assistant", "content": output_seq})
    
    vl_inputs = tokenize_vision_language_inputs(
        msg,
        process_mode,
        processor,
        max_length=max_length,
        has_target=has_target,
        dataset_entry=dataset_entry,
        sample_id=sample_id,
        target=output_seq,
    )
    
    return vl_inputs

def deterministic_test_time_n_action_steps(action_horizon: int, ds_id: int, local_idx: int, epoch: int) -> int:
    # Avoiding the random salt effects of Python's default hash with stable hashing
    key = f"{ds_id}-{local_idx}-{epoch}".encode("utf-8")
    h = hashlib.sha1(key).hexdigest()
    # Take the first 8 bytes as an integer
    val = int(h[:8], 16)
    return 1 + (val % action_horizon)

class StreamingLeRobotSampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lerobot_dataset: LeRobotDataset, # a LeRobot Dataset instance
        use_quantile: bool,
        sample_ratio: float,
        processor: AutoProcessor,
        fast_tokenizer: AutoProcessor,
        window_size: int,
        action_horizon: int,
        process_mode: str,
        dataset_id: int,
        max_pad_state_and_action_length: int,
        dataset_entry: str,
        max_length: int = 1200,
        requirements: ObjectiveRequirements | None = None,
        target_text_field: str | None = None,
        max_transient_retries: int = 2,
        episode_input=None,
        flow_reader=None,
    ):
        self.ds = lerobot_dataset
        self.use_quantile = use_quantile
        self.sample_ratio = sample_ratio
        self.processor = processor
        self.fast_tokenizer = fast_tokenizer
        self.window_size = max(1, window_size)
        self.action_horizon = action_horizon
        self.process_mode = process_mode
        self.dataset_id = dataset_id
        self.max_pad_state_and_action_length = max_pad_state_and_action_length
        self.dataset_entry = dataset_entry
        self.max_length = max_length
        self.requirements = requirements or resolve_objective_requirements(
            "vlm_and_action",
            adapter="lerobot_v2",
            target_text_field=target_text_field,
        )
        self.target_text_field = target_text_field
        self.max_transient_retries = max_transient_retries
        self.episode_input = episode_input
        self.flow_reader = flow_reader

        self.fps = self.ds.meta.fps
        self.camera_keys = self.ds.meta.camera_keys
        self.stats = self.ds.meta.stats if self.requirements.requires_stats else None
        self.grounding_camera_keys = tuple(self.ds.meta.grounding_camera_keys or ())

        self.data_source = self.ds
        if not self.requirements.requires_action:
            required_columns = {
                "episode_index",
                "task_index",
                "timestamp",
                "index",
                *self.camera_keys,
            }
            if self.requirements.requires_target and self.target_text_field:
                required_columns.add(self.target_text_field)
            available_columns = set(self.ds.hf_dataset.column_names)
            projection = sorted(required_columns & available_columns)
            missing = required_columns - set(projection) - set(self.ds.meta.video_keys)
            if missing:
                raise ValueError(
                    f"{dataset_entry}: required AR-only columns are missing: {sorted(missing)}"
                )
            self.data_source = copy.copy(self.ds)
            self.data_source.hf_dataset = self.ds.hf_dataset.select_columns(projection)

        self._epoch = 0  # used for deterministic randomness
        self._init_sampled_indices()

        self.ecot_supported = (
            self.requirements.requires_fast_tokenizer
            and self.is_ecot_enhanced(self.ds.hf_dataset)
        )
        if self.ecot_supported:
            print(f"{dataset_entry} is ECoT-enhanced.")
        else:
            print(f"{dataset_entry} is not ECoT-enhanced.")

    def is_ecot_enhanced(self, hf_dataset) -> bool:
        """
        Check if a Lerobot dataset is ECoT-enhanced 
        (contains required annotation columns).
        
        Args:
            hf_dataset: HuggingFace Dataset object or its .features mapping.
        
        Returns:
            bool: True if dataset contains all required ECoT columns, False otherwise.
        """
        # Required columns for ECoT-enhanced datasets
        ecot_required_columns = {"future_sub_tasks", "bbox", "cot"}
        
        # Extract dataset features mapping
        features = getattr(hf_dataset, "features", hf_dataset)
        
        # Check if the dataset contains all required columns
        dataset_columns = set(features.keys())
        return ecot_required_columns.issubset(dataset_columns)

    def _init_sampled_indices(self):
        if getattr(self, "fixed_probe_order", False):
            self.subset_indices = self.probe_indices
            return
        total = len(self.ds)
        n_samples = max(1, int(total * self.sample_ratio))
        rng = random.Random(42 + self._epoch)
        self.subset_indices = rng.sample(range(total), n_samples)

    def __len__(self):
        return len(self.subset_indices)

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._init_sampled_indices()

    def _obtain_delta_timestamps(self, test_time_n_action_steps: int):
        delta_timestamps = dict()

        # During inference, we set `test_time_n_action_steps` to an integer between [1, action_horizon]. 
        # Thus, during training, the observation images in the prompt should come from the delta indices: [-(N-1)*test_time_n_action_steps, ..., -1*test_time_n_action_steps, 0],
        # where N is `window_size` and `test_time_n_action_steps` is a random integer between 1 and action_horizon
        slide_window_observation_delta_indices = list(range(self.window_size))[::-1]
        slide_window_observation_delta_indices = [-idx * test_time_n_action_steps for idx in slide_window_observation_delta_indices]

        if self.window_size > 1:
            # set camera's delta timestamps
            for camera_key in self.camera_keys:
                delta_timestamps[camera_key] = [index / self.fps for index in slide_window_observation_delta_indices]

        if self.requirements.requires_action:
            action_delta_indices = list(range(self.action_horizon))
            delta_timestamps["action"] = [idx / self.fps for idx in action_delta_indices]
        return delta_timestamps

    def _get_item_once(self, sample_index: int):
        test_time_n_action_steps = deterministic_test_time_n_action_steps(
            self.action_horizon,
            ds_id=self.dataset_id,
            local_idx=sample_index,
            epoch=self._epoch,
        )
        delta_timestamps = self._obtain_delta_timestamps(test_time_n_action_steps)
        data_sample = self.data_source.getitem_with_delta_timestamps(
            sample_index, delta_timestamps
        )
        if getattr(self, "episode_input", None) is not None:
            episode_id = int(data_sample["episode_index"])
            if episode_id not in self.episode_input.instructions:
                raise DatasetIntegrityError(f"{self.dataset_entry}: episode outside fixed input manifest: {episode_id}")
            data_sample = dict(data_sample, task=self.episode_input.instructions[episode_id])
        if getattr(self, "downstream_contract_version", 1) == 2:
            from utils.downstream_contracts import validate_source_images
            validate_source_images(data_sample, self.spec)
            ep = int(data_sample["episode_index"])
            frame = sample_index - self.downstream_episode_offsets[ep]
            if not 0 <= frame < self.ds.meta.episodes[ep]["length"] or abs(float(data_sample["timestamp"]) - frame/self.fps) > 5e-4:
                raise DatasetIntegrityError(f"{self.dataset_entry}/{ep}/{frame}: frame/timestamp alignment changed")
            if "frame_index" in data_sample and int(data_sample["frame_index"]) != frame:
                raise DatasetIntegrityError("source frame_index differs from episode-relative row")
            # Published GR1 v2 Parquet has no frame_index column. Identity is
            # the episode-relative row, validated against its source timestamp.
            data_sample = dict(data_sample, frame_index=torch.tensor(frame))
        sample_id = v2_sample_identity(data_sample, sample_index)

        try:
            result = self._preprocess_item(data_sample, sample_id)
            if getattr(self, "flow_reader", None) is not None:
                result.update(self.flow_reader.read(
                    int(data_sample["episode_index"]), int(data_sample["frame_index"])))
            return result
        except DatasetIntegrityError:
            raise
        except Exception as error:
            if is_transient_io_error(error):
                raise
            raise DatasetIntegrityError(
                f"dataset_entry={self.dataset_entry} {sample_id}: {error}"
            ) from error

    def _preprocess_item(self, data_sample: dict, sample_id: str):

        action_expert_inputs = {}
        if self.requirements.requires_state:
            action_expert_inputs = prepare_action_expert_inputs_cpu(
                data_sample,
                self.stats,
                self.max_pad_state_and_action_length,
                self.use_quantile,
                dataset_entry=self.dataset_entry,
                sample_id=sample_id,
            )
        prompt_suffix = ""
        sub_task_flag = 0
        if self.ecot_supported and random.random() < 0.1:
            prompt_suffix = "Sub task:"
            sub_task_flag = 1

        vl_inputs = prepare_qwen_vl_inputs_cpu(
            data=data_sample,
            camera_keys=self.camera_keys,
            grounding_camera_keys=self.grounding_camera_keys,
            processor=self.processor,
            process_mode=self.process_mode,
            prompt_suffix=prompt_suffix,
            fast_tokenizer=self.fast_tokenizer,
            max_length=self.max_length,
            requirements=self.requirements,
            target_text_field=self.target_text_field,
            dataset_entry=self.dataset_entry,
            sample_id=sample_id,
        )
        result = {
            **action_expert_inputs,
            **vl_inputs,
            "sub_task_flag": torch.tensor(sub_task_flag),
        }
        if getattr(self, "downstream_contract_version", 1) == 2:
            if self.spec.action_dim == 44:
                lo = torch.as_tensor(self.stats["action"]["min"])
                hi = torch.as_tensor(self.stats["action"]["max"])
                result["gr1_constant_dimensions"] = lo == hi
            result.update(downstream_dataset_entry=self.dataset_entry,
                downstream_episode_id=data_sample["episode_index"].reshape(()),
                downstream_frame_id=data_sample["frame_index"].reshape(()),
                downstream_time_valid=~data_sample["action_is_pad"],
                downstream_camera_keys=list(self.camera_keys))
        if getattr(self, "track_downstream", False):
            result.update(dataset_id=torch.tensor(self.dataset_id),
                          sample_global_index=torch.as_tensor(data_sample["index"]).reshape(()),
                          ar_eligible=torch.tensor(False), fm_eligible=torch.tensor(True))
        return result

    def __getitem__(self, idx: int):
        sample_index = self.subset_indices[idx]
        return run_with_same_sample_retries(
            lambda: self._get_item_once(sample_index),
            dataset_entry=self.dataset_entry,
            sample_id=f"sample={sample_index}",
            max_transient_retries=self.max_transient_retries,
        )

class HFDatasetWrapper:
    def __init__(self, path, split="train"):
        self.ds = load_dataset(
            "parquet",
            data_dir=path,
            split=split,
            keep_in_memory=False,
        )

    def get(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)

class VQADataset(torch.utils.data.Dataset):
    """
    Parquet + image-bytes VQA Dataset.
    Safe for:
      - DataLoader(num_workers > 0)
      - DDP / multi-node
    """

    def __init__(
        self,
        dataset_path: str,
        sample_ratio: float,
        processor: AutoProcessor,
        process_mode: str = "train",
        max_pad_state_and_action_length: int = 64,
        action_horizon: int = 32,
        max_length: int = 1200,
        dataset_entry: str = "vqa_parquet",
        max_transient_retries: int = 2,
    ):
        self.processor = processor
        self.process_mode = process_mode
        self.max_pad_state_and_action_length = max_pad_state_and_action_length
        self.action_horizon = action_horizon
        self.sample_ratio = sample_ratio
        self.max_length = max_length
        self.dataset_entry = dataset_entry
        self.max_transient_retries = max_transient_retries

        # parquet dataset
        self.hf_ds = HFDatasetWrapper(f"{dataset_path}/data")

        self._epoch = 0
        self._init_sampled_indices()

    def _init_sampled_indices(self):
        total = len(self.hf_ds)
        n_samples = max(1, int(total * self.sample_ratio))
        rng = random.Random(42 + self._epoch)
        self.subset_indices = rng.sample(range(total), n_samples)

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._init_sampled_indices()

    def __len__(self):
        return len(self.subset_indices)

    def generate_dummy_action_expert_inputs(self):
        return {
            "observation.state": torch.zeros(
                1, self.max_pad_state_and_action_length
            ),
            "state_mask": torch.zeros(
                1, self.max_pad_state_and_action_length, dtype=torch.bool
            ),
            "action": torch.zeros(
                self.action_horizon, self.max_pad_state_and_action_length
            ),
            "action_mask": torch.zeros(
                self.action_horizon,
                self.max_pad_state_and_action_length,
                dtype=torch.bool,
            ),
            "action_supervision_available": torch.tensor(False),
        }

    def _decode_image(self, img_bytes):
        # PNG image bytes -> PIL.Image
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _inject_images(self, msg, image_bytes_dict):
        """
        msg: parsed JSON
        image_bytes_dict: {0: bytes, 1: bytes, 2: bytes}
        """
        for turn in msg:
            content = turn.get("content")
            if not isinstance(content, list):
                continue

            new_content = []
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "image"
                    and "image_index" in item
                ):
                    idx = item["image_index"]
                    img_bytes = image_bytes_dict[idx]
                    img = self._decode_image(img_bytes)

                    new_item = dict(item)
                    new_item.pop("image_index", None)
                    new_item["image"] = img
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            turn["content"] = new_content
        return msg

    def _get_item_once(self, real_idx: int):
        sample = self.hf_ds.get(real_idx)
        image_bytes_dict = {
            int(key[5:]): value
            for key, value in sample.items()
            if key.startswith("image") and value is not None
        }
        msg = orjson.loads(sample["json"]) if orjson is not None else json.loads(sample["json"])
        msg = self._inject_images(msg, image_bytes_dict)
        vl_inputs = tokenize_vision_language_inputs(
            msg,
            self.process_mode,
            self.processor,
            max_length=self.max_length,
            dataset_entry=self.dataset_entry,
            sample_id=f"sample={real_idx}",
        )
        action_expert_inputs = self.generate_dummy_action_expert_inputs()
        return {
            **action_expert_inputs,
            **vl_inputs,
            "sub_task_flag": torch.tensor(0),
        }

    def __getitem__(self, idx: int):
        real_idx = self.subset_indices[idx]
        return run_with_same_sample_retries(
            lambda: self._get_item_once(real_idx),
            dataset_entry=self.dataset_entry,
            sample_id=f"sample={real_idx}",
            max_transient_retries=self.max_transient_retries,
        )

def build_concat_streaming_dataset(
    dataset_entries: list[str],
    model_name_or_path: str,
    fast_tokenizer_path: str,
    window_size: int,
    action_horizon: int,
    accelerator,
    process_mode: str = "train",
    max_pad_state_and_action_length: int = 64,
    loss_type: str = "vlm_and_action",
    max_length: int = 1200,
    dataset_sample_ratios: list[float] | None = None,
    optical_flow_config=None,
    optical_flow_data_root=None,
    slot_config=None,
    slot_reader=None,
    aux_dataset_config=None,
    downstream_runtime_state=False,
    downstream_contract_version=1,
    downstream_probe_manifest=None,
):
    auxiliary_entries = None
    # An explicit route selects the production datasets/sidecars independently
    # of auxiliary objective switches.  Readers and label access remain gated
    # by the effective enabled flags below.
    if aux_dataset_config is not None:
        payload = json.loads(Path(aux_dataset_config).read_text())
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported auxiliary dataset configuration")
        auxiliary_entries = payload.get("datasets")
        if not isinstance(auxiliary_entries, dict) or not auxiliary_entries:
            raise ValueError("auxiliary dataset configuration requires nonempty datasets routes")
    if dataset_sample_ratios is not None and len(dataset_sample_ratios) != len(dataset_entries):
        raise ValueError("dataset_sample_ratios must have the same length as dataset_entries")
    if optical_flow_config is not None and optical_flow_config.enabled and not any(
        DATASET2FEATURE.get(name, {}).get("optical_flow_manifest")
        or (auxiliary_entries or {}).get(name, {}).get("optical_flow_manifest")
        for name in dataset_entries
    ):
        raise ValueError("enabled OF requires a registered per-dataset flow manifest")
    selected_entries = []
    needs_fast = False
    flow_label_source = 1
    for dataset_id, dataset_entry in enumerate(dataset_entries):
        if dataset_entry not in DATASET2FEATURE:
            raise ValueError(f"Unknown dataset entry {dataset_entry!r}")
        entry = dict(DATASET2FEATURE[dataset_entry])
        entry["dataset_entry"] = dataset_entry
        if auxiliary_entries is not None:
            if dataset_entry not in auxiliary_entries:
                raise ValueError(f"auxiliary dataset route missing: {dataset_entry}")
            route = auxiliary_entries[dataset_entry]
            if not isinstance(route, dict) or not route:
                raise ValueError(f"invalid auxiliary dataset route: {dataset_entry}")
            entry.update(route)
        entry["flow_enabled"] = bool(optical_flow_config is not None and optical_flow_config.enabled)
        entry["slot_enabled"] = bool(slot_config is not None and slot_config.enabled)
        if optical_flow_config is not None and optical_flow_config.enabled and entry.get("optical_flow_manifest"):
            entry.update(optical_flow_data_root=entry.get("optical_flow_data_root", optical_flow_data_root),
                         flow_delta_frames=entry.get("flow_delta_frames", optical_flow_config.flow_delta_frames),
                         flow_label_source=flow_label_source)
        if dataset_sample_ratios is not None:
            entry["sample_ratio"] = dataset_sample_ratios[dataset_id]
        sample_ratio = entry.get("sample_ratio")
        if (
            not isinstance(sample_ratio, (int, float))
            or not math.isfinite(sample_ratio)
            or not 0 < sample_ratio <= 1
        ):
            raise ValueError(
                f"Dataset entry {dataset_entry!r} sample_ratio must be finite and in (0, 1]"
            )
        dataset_type = entry["dataset_type"]
        adapter_name = None
        if dataset_type == "vla":
            adapter_name = resolve_dataset_adapter_name(entry)
            if adapter_name == "lerobot_v3_future_difference" and window_size != 1:
                raise ValueError(
                    f"{dataset_entry}: v3 datasets require window_size=1"
                )
        elif dataset_type == "vlm":
            adapter_name = "vqa_parquet"
        requirements = resolve_objective_requirements(
            loss_type,
            adapter=adapter_name,
            target_text_field=entry.get("target_text_field"),
            dataset_type=dataset_type,
            dataset_entry=dataset_entry,
        )
        needs_fast = needs_fast or requirements.requires_fast_tokenizer
        selected_entries.append(
            (dataset_id, dataset_entry, entry, adapter_name, requirements)
        )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    fast_tokenizer = (
        AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        if needs_fast
        else None
    )

    datasets = []
    for dataset_id, dataset_entry, entry, adapter_name, requirements in selected_entries:
        print(f"loading {dataset_entry}")
        dataset_path = entry["dataset_path"]
        sample_ratio = entry["sample_ratio"]
        dataset_type = entry["dataset_type"]

        if dataset_type == "vla":
            if adapter_name == "lerobot_v2":
                from utils.episode_input import resolve_episode_input
                episode_input = resolve_episode_input(dataset_entry, entry)
                loader_options = {}
                if episode_input is not None:
                    loader_options["episodes"] = list(episode_input.episode_ids)
                if downstream_probe_manifest:
                    from utils.downstream_probe_data import read_probe
                    rows = read_probe(downstream_probe_manifest, dataset_entries)
                    loader_options["episodes"] = sorted({ep for name, ep, frame in rows if name == dataset_entry})
                for option in ("hf_loading_workers", "video_backend"):
                    if option in entry:
                        loader_options[option] = entry[option]
                lerobot_dataset = LeRobotDataset(
                    repo_id=dataset_path.split("/")[-1],
                    root=dataset_path,
                    load_stats=requirements.requires_stats,
                    **loader_options,
                )
                ds = StreamingLeRobotSampleDataset(
                    lerobot_dataset=lerobot_dataset,
                    use_quantile=entry["use_quantile"],
                    sample_ratio=sample_ratio,
                    processor=processor,
                    fast_tokenizer=fast_tokenizer,
                    window_size=window_size,
                    action_horizon=action_horizon,
                    process_mode=process_mode,
                    dataset_id=dataset_id,
                    max_pad_state_and_action_length=max_pad_state_and_action_length,
                    dataset_entry=dataset_entry,
                    max_length=max_length,
                    requirements=requirements,
                    target_text_field=entry.get("target_text_field"),
                    max_transient_retries=entry.get("max_transient_retries", 2),
                    episode_input=episode_input,
                )
                ds.spec = resolve_dataset_spec(
                    dataset_entry,
                    entry,
                    action_horizon=action_horizon,
                    window_size=window_size,
                    requirements=requirements,
                    v2_metadata=lerobot_dataset.meta,
                    downstream_contract_version=downstream_contract_version,
                )
                if entry["flow_enabled"]:
                    from dataclasses import replace
                    from utils.optical_flow_reader import OpticalFlowReader
                    from utils.aux_data_contract import digest_file
                    if not entry.get("optical_flow_manifest"):
                        raise ValueError(f"{dataset_entry}: enabled OF requires an explicit manifest")
                    flow_root = Path(entry["optical_flow_data_root"])
                    flow_manifest = flow_root / entry["optical_flow_manifest"]
                    ds.flow_reader = OpticalFlowReader(flow_root, flow_manifest,
                        delta_frames=optical_flow_config.flow_delta_frames,
                        label_source=flow_label_source)
                    ds.spec = replace(ds.spec, auxiliary_contract={
                        "optical_flow": {"root": str(flow_root), "manifest": str(flow_manifest),
                            "manifest_sha256": digest_file(flow_manifest),
                            "delta_frames": optical_flow_config.flow_delta_frames,
                            "join": ["episode_index", "frame_index"], "tail_policy": "FM only"}})
                    from utils.downstream_probe_data import configure_downstream_dataset
                    configure_downstream_dataset(ds, downstream_probe_manifest)
                    if downstream_probe_manifest:
                        ds.spec = replace(ds.spec, auxiliary_contract={**ds.spec.auxiliary_contract,
                            "probe_sha256": digest_file(downstream_probe_manifest)})
            else:
                if adapter_name == "stage05_mixed_pretraining":
                    from utils.dataset import Stage05MixedPretrainingDataset

                    adapter_factory = Stage05MixedPretrainingDataset
                else:
                    from utils.lerobot_v3_dataset import LeRobotV3FutureDifferenceDataset
                    adapter_factory = LeRobotV3FutureDifferenceDataset
                ds = adapter_factory(
                    entry=entry,
                    processor=processor,
                    loss_type=loss_type,
                    max_length=max_length,
                    action_horizon=action_horizon,
                    max_pad_state_and_action_length=max_pad_state_and_action_length,
                    dataset_id=dataset_id,
                )
        elif dataset_type == "vlm":
            ds = VQADataset(
                dataset_path=dataset_path,
                sample_ratio=sample_ratio,
                processor=processor,
                process_mode=process_mode,
                max_pad_state_and_action_length=max_pad_state_and_action_length,
                action_horizon=action_horizon,
                max_length=max_length,
                dataset_entry=dataset_entry,
                max_transient_retries=entry.get("max_transient_retries", 2),
            )
            ds.requirements = requirements
            ds.spec = ResolvedDatasetSpec(
                dataset_entry=dataset_entry,
                dataset_path=str(Path(dataset_path).expanduser().resolve()),
                dataset_type="vlm",
                adapter="vqa_parquet",
                target_text_field="json",
                camera_keys=(),
                grounding_camera_keys=(),
                state_key="",
                action_key="",
                state_dim=0,
                action_dim=0,
                action_horizon=action_horizon,
                stats_path=None,
                stats_key=None,
                normalization="none",
                normalization_stats=None,
                sample_ratio=float(sample_ratio),
                training_eligibility_exists=False,
                training_eligibility_used=False,
                data_version=None,
            )
        else:
            raise ValueError(
                f"Dataset '{dataset_path}' is not configured. "
                "To use this dataset, please add its configuration to `utils/constants.py`. "
                "Refer to the existing dataset entries for the required format."
            )
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if slot_config is not None and slot_config.enabled:
            from utils.slot_supervision import SlotSupervisedDataset
            ds = SlotSupervisedDataset(ds, slot_reader, slot_config, optical_flow_config)
        datasets.append(ds)

    concat = ConcatDataset(datasets)
    if not len(concat):
        raise ValueError("selected datasets contain no eligible supervision samples")
    if downstream_runtime_state:
        from utils.downstream_probe_data import configure_downstream_dataset
        for dataset in datasets:
            configure_downstream_dataset(dataset, downstream_probe_manifest, entries=dataset_entries,
                                         contract_version=downstream_contract_version)
        # Probe configuration can change the number of indexable samples.
        concat = torch.utils.data.ConcatDataset(datasets)
        if downstream_probe_manifest:
            from utils.downstream_probe_data import bind_global_probe_order
            bind_global_probe_order(concat, downstream_probe_manifest)
    concat.resolved_dataset_manifest = build_resolved_dataset_manifest(
        [dataset.spec for dataset in datasets], loss_type
    )
    return concat

def custom_collate_fn(batch):
    """
    Generic collate_fn that stacks all keys except for certain keys,
    for which it uses cat along the first dimension.
    """
    if not batch:
        raise DatasetIntegrityError("custom_collate_fn received an empty batch")
    none_indices = [index for index, item in enumerate(batch) if item is None]
    if none_indices:
        raise DatasetIntegrityError(
            f"custom_collate_fn received None sample at batch index {none_indices[0]}"
        )

    cat_keys = {'pixel_values', 'image_grid_thw'}
    metadata_keys = {'task', 'train_data', 'slot_data', 'stats_key', 'downstream_dataset_entry', 'downstream_camera_keys'}
    from utils.downstream_contracts import IDENTITY_KEYS
    if any(IDENTITY_KEYS.intersection(item) for item in batch) and any(not IDENTITY_KEYS.issubset(item) for item in batch):
        raise DatasetIntegrityError('incomplete downstream sample identity in collator')
    token_stat_keys = set(TOKENIZATION_METRIC_SCHEMA)
    token_validity_keys = {
        token_metric_validity_key(key) for key in TOKENIZATION_METRIC_SCHEMA
    }
    common_keys = set.intersection(*(set(item) for item in batch))
    common_keys -= token_stat_keys | token_validity_keys
    keys = common_keys | token_stat_keys
    if any('labels' in item for item in batch):
        keys.add('labels')
    result = {}
    if any(any(k.startswith("slot_Q") for k in item) for item in batch):
        from utils.slot_labels import empty_slot_labels
        for key, default in empty_slot_labels().items():
            result[key] = torch.stack([item.get(key, default) for item in batch])
            keys.discard(key)
    flow_keys = {key for item in batch for key in item if key.startswith("flow_")}
    if flow_keys:
        flow_metadata_keys = {"flow_dataset_id", "flow_camera", "flow_manifest_sha256",
                              "flow_generation_identity", "flow_label_identity",
                              "flow_label_file_sha256"}
        for key in flow_keys:
            if key in {"flow_target", "flow_valid_mask"}:
                result[key] = {index: item[key] for index, item in enumerate(batch) if item.get(key) is not None}
            elif key in flow_metadata_keys:
                result[key] = [item.get(key, "") for item in batch]
            else:
                default = torch.tensor(False) if key == "flow_supervision_available" else torch.tensor(-1)
                result[key] = torch.stack([item.get(key, default) for item in batch])
        keys -= flow_keys
    for key in keys:
        if key in metadata_keys:
            result[key] = [item[key] for item in batch]
            continue
        if key == 'labels':
            items = [
                item.get('labels', torch.full_like(item['input_ids'], -100))
                for item in batch
            ]
        elif key in token_stat_keys:
            spec = TOKENIZATION_METRIC_SCHEMA[key]
            exemplar = next(
                (item[key] for item in batch if key in item),
                torch.tensor(False if spec.dtype == torch.bool else 0, dtype=spec.dtype),
            )
            items = [item.get(key, torch.zeros_like(exemplar)) for item in batch]
            validity_key = token_metric_validity_key(key)
            validity = [
                item.get(validity_key, torch.tensor(key in item, dtype=torch.bool))
                for item in batch
            ]
            result[validity_key] = torch.stack(validity, dim=0).to(torch.bool)
        else:
            items = [item[key] for item in batch]
        if key in cat_keys:
            result[key] = torch.cat(items, dim=0)
        else:
            result[key] = torch.stack(items, dim=0)

    if 'input_ids' in result and 'attention_mask' in result:
        attention_mask = result['attention_mask']  # [B, L]
        valid_lens = attention_mask.sum(dim=1).long()  # [B]
        max_len = int(valid_lens.max().item()) + 1
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in result:
                result[key] = result[key][:, :max_len].contiguous()
        result['padding_token_count'] = (
            result['attention_mask'].shape[1]
            - result['attention_mask'].to(torch.long).sum(dim=1)
        )
        result[token_metric_validity_key('padding_token_count')] = torch.ones(
            result['attention_mask'].shape[0], dtype=torch.bool
        )

    return result


class EpochGroupedSampler(Sampler[int]):
    """Deterministically shuffle sample units while keeping declared groups intact."""

    def __init__(self, concat_dataset: ConcatDataset, seed: int = 42):
        self.concat_dataset = concat_dataset
        self.seed = int(seed)
        self.epoch = 0
        self._units = []
        self._dataset_units = []
        offset = 0
        for dataset in concat_dataset.datasets:
            dataset_units = []
            range_builder = getattr(dataset, "sampling_group_ranges", None)
            group_builder = getattr(dataset, "sampling_groups", None)
            if callable(range_builder):
                ranges = range_builder()
                previous = 0
                for start, stop in ranges:
                    if start != previous or not start < stop or stop > len(dataset):
                        raise ValueError(
                            "sampling_group_ranges must exactly cover local indices in order"
                        )
                    unit = range(offset + start, offset + stop)
                    unit = unit[0] if len(unit) == 1 else unit
                    self._units.append(unit)
                    dataset_units.append(unit)
                    previous = stop
                if previous != len(dataset):
                    raise ValueError("sampling_group_ranges do not cover the dataset")
                self._dataset_units.append(dataset_units)
                offset += len(dataset)
                continue
            local_groups = group_builder() if callable(group_builder) else None
            if local_groups is None:
                dataset_units = list(range(offset, offset + len(dataset)))
                self._units.extend(dataset_units)
                self._dataset_units.append(dataset_units)
                offset += len(dataset)
                continue
            flattened = [index for group in local_groups for index in group]
            if sorted(flattened) != list(range(len(dataset))):
                raise ValueError(
                    "sampling_groups must cover every local dataset index exactly once"
                )
            for group in local_groups:
                if not group:
                    raise ValueError("sampling_groups must not contain empty groups")
                global_group = tuple(offset + index for index in group)
                unit = global_group[0] if len(global_group) == 1 else global_group
                self._units.append(unit)
                dataset_units.append(unit)
            self._dataset_units.append(dataset_units)
            offset += len(dataset)
        if offset != len(concat_dataset):
            raise ValueError("sampler length does not match concatenated dataset")
        block_sizes = [
            getattr(dataset, "natural_mix_block_size", None)
            for dataset in concat_dataset.datasets
        ]
        self.natural_mix_block_size = (
            int(block_sizes[0])
            if len(block_sizes) > 1
            and block_sizes[0] is not None
            and all(value == block_sizes[0] for value in block_sizes)
            else None
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if hasattr(self.concat_dataset, "downstream_probe_order"):
            yield from self.concat_dataset.downstream_probe_order
            return
        if all(getattr(dataset, "fixed_probe_order", False) for dataset in self.concat_dataset.datasets):
            yield from range(len(self.concat_dataset))
            return
        if self.natural_mix_block_size is not None:
            yield from self._iter_natural_frame_mix()
            return
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        for unit_index in torch.randperm(len(self._units), generator=generator).tolist():
            unit = self._units[unit_index]
            if isinstance(unit, (tuple, range)):
                yield from unit
            else:
                yield unit

    @staticmethod
    def _flatten_units(units, generator):
        for unit_index in torch.randperm(len(units), generator=generator).tolist():
            unit = units[unit_index]
            if isinstance(unit, (tuple, range)):
                yield from unit
            else:
                yield unit

    def _iter_natural_frame_mix(self):
        streams = []
        remaining = []
        for dataset_index, (dataset, units) in enumerate(
            zip(self.concat_dataset.datasets, self._dataset_units)
        ):
            generator = torch.Generator()
            generator.manual_seed(
                self.seed + self.epoch * 1_000_003 + dataset_index * 10_007
            )
            streams.append(iter(self._flatten_units(units, generator)))
            remaining.append(len(dataset))

        schedule_generator = torch.Generator()
        schedule_generator.manual_seed(self.seed + self.epoch * 1_000_003 + 97)
        while (total_remaining := sum(remaining)) > 0:
            block_size = min(self.natural_mix_block_size, total_remaining)
            desired = [block_size * count / total_remaining for count in remaining]
            allocation = [min(count, int(value)) for count, value in zip(remaining, desired)]
            unassigned = block_size - sum(allocation)
            priorities = sorted(
                range(len(remaining)),
                key=lambda index: (desired[index] - allocation[index], remaining[index], -index),
                reverse=True,
            )
            for dataset_index in priorities:
                if not unassigned:
                    break
                if allocation[dataset_index] < remaining[dataset_index]:
                    allocation[dataset_index] += 1
                    unassigned -= 1
            if unassigned:
                raise RuntimeError("natural frame mixer could not allocate a complete block")
            schedule = torch.repeat_interleave(
                torch.arange(len(allocation), dtype=torch.long),
                torch.tensor(allocation, dtype=torch.long),
            )
            schedule = schedule[
                torch.randperm(len(schedule), generator=schedule_generator)
            ]
            for dataset_index in schedule.tolist():
                yield next(streams[dataset_index])
                remaining[dataset_index] -= 1

    def __len__(self) -> int:
        return len(self.concat_dataset)


class EpochGroupedDistributedBatchSampler(Sampler[list[int]]):
    """Shard one padded epoch across ranks before forming local micro-batches."""

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        batch_size_per_device: int,
        num_processes: int,
        seed: int = 42,
        gradient_accumulation_steps: int = 1,
        pad_epoch_to_optimizer_batch: bool = False,
    ):
        if batch_size_per_device < 1:
            raise ValueError("batch_size_per_device must be positive")
        if num_processes < 1:
            raise ValueError("num_processes must be positive")
        self.sampler = EpochGroupedSampler(concat_dataset, seed=seed)
        self.batch_size = int(batch_size_per_device)
        self.num_processes = int(num_processes)
        self.drop_last = False
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.pad_epoch_to_optimizer_batch = bool(pad_epoch_to_optimizer_batch)
        if self.pad_epoch_to_optimizer_batch and len(self.sampler) < 1:
            raise ValueError("cannot pad an empty dataset")

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    @property
    def samples_per_process(self) -> int:
        if self.pad_epoch_to_optimizer_batch:
            unit = self.num_processes * self.batch_size * self.gradient_accumulation_steps
            return math.ceil(len(self.sampler) / unit) * unit // self.num_processes
        return math.ceil(len(self.sampler) / self.num_processes)

    @property
    def batches_per_process(self) -> int:
        return math.ceil(self.samples_per_process / self.batch_size)

    def __iter__(self):
        padding_size = self.samples_per_process * self.num_processes - len(self.sampler)
        first_indices = []
        buffers = [[] for _ in range(self.num_processes)]
        position = 0

        def consume(index):
            nonlocal position
            buffers[position % self.num_processes].append(index)
            position += 1
            if all(len(buffer) >= self.batch_size for buffer in buffers):
                result = [buffer[: self.batch_size] for buffer in buffers]
                for process_index in range(self.num_processes):
                    del buffers[process_index][: self.batch_size]
                return result
            return None

        for index in self.sampler:
            if len(first_indices) < padding_size:
                first_indices.append(index)
            ready = consume(index)
            if ready is not None:
                yield from ready
        if self.pad_epoch_to_optimizer_batch:
            import itertools
            padding_indices = itertools.islice(itertools.cycle(first_indices), padding_size)
        else:
            padding_indices = first_indices
        for index in padding_indices:
            ready = consume(index)
            if ready is not None:
                yield from ready
        if any(buffers):
            if not all(len(buffer) == len(buffers[0]) for buffer in buffers):
                raise RuntimeError("streaming distributed sampler produced uneven shards")
            yield from buffers

    def __len__(self) -> int:
        return self.num_processes * self.batches_per_process


def resolve_dataloader_num_workers(concat_dataset, requested: int | None) -> int:
    if requested is not None:
        if requested < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        return requested
    has_grouped_dataset = any(
        callable(getattr(dataset, "sampling_groups", None))
        or callable(getattr(dataset, "sampling_group_ranges", None))
        for dataset in concat_dataset.datasets
    )
    return 4 if has_grouped_dataset else 24

def create_dataloader_for_concat(
    concat_dataset,
    batch_size_per_device: int,
    num_processes: int = 1,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    seed: int = 42,
    gradient_accumulation_steps: int = 1,
    pad_epoch_to_optimizer_batch: bool = False,
):
    batch_sampler = EpochGroupedDistributedBatchSampler(
        concat_dataset,
        batch_size_per_device=batch_size_per_device,
        num_processes=num_processes,
        seed=seed,
        gradient_accumulation_steps=gradient_accumulation_steps,
        pad_epoch_to_optimizer_batch=pad_epoch_to_optimizer_batch,
    )
    kwargs = {
        "dataset": concat_dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": False,
        "collate_fn": custom_collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(**kwargs)

    return dataloader


def set_dataloader_epoch(dataloader, epoch_sampler, epoch):
    epoch_sampler.set_epoch(epoch)
    # Accelerate resets the sampler from its own iteration at __iter__ entry.
    # A fresh resumed loader must therefore receive the saved epoch as well.
    set_epoch = getattr(dataloader, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)


def validate_fast_resume_datasets(concat_dataset, *, downstream=False):
    from utils.dataset import Stage05MixedPretrainingDataset
    from utils.slot_supervision import SlotSupervisedDataset
    for dataset in concat_dataset.datasets:
        if downstream:
            if (not isinstance(dataset, StreamingLeRobotSampleDataset)
                    or dataset.window_size != 1 or dataset.ecot_supported
                    or dataset.requirements.requires_target or dataset.sample_ratio != 1.0):
                raise ValueError("downstream index resume requires deterministic action-only LeRobot inputs")
            continue
        if type(dataset) is SlotSupervisedDataset:
            dataset = dataset.dataset
        if type(dataset) is not Stage05MixedPretrainingDataset or not dataset.frozen_index:
            raise ValueError("fast data resume requires deterministic frozen Stage05 datasets")


def resume_dataloader_at_batch(dataloader, epoch_sampler, *, epoch, batch_idx):
    """Skip indices before dataset reads; retain the caller's absolute batch cursor."""
    if type(batch_idx) is not int or not 0 <= batch_idx <= len(dataloader):
        raise ValueError("resume batch cursor is outside the saved epoch")
    if batch_idx == len(dataloader):
        # An empty DataLoaderShard may yield None; an epoch-boundary resume is empty.
        return ()
    if batch_idx:
        from accelerate import skip_first_batches
        original_sampler = dataloader.batch_sampler
        dataloader = skip_first_batches(dataloader, num_batches=batch_idx)
        # Accelerate's skip wrapper omits the attributes used to compute tail remainder.
        for name in ("batch_size", "num_processes", "split_batches"):
            if hasattr(original_sampler, name):
                setattr(dataloader.batch_sampler, name, getattr(original_sampler, name))
    set_dataloader_epoch(dataloader, epoch_sampler, epoch)
    return dataloader

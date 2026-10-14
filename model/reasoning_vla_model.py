from transformers.feature_extraction_utils import BatchFeature
from dataclasses import replace
from .qwen_vl_backbone import QwenVLBackbone, validate_action_conditioning_norm
from torch import nn
from typing import Tuple
from .flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig
from .difference_query import (
    resolve_difference_query_config,
    save_difference_query_artifacts,
)

import torch
import json
import os
from safetensors.torch import save_file, load_file
import math
from pathlib import Path
from utils.dataset_manifest import write_resolved_dataset_manifest
from utils.checkpoint_metadata import checkpoint_metadata
from utils.optical_flow_config import OpticalFlowConfig, resolve_stage, stage_description, stage_loss_metadata
from utils.optical_flow_checkpoint import (
    resolve_flow_checkpoint, load_flow_weights, save_flow_artifacts, module_checksum,
    initial_stage_training_state, read_stage_training_state, checkpoint_stage_metadata,
)
from .optical_flow_aux_head import build_optical_flow_head
from utils.checkpoint_contract import (
    CHECKPOINT_LOAD_PURPOSES,
    DOWNSTREAM_FINETUNE,
    INFERENCE,
    STAGE05_AR_RESUME,
    STAGE05_AR_TO_JOINT,
    STAGE05_JOINT_RESUME,
    STAGE05_AR_JOINT_CONTRACT_KEY,
    build_stage05_ar_joint_contract,
    _checkpoint_has_stage05_identity,
    is_stage05_four_dataset_manifest,
    validate_checkpoint_load_purpose_arguments,
    validate_generic_action_expert_contract,
    validate_stage05_checkpoint_for_purpose,
    validate_stage05_resume_artifacts,
)


_LOSS_TYPES = {"vlm", "action", "vlm_and_action"}
CHECKPOINT_METADATA_NAME = "zr0_checkpoint_metadata.json"
CHECKPOINT_METADATA_VERSION = 1


def _checkpoint_kind_for_loss_type(loss_type: str) -> str:
    return {
        "vlm": "ar_only",
        "action": "action_only",
        "vlm_and_action": "joint",
        "aux": "aux_only",
    }[loss_type]


def _read_checkpoint_kind(directory) -> str | None:
    if not directory:
        return None
    path = Path(directory) / CHECKPOINT_METADATA_NAME
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"failed to read checkpoint metadata {path}: {error}") from error
    kind = metadata.get("checkpoint_kind")
    if metadata.get("version") != CHECKPOINT_METADATA_VERSION or kind not in {
        "ar_only",
        "joint",
        "action_only",
        "aux_only",
    }:
        raise ValueError(f"invalid checkpoint metadata in {path}")
    return kind

class ZR0Model(nn.Module):
    def __init__(
            self,
            vlm_name_or_path: str,
            action_expert_name_or_path: str,
            action_expert_config: FlowmatchingActionHeadConfig,
            tune_vlm=None,
            tune_action_expert=True,
            detach_vlm_outputs_for_action_expert=False,
            lora_args=None,
            use_difference_query=None,
            num_difference_queries=None,
            vlm_attention_backend=None,
            loss_type=None,
            checkpoint_load_purpose=None,
            resume_training=False,
            action_expert_config_path=None,
            training_stage=None,
            slot_aux_type="none",
            slot_config=None,
            slot_supervision_stats=None,
            optical_flow_config=None,
            init_from_checkpoint=None,
            resume_from_checkpoint=None,
            action_expert_init_seed=42,
            action_conditioning_norm="final_rmsnorm",
            allow_aux_loss_weight_override=False,
        ):
        super().__init__()
        if init_from_checkpoint and (resume_from_checkpoint or resume_training):
            raise ValueError("init_from_checkpoint and resume are mutually exclusive")
        source = init_from_checkpoint or resume_from_checkpoint
        if source:
            if vlm_name_or_path and Path(vlm_name_or_path).resolve() != Path(source).resolve():
                raise ValueError("VLM source conflicts with explicit checkpoint source")
            vlm_name_or_path = source
        resume_training = resume_training or bool(resume_from_checkpoint)
        self.training_stage = training_stage
        if tune_vlm is None:
            tune_vlm = training_stage != "stage2_aux"
        from utils.slot_config import SlotConfig, resolve_query_layout
        from utils.slot_checkpoint import resolve_slot_checkpoint, validate_slot_stats
        requested_slot = slot_config
        if requested_slot is None and slot_aux_type != "none":
            raise ValueError("enabled slot_aux_type requires SlotConfig with explicit positive loss weight")
        self.slot_config, slot_payload = resolve_slot_checkpoint(
            vlm_name_or_path if training_stage is not None else None, requested_slot, resume=resume_training,
            stage=training_stage, initialize=bool(init_from_checkpoint),
            allow_loss_weight_override=allow_aux_loss_weight_override)
        slot_aux_type = self.slot_config.slot_aux_type
        self.optical_flow_config, flow_payload = resolve_flow_checkpoint(
            vlm_name_or_path if training_stage is not None else None,
            optical_flow_config, stage=training_stage, resume=resume_training, initialize=bool(init_from_checkpoint),
            allow_loss_weight_override=allow_aux_loss_weight_override)
        loss_type = resolve_stage(training_stage, loss_type, flow=self.optical_flow_config, slot=self.slot_config)
        if slot_supervision_stats is not None and slot_payload is not None and checkpoint_metadata(slot_supervision_stats) != checkpoint_metadata(slot_payload["statistics"]):
            raise ValueError("requested Slot statistics conflict with checkpoint")
        self.slot_supervision_stats = slot_payload["statistics"] if slot_payload else slot_supervision_stats
        if self.slot_config.enabled:
            if self.slot_supervision_stats is None:
                raise ValueError("Slot requires audited fixed train statistics")
            validate_slot_stats(self.slot_supervision_stats, self.slot_config)
        if self.slot_config.enabled and not resume_training and not init_from_checkpoint:
            raise ValueError("Slot stages require explicit init_from_checkpoint or same-stage resume")
        self.stage_description = stage_description(training_stage, self.optical_flow_config, slot_aux_type)
        self.source_stage_training_state = (read_stage_training_state(flow_payload, OpticalFlowConfig(**flow_payload["config"]))
                                            if flow_payload else None)
        self.stage_training_state = None
        if training_stage is not None:
            self.stage_training_state = (dict(self.source_stage_training_state)
                if resume_training and self.source_stage_training_state is not None
                else initial_stage_training_state(training_stage, self.optical_flow_config, slot_aux_type))
        if resume_training and training_stage in {"stage3_joint", "downstream_flow"}:
            if action_expert_name_or_path and Path(action_expert_name_or_path).resolve() != Path(vlm_name_or_path).resolve():
                raise ValueError("stage3 resume requires Expert weights from the same checkpoint")
            action_expert_name_or_path = vlm_name_or_path
        if training_stage in {"stage1_ar", "stage2_aux"}:
            if action_expert_name_or_path:
                raise ValueError("stage1/stage2 cannot load Action Expert weights")
            tune_action_expert = False
        if training_stage == "stage3_joint" and not tune_action_expert:
            raise ValueError("stage3_joint requires a trainable Action Expert")
        if training_stage == "downstream_flow":
            if not tune_vlm or not tune_action_expert or detach_vlm_outputs_for_action_expert:
                raise ValueError("downstream_flow requires trainable, connected VLM/Query/Expert")
            if checkpoint_load_purpose != DOWNSTREAM_FINETUNE or not (source or resume_training):
                raise ValueError("downstream_flow requires explicit downstream initialization/resume")
            if not action_expert_name_or_path or Path(action_expert_name_or_path).resolve() != Path(vlm_name_or_path).resolve():
                raise ValueError("downstream_flow requires all weights from the same checkpoint")
        if training_stage and training_stage != "downstream_flow" and action_expert_name_or_path and not resume_training and Path(action_expert_name_or_path).resolve() == Path(vlm_name_or_path).resolve():
            raise ValueError("initialization requires an independent Action Expert checkpoint")
        if loss_type not in _LOSS_TYPES and loss_type != "aux":
            raise ValueError(f"loss_type must be one of {sorted(_LOSS_TYPES)}")
        validate_checkpoint_load_purpose_arguments(
            checkpoint_load_purpose, resume_training=resume_training
        )
        if checkpoint_load_purpose == STAGE05_AR_RESUME and loss_type != "vlm":
            raise ValueError("stage05_ar_resume requires loss_type=vlm")
        if loss_type == "vlm":
            if not tune_vlm:
                raise ValueError("loss_type=vlm requires tune_vlm=True")
            if tune_action_expert:
                raise ValueError("loss_type=vlm requires tune_action_expert=False")
            if action_expert_name_or_path:
                raise ValueError(
                    "loss_type=vlm does not accept action_expert_name_or_path"
                )
        self.loss_type = loss_type
        self.checkpoint_load_purpose = checkpoint_load_purpose
        self.resolved_dataset_manifest = None
        self.action_expert_source_action_horizon = int(
            action_expert_config.action_horizon
        )
        source_kind = _read_checkpoint_kind(vlm_name_or_path)

        for checkpoint_source in dict.fromkeys(
            source
            for source in (vlm_name_or_path, action_expert_name_or_path)
            if source
        ):
            checkpoint_path = Path(checkpoint_source)
            if not checkpoint_path.is_dir():
                continue
            metadata_path = checkpoint_path / CHECKPOINT_METADATA_NAME
            metadata_payload = {}
            if metadata_path.is_file():
                try:
                    metadata_payload = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                except Exception as error:
                    raise ValueError(
                        f"failed to read checkpoint metadata {metadata_path}: {error}"
                    ) from error
            if training_stage is not None:
                from utils.checkpoint_contract import validate_action_expert_config_provenance
                provenance = validate_action_expert_config_provenance(checkpoint_path, metadata_payload)
                if provenance is not None:
                    self.action_expert_config_original_bytes = (checkpoint_path / provenance["source_file"]).read_bytes()
            if _checkpoint_has_stage05_identity(
                checkpoint_path.resolve(), metadata_payload
            ) and checkpoint_load_purpose is None and training_stage is None:
                raise ValueError(
                    "Stage05 checkpoint requires an explicit checkpoint_load_purpose"
                )
            if training_stage is not None and _checkpoint_has_stage05_identity(checkpoint_path.resolve(), metadata_payload):
                source_purpose = (STAGE05_AR_TO_JOINT if metadata_payload.get("checkpoint_kind") == "ar_only" else DOWNSTREAM_FINETUNE)
                source_config = validate_stage05_checkpoint_for_purpose(
                    checkpoint_path, purpose=source_purpose,
                    external_config_path=checkpoint_path / "action_expert_config.json",
                    requested_action_horizon=action_expert_config.action_horizon,
                    expected_action_dim=action_expert_config.action_dim,
                    expected_state_dim=action_expert_config.state_dim,
                    expected_num_difference_queries=num_difference_queries or 32)
                if source_config.payload != action_expert_config.to_dict():
                    raise ValueError("inherited checkpoint Action Expert runtime config mismatch")
            if "action_expert_contract" in metadata_payload and (training_stage is None or checkpoint_source == action_expert_name_or_path):
                from utils.action_expert_config import read_vlm_hidden_size

                allow_horizon_override = (
                    not resume_training
                    and bool(action_expert_name_or_path)
                    and loss_type in {"action", "vlm_and_action"}
                    and checkpoint_load_purpose in (None, DOWNSTREAM_FINETUNE)
                )
                verified = validate_generic_action_expert_contract(
                    checkpoint_path,
                    expected_action_dim=action_expert_config.action_dim,
                    expected_state_dim=action_expert_config.state_dim,
                    expected_vlm_hidden_size=read_vlm_hidden_size(vlm_name_or_path),
                    **(
                        {"action_horizon_override": action_expert_config.action_horizon}
                        if allow_horizon_override else
                        {"expected_action_horizon": action_expert_config.action_horizon}
                    ),
                )
                if action_expert_config.to_dict() != verified.payload:
                    raise ValueError("requested Action Expert config differs from the verified contract")
                action_expert_config = verified.config
                self.action_expert_source_action_horizon = verified.source_action_horizon

        # Model construction is also a protected production boundary.  The
        # launcher and train_vla perform the same check, but callers that use
        # ZR0Model directly must not be able to allocate a model around an
        # unverified checkpoint contract.
        if checkpoint_load_purpose is not None:
            purpose_source = (
                action_expert_name_or_path
                if checkpoint_load_purpose in {DOWNSTREAM_FINETUNE, INFERENCE}
                and action_expert_name_or_path
                else vlm_name_or_path
            )
            validator = (
                validate_stage05_resume_artifacts
                if checkpoint_load_purpose == STAGE05_JOINT_RESUME
                else validate_stage05_checkpoint_for_purpose
            )
            validator_kwargs = {
                "external_config_path": action_expert_config_path,
                "requested_action_horizon": action_expert_config.action_horizon,
                "expected_action_dim": int(action_expert_config.action_dim),
                "expected_state_dim": int(action_expert_config.state_dim),
                "expected_num_difference_queries": (
                    num_difference_queries
                    if num_difference_queries is not None
                    else 32
                ),
            }
            if validator is validate_stage05_checkpoint_for_purpose:
                validator_kwargs.update(
                    purpose=checkpoint_load_purpose,
                    resume_training=resume_training,
                )
            validated_source = validator(purpose_source, **validator_kwargs)
            self.action_expert_source_action_horizon = int(
                validated_source.source_action_horizon
            )
            if checkpoint_load_purpose == STAGE05_AR_RESUME:
                if action_expert_config.to_dict() != validated_source.payload:
                    raise ValueError("Stage05 AR resume requested Expert config mismatch")
                action_expert_config = validated_source.config
        if training_stage is None and loss_type == "vlm" and source_kind not in (None, "ar_only"):
            raise ValueError(
                f"checkpoint kind {source_kind!r} cannot be loaded for loss_type=vlm"
            )
        if loss_type == "vlm_and_action" and source_kind == "action_only":
            raise ValueError(
                "checkpoint kind 'action_only' cannot provide the VLM for joint training"
            )
        if training_stage is None and loss_type == "vlm_and_action" and source_kind == "ar_only" and action_expert_name_or_path:
            raise ValueError(
                "joint warm start from an ar_only checkpoint must randomly initialize "
                "the Action Expert and forbids action_expert_name_or_path"
            )
        if (
            source_kind == "ar_only"
            and training_stage is None
            and loss_type == "vlm_and_action"
            and checkpoint_load_purpose not in (None, STAGE05_AR_TO_JOINT)
        ):
            raise ValueError(
                "an ar_only checkpoint requires checkpoint_load_purpose="
                "'stage05_ar_to_joint' for Joint initialization"
            )
        if training_stage is None and loss_type == "vlm_and_action" and source_kind == "joint":
            same_checkpoint = action_expert_name_or_path and (
                Path(action_expert_name_or_path).resolve()
                == Path(vlm_name_or_path).resolve()
            )
            if not same_checkpoint:
                raise ValueError(
                    "joint checkpoint resume requires the same checkpoint as the "
                    "Action Expert weight source"
                )
        action_kind = _read_checkpoint_kind(action_expert_name_or_path)
        if action_kind == "ar_only":
            raise ValueError("checkpoint kind 'ar_only' cannot provide Action Expert weights")

        self.action_expert_config = action_expert_config
        action_horizon = action_expert_config.action_horizon
        max_seq_len = action_expert_config.max_seq_len
        invalid_horizon = (
            isinstance(action_horizon, bool)
            or not isinstance(action_horizon, int)
            or action_horizon <= 0
        )
        invalid_max_seq_len = (
            isinstance(max_seq_len, bool)
            or not isinstance(max_seq_len, int)
            or max_seq_len <= 0
        )
        dit_config = getattr(action_expert_config, "diffusion_transformer_cfg", {})
        max_position_embeddings = (
            dit_config.get("max_num_positional_embeddings")
            if isinstance(dit_config, dict)
            else None
        )
        invalid_position_capacity = (
            isinstance(max_position_embeddings, bool)
            or not isinstance(max_position_embeddings, int)
            or max_position_embeddings <= 0
            or action_horizon + 1 > max_position_embeddings
        )
        if (
            invalid_horizon
            or invalid_max_seq_len
            or action_horizon + 1 > max_seq_len
            or invalid_position_capacity
        ):
            raise ValueError(
                "Action Expert action_horizon must be a positive integer whose "
                "state-plus-action sequence fits max_seq_len and positional capacity; "
                f"got action_horizon={action_horizon}, max_seq_len={max_seq_len}, "
                f"max_num_positional_embeddings={max_position_embeddings}"
            )

        difference_query_config = resolve_difference_query_config(
            vlm_name_or_path,
            action_expert_name_or_path if training_stage is None else None,
            use_difference_query=use_difference_query,
            num_difference_queries=num_difference_queries,
            vlm_attention_backend=vlm_attention_backend,
        )
        self.use_difference_query = difference_query_config.enabled
        self.action_conditioning_norm = validate_action_conditioning_norm(
            action_conditioning_norm, use_difference_query=self.use_difference_query
        )
        self.num_difference_queries = (
            difference_query_config.num_difference_queries
        )
        self.vlm_attention_backend = difference_query_config.attention_backend
        self.query_role_layout = resolve_query_layout(self.num_difference_queries, self.optical_flow_config.num_flow_queries,
                                                      slot_enabled=self.slot_config.enabled, query_enabled=self.use_difference_query)
        if slot_payload and self.query_role_layout != slot_payload["query_layout"]:
            raise ValueError("Slot checkpoint query role layout conflict")
        if self.optical_flow_config.enabled and (
            not self.use_difference_query or self.optical_flow_config.num_flow_queries > self.num_difference_queries
        ):
            raise ValueError("OF requires enabled Difference Query and num_flow_queries <= total queries")

        self.backbone = QwenVLBackbone(
            vlm_name_or_path,
            tune_vlm,
            lora_args,
            resolved_difference_query_config=difference_query_config,
            action_conditioning_norm=self.action_conditioning_norm,
        )
        print("the size of VLM's last layer hidden state:", self.backbone.model.config.text_config.hidden_size)
        actual_vlm_hidden_size = self.backbone.model.config.text_config.hidden_size
        if self.action_expert_config.vlm_output_embedding_dim != actual_vlm_hidden_size:
            raise ValueError(
                "Action Expert config hidden size does not match the VLM: "
                f"{self.action_expert_config.vlm_output_embedding_dim} != "
                f"{actual_vlm_hidden_size}"
            )
        # self.action_expert_config.vlm_output_embedding_dim = self.backbone.model.config.hidden_size
        self.action_expert = None
        if loss_type not in {"vlm", "aux"}:
            if training_stage is None:
                self.action_expert = FlowmatchingActionHead(self.action_expert_config, tune_action_expert)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(action_expert_init_seed)
                    self.action_expert = FlowmatchingActionHead(self.action_expert_config, tune_action_expert)
        self.optical_flow_aux = None
        self.slot_aux = None
        if training_stage is not None:
            before = {"vlm": module_checksum(self.backbone.model),
                      "query": module_checksum(self.backbone.difference_query) if self.use_difference_query else None}
            self.optical_flow_aux = build_optical_flow_head(actual_vlm_hidden_size, self.optical_flow_config)
            if self.optical_flow_config.enabled:
                load_flow_weights(self.optical_flow_aux, vlm_name_or_path, flow_payload)
            if self.slot_config.enabled:
                from model.structured_slot_head import build_slot_head
                from utils.slot_checkpoint import load_slot_weights
                self.slot_aux = build_slot_head(actual_vlm_hidden_size, self.query_role_layout["num_slot_queries"], self.slot_config)
                load_slot_weights(self.slot_aux, vlm_name_or_path, slot_payload)
                print("Slot Head parameter count:", sum(p.numel() for p in self.slot_aux.parameters()))
            after = {"vlm": module_checksum(self.backbone.model),
                     "query": module_checksum(self.backbone.difference_query) if self.use_difference_query else None}
            if before != after:
                raise RuntimeError("auxiliary initialization changed VLM/Query")
            self.aux_initialization = {"source": str(vlm_name_or_path), "before": before, "after": after,
                                       "flow_source": "checkpoint" if flow_payload and
                                           flow_payload["config"]["optical_flow_aux_type"] != "none" and
                                           not flow_payload.get("replace_flow_head") else "random",
                                       "action_expert_source": action_expert_name_or_path or "random_or_absent",
                                       "action_expert_init_seed": action_expert_init_seed}
            print(self.stage_description)
            if self.optical_flow_aux is not None:
                print("Optical Flow parameter counts:", self.optical_flow_aux.parameter_counts())
        self.detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert
        
        if action_expert_name_or_path:
            print(f"load pre-trained weights from {action_expert_name_or_path} to initialize the action expert")
            loaded_state = load_file(
                os.path.join(action_expert_name_or_path, "action_expert.safetensors")
            )
            expected_state = self.action_expert.state_dict()
            if set(loaded_state) != set(expected_state):
                raise ValueError(
                    "Action Expert checkpoint state_dict keys do not match the "
                    "validated runtime architecture"
                )
            shape_mismatches = [
                name
                for name in expected_state
                if tuple(loaded_state[name].shape) != tuple(expected_state[name].shape)
            ]
            if shape_mismatches:
                raise ValueError(
                    "Action Expert checkpoint has incompatible parameter shapes: "
                    + ", ".join(shape_mismatches[:5])
                )
            self.action_expert.load_state_dict(loaded_state, strict=True)

    def _require_action_expert(self) -> FlowmatchingActionHead:
        if self.action_expert is None:
            raise ValueError("this ar_only model has no Action Expert")
        return self.action_expert

    def _validate_action_conditioning(
        self, backbone_outputs: BatchFeature, expected_batch_size: int
    ) -> None:
        if not self.use_difference_query:
            return

        embeddings = backbone_outputs.get("backbone_embeddings")
        attention_mask = backbone_outputs.get("action_expert_cross_attn_mask")
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
            raise ValueError("Difference Query action conditioning must have shape [B, Nq, H]")
        if embeddings.shape[0] != expected_batch_size:
            raise ValueError(
                "Difference Query action conditioning batch mismatch: "
                f"{embeddings.shape[0]} != {expected_batch_size}"
            )
        if embeddings.shape[1] != self.num_difference_queries:
            raise ValueError(
                "Difference Query action conditioning Nq mismatch: "
                f"{embeddings.shape[1]} != {self.num_difference_queries}"
            )
        expected_hidden_size = self.action_expert_config.vlm_output_embedding_dim
        if embeddings.shape[2] != expected_hidden_size:
            raise ValueError(
                "Difference Query action conditioning hidden size mismatch: "
                f"{embeddings.shape[2]} != {expected_hidden_size}"
            )
        expected_mask_shape = embeddings.shape[:2]
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != expected_mask_shape:
            actual_shape = getattr(attention_mask, "shape", None)
            raise ValueError(
                "Difference Query action conditioning mask shape mismatch: "
                f"{actual_shape} != {expected_mask_shape}"
            )
        if attention_mask.dtype != torch.bool:
            raise ValueError("Difference Query action conditioning mask must have bool dtype")
        if attention_mask.device != embeddings.device:
            raise ValueError(
                "Difference Query action conditioning mask and hidden states must share a device"
            )
        if not attention_mask.all():
            raise ValueError("Difference Query action conditioning mask must be all True")
        if not torch.isfinite(embeddings).all():
            raise ValueError("Difference Query action conditioning hidden states must be finite")

    def prepare_vla_input(self, batch_inputs) -> Tuple[BatchFeature, BatchFeature]:
        backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
        action_expert_inputs = self._require_action_expert().prepare_inputs(batch_inputs)
        # print(f"prepare a batched samples spends {time.time()-start_time}s")
        return backbone_inputs, action_expert_inputs

    @torch.inference_mode()
    def get_action_direct(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5) -> BatchFeature:
        action_expert = self._require_action_expert()
        with torch.no_grad():
            backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
            backbone_outputs = self.backbone(
                backbone_inputs, compute_vlm_loss=False
            )
            self._validate_action_conditioning(
                backbone_outputs, expected_batch_size=batch_inputs["input_ids"].shape[0]
            )

            action_expert_inputs = action_expert.prepare_inputs(batch_inputs)
            action_expert_outputs = action_expert.get_action(
                backbone_outputs, action_expert_inputs, num_denoised_steps
            )
        # return action_pred
        return BatchFeature(data={"action_pred": action_expert_outputs["action_pred"]})
    
    @torch.inference_mode()
    def get_n_actions_direct(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5, n: int = 8) -> BatchFeature:
        # sampling `n` different action chunks
        action_expert = self._require_action_expert()
        with torch.no_grad():
            backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
            backbone_outputs = self.backbone(
                backbone_inputs, compute_vlm_loss=False
            )
            self._validate_action_conditioning(
                backbone_outputs, expected_batch_size=batch_inputs["input_ids"].shape[0]
            )

            action_expert_inputs = action_expert.prepare_inputs(batch_inputs)
            action_preds_list = []
            for i in range(n):
                action_expert_outputs = action_expert.get_action(
                    backbone_outputs, action_expert_inputs, num_denoised_steps
                )
                action_preds_list.append(action_expert_outputs["action_pred"][0])
        n_action_preds = torch.stack(action_preds_list, dim=0)

        # return action_pred
        return BatchFeature({"n_action_preds": n_action_preds})
  
    @torch.inference_mode()
    def get_action_subtask(self, batch_inputs: BatchFeature, num_denoised_steps: int = 5) -> BatchFeature:
        action_expert = self._require_action_expert()
        if self.use_difference_query:
            raise NotImplementedError(
                "subtask/autoregressive generation is not implemented with Difference Query"
            )
        with torch.no_grad():
            prompt_len = len(batch_inputs["input_ids"][0])
            
            # 1. generate subtask
            new_input_ids = self.backbone.generate(
                input_ids = batch_inputs["input_ids"],
                attention_mask = batch_inputs["attention_mask"],
                pixel_values = batch_inputs["pixel_values"],
                image_grid_thw = batch_inputs["image_grid_thw"],
                do_sample=False,
                eos_token_id=[153718]
            )
            new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.long)

            print("output subtask:", self.backbone.processor.tokenizer.decode(new_input_ids[0][prompt_len:]))

            # 2. perpare new VLM inputs (prompt + subtask)
            backbone_inputs = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "pixel_values": batch_inputs["pixel_values"],
                "image_grid_thw": batch_inputs["image_grid_thw"],
                "sub_task_flag": batch_inputs["sub_task_flag"]
            }
            # 3. VLM forward
            backbone_outputs = self.backbone(
                backbone_inputs, compute_vlm_loss=False
            )
            
            # 4. action expert diffusion
            action_expert_inputs = action_expert.prepare_inputs(batch_inputs)
            action_expert_outputs = action_expert.get_action(
                backbone_outputs, action_expert_inputs, num_denoised_steps
            )
        # return action_pred
        return BatchFeature(data={"action_pred": action_expert_outputs["action_pred"]})

    def _resolve_training_loss_type(self, dynamic_loss_type, loss_type) -> str:
        if dynamic_loss_type is not None and loss_type is not None and dynamic_loss_type != loss_type:
            raise ValueError(
                "dynamic_loss_type and loss_type must match when both are provided"
            )
        requested_loss_type = loss_type if loss_type is not None else dynamic_loss_type
        if requested_loss_type is None:
            return self.loss_type
        if requested_loss_type not in _LOSS_TYPES | {"aux"}:
            raise ValueError(
                f"Unrecognized loss_type: {requested_loss_type}. "
                "Expected one of [vlm, action, vlm_and_action, aux (stage2_aux only)]"
            )
        if requested_loss_type != self.loss_type:
            raise ValueError(
                f"forward loss mode mismatch: constructed mode={self.loss_type!r}, "
                f"requested mode={requested_loss_type!r}"
            )
        return self.loss_type

    @staticmethod
    def _validate_loss_weight(value, name: str, *, required: bool) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite non-negative number") from error
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        if required and numeric_value == 0:
            raise ValueError(f"{name} must be greater than zero for this loss_type")
        return numeric_value

    @staticmethod
    def _validate_training_inputs(batch_inputs, loss_type: str) -> None:
        input_ids = batch_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 1:
            raise ValueError("input_ids must be a non-empty Tensor with a batch dimension")
        batch_size = input_ids.shape[0]
        if loss_type in ("vlm", "vlm_and_action"):
            labels = batch_inputs.get("labels")
            if not isinstance(labels, torch.Tensor):
                raise ValueError("labels are required for vlm and vlm_and_action training")
            if labels.shape != input_ids.shape:
                raise ValueError("labels shape must match input_ids")
            if loss_type == "vlm" and not (labels != -100).any(dim=1).all():
                raise ValueError(
                    "AR-only labels must contain supervision in each sample, not only -100"
                )
        if loss_type in ("action", "vlm_and_action"):
            required_keys = (
                "observation.state",
                "state_mask",
                "action",
                "action_mask",
            )
            for key in required_keys:
                if key not in batch_inputs:
                    raise ValueError(f"{key} is required for action training")
            tensors = {}
            for key in required_keys:
                value = batch_inputs[key]
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"{key} must be a non-empty Tensor")
                if value.numel() == 0 or value.ndim < 1:
                    raise ValueError(f"{key} must be a non-empty Tensor")
                if value.shape[0] != batch_size:
                    raise ValueError(f"{key} batch dimension must match input_ids")
                tensors[key] = value
            if tensors["state_mask"].shape != tensors["observation.state"].shape:
                raise ValueError("state_mask shape must match observation.state")
            if tensors["action_mask"].shape != tensors["action"].shape:
                raise ValueError("action_mask shape must match action")
            action_supervision = batch_inputs.get("action_supervision_available")
            if action_supervision is None:
                action_supervision = torch.ones(
                    batch_size, dtype=torch.bool, device=input_ids.device
                )
            elif (
                not isinstance(action_supervision, torch.Tensor)
                or action_supervision.shape != (batch_size,)
            ):
                raise ValueError(
                    "action_supervision_available must have shape [batch_size]"
                )
            action_supervision = action_supervision.to(
                device=input_ids.device, dtype=torch.bool
            )
            strict_joint_fm = batch_inputs.get("strict_joint_fm")
            if strict_joint_fm is not None:
                if not isinstance(strict_joint_fm, torch.Tensor) or strict_joint_fm.shape != (batch_size,):
                    raise ValueError("strict_joint_fm must have shape [batch_size]")
                strict_joint_fm = strict_joint_fm.to(device=input_ids.device, dtype=torch.bool)
            else:
                strict_joint_fm = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
            if loss_type == "action" and not action_supervision.all():
                raise ValueError(
                    "action-only training does not accept samples without action supervision"
                )
            per_sample_state_mask = tensors["state_mask"].to(dtype=torch.bool).reshape(batch_size, -1)
            per_sample_action_mask = tensors["action_mask"].to(dtype=torch.bool).reshape(batch_size, -1)
            state_valid = per_sample_state_mask.any(dim=1)
            action_valid = per_sample_action_mask.any(dim=1)
            if (action_supervision & ~state_valid).any():
                raise ValueError(
                    "VLA state_mask must contain a valid state for each sample with action supervision"
                )
            if (strict_joint_fm & (~action_supervision | ~action_valid)).any():
                raise ValueError(
                    "strict Joint samples require FM_count > 0 for every sample"
                )
            if (action_supervision & ~action_valid).any():
                raise ValueError(
                    "VLA action_mask must contain a valid action for each sample with action supervision"
                )
            if (~action_supervision & action_valid).any():
                raise ValueError(
                    "samples without action supervision must have an all-zero action_mask"
                )

    @staticmethod
    def _loss_outputs(
        *,
        ar_loss,
        ar_loss_sum,
        ar_loss_count,
        flow_matching_loss,
        flow_matching_loss_sum,
        flow_matching_loss_count,
        vlm_loss_weight: float,
        action_expert_loss_weight: float,
    ) -> BatchFeature:
        data = {
            "vlm_loss_weight": vlm_loss_weight,
            "action_expert_loss_weight": action_expert_loss_weight,
            "vlm_loss weight": vlm_loss_weight,
            "action_expert_loss weight": action_expert_loss_weight,
        }
        total_loss = None
        if ar_loss is not None:
            weighted_ar_loss = vlm_loss_weight * ar_loss
            data.update(
                {
                    "vlm_loss": ar_loss,
                    "ar_loss": ar_loss,
                    "weighted_vlm_loss": weighted_ar_loss,
                    "weighted_ar_loss": weighted_ar_loss,
                    "ar_loss_sum": ar_loss_sum,
                    "ar_loss_count": ar_loss_count,
                }
            )
            total_loss = weighted_ar_loss
        if flow_matching_loss is not None:
            weighted_flow_matching_loss = action_expert_loss_weight * flow_matching_loss
            data.update(
                {
                    "action_expert_loss": flow_matching_loss,
                    "flow_matching_loss": flow_matching_loss,
                    "weighted_action_expert_loss": weighted_flow_matching_loss,
                    "weighted_flow_matching_loss": weighted_flow_matching_loss,
                    "flow_matching_loss_sum": flow_matching_loss_sum,
                    "flow_matching_loss_count": flow_matching_loss_count,
                }
            )
            total_loss = (
                weighted_flow_matching_loss
                if total_loss is None
                else total_loss + weighted_flow_matching_loss
            )
        data["total_loss"] = total_loss
        data["loss"] = total_loss
        return BatchFeature(data=data)

    def forward(
        self,
        batch_inputs,
        training_progress: float,
        dynamic_loss_type=None,
        vlm_loss_weight=1.0,
        action_expert_loss_weight=1.0,
        *,
        loss_type=None,
    ) -> BatchFeature:
        resolved_loss_type = self._resolve_training_loss_type(
            dynamic_loss_type, loss_type
        )
        if self.action_expert is None and resolved_loss_type not in {"vlm", "aux"}:
            raise ValueError("ar_only model only supports loss_type=vlm")
        vlm_loss_weight = self._validate_loss_weight(
            vlm_loss_weight,
            "vlm_loss_weight",
            required=resolved_loss_type in ("vlm", "vlm_and_action"),
        )
        action_expert_loss_weight = self._validate_loss_weight(
            action_expert_loss_weight,
            "action_expert_loss_weight",
            required=resolved_loss_type in ("action", "vlm_and_action"),
        )
        if (
            self.use_difference_query
            and self.detach_vlm_outputs_for_action_expert
            and resolved_loss_type in ("action", "vlm_and_action")
        ):
            raise ValueError(
                "detach_vlm_outputs_for_action_expert=True is incompatible with "
                "Difference Query action training"
            )
        self._validate_training_inputs(batch_inputs, resolved_loss_type)
        if resolved_loss_type in ("action", "vlm_and_action"):
            actual_horizon = batch_inputs["action"].shape[-2]
            expected_horizon = self.action_expert_config.action_horizon
            if actual_horizon != expected_horizon:
                raise ValueError(
                    f"action horizon mismatch: model={expected_horizon}, batch={actual_horizon}"
                )
        backbone_inputs = self.backbone.prepare_inputs(batch_inputs)
        backbone_outputs = self.backbone(
            backbone_inputs,
            compute_vlm_loss=resolved_loss_type in ("vlm", "vlm_and_action"),
        )
        flow_outputs = None
        slot_outputs = None
        if getattr(self, "slot_aux", None) is not None:
            from utils.slot_loss import slot_loss
            query_states = backbone_outputs["backbone_embeddings"][:, -self.num_difference_queries:, :]
            slot_outputs = slot_loss(self.slot_aux(query_states), batch_inputs, self.slot_config, self.slot_supervision_stats)
        if getattr(self, "optical_flow_aux", None) is not None:
            from utils.optical_flow_loss import optical_flow_loss
            prediction = self.optical_flow_aux(backbone_outputs["backbone_embeddings"][:, -self.optical_flow_config.num_flow_queries:, :])
            flow_outputs = optical_flow_loss(prediction, batch_inputs, self.optical_flow_config)
        if resolved_loss_type == "aux":
            weighted = backbone_outputs["backbone_embeddings"].sum() * 0
            if flow_outputs is not None:
                weighted = weighted + self.optical_flow_config.optical_flow_loss_weight * flow_outputs["optical_flow_loss"]
            if slot_outputs is not None:
                weighted = weighted + slot_outputs["slot_loss_weighted"]
            return BatchFeature({**(flow_outputs or {}), **(slot_outputs or {}), "loss": weighted, "total_loss": weighted,
                                 **stage_loss_metadata(self.training_stage,
                                     slot=slot_outputs is not None and slot_outputs["slot_valid_samples"] > 0,
                                     slot_enabled=getattr(getattr(self, "slot_config", None), "enabled", False),
                                     flow=flow_outputs is not None and flow_outputs["optical_flow_loss_count"] > 0)})
        vlm_loss = backbone_outputs.get("vlm_loss")
        ar_loss_count = None
        ar_loss_sum = None
        if isinstance(vlm_loss, torch.Tensor):
            ar_loss_count = (
                batch_inputs["labels"][..., 1:].ne(-100).sum().detach().to(torch.float32)
            )
            ar_loss_sum = vlm_loss.float() * ar_loss_count

        if resolved_loss_type == "vlm":
            if not isinstance(vlm_loss, torch.Tensor):
                raise ValueError("backbone did not return vlm_loss for vlm training")
            result = self._loss_outputs(
                ar_loss=vlm_loss,
                ar_loss_sum=ar_loss_sum,
                ar_loss_count=ar_loss_count,
                flow_matching_loss=None,
                flow_matching_loss_sum=None,
                flow_matching_loss_count=None,
                vlm_loss_weight=vlm_loss_weight,
                action_expert_loss_weight=action_expert_loss_weight,
            )
            if getattr(self, "training_stage", None):
                result.update(stage_loss_metadata(self.training_stage, ar=ar_loss_count > 0))
            return result

        if self.detach_vlm_outputs_for_action_expert:
            backbone_outputs["backbone_embeddings"] = backbone_outputs["backbone_embeddings"].detach()

        self._validate_action_conditioning(
            backbone_outputs, expected_batch_size=batch_inputs["input_ids"].shape[0]
        )
        
        action_expert = self._require_action_expert()
        action_expert_inputs = action_expert.prepare_inputs(batch_inputs)
        action_expert_outputs = action_expert(
            backbone_outputs, action_expert_inputs, training_progress
        )
        action_expert_loss = action_expert_outputs["action_expert_loss"]
        flow_matching_loss_count = action_expert_outputs.get(
            "flow_matching_loss_count"
        )
        flow_matching_loss_sum = action_expert_outputs.get("flow_matching_loss_sum")
        if flow_matching_loss_count is None or flow_matching_loss_sum is None:
            flow_matching_loss_count = (
                batch_inputs["action_mask"].sum().detach().to(torch.float32)
            )
            flow_matching_loss_sum = (
                action_expert_loss.float() * flow_matching_loss_count
            )

        if resolved_loss_type == "vlm_and_action" and not isinstance(vlm_loss, torch.Tensor):
            raise ValueError("backbone did not return vlm_loss for vlm_and_action training")
        result = self._loss_outputs(
            ar_loss=vlm_loss if resolved_loss_type == "vlm_and_action" else None,
            ar_loss_sum=ar_loss_sum if resolved_loss_type == "vlm_and_action" else None,
            ar_loss_count=ar_loss_count if resolved_loss_type == "vlm_and_action" else None,
            flow_matching_loss=action_expert_loss,
            flow_matching_loss_sum=flow_matching_loss_sum,
            flow_matching_loss_count=flow_matching_loss_count,
            vlm_loss_weight=vlm_loss_weight,
            action_expert_loss_weight=action_expert_loss_weight,
        )
        result.update({key: value for key, value in action_expert_outputs.items() if key.startswith("gr1_")})
        if flow_outputs is not None:
            result.update(flow_outputs)
            result["loss"] = result["loss"] + self.optical_flow_config.optical_flow_loss_weight * flow_outputs["optical_flow_loss"]
            result["total_loss"] = result["loss"]
        if getattr(self, "training_stage", None):
            if slot_outputs is not None:
                result.update(slot_outputs)
                result["loss"] = result["loss"] + slot_outputs["slot_loss_weighted"]
                result["total_loss"] = result["loss"]
            result.update(stage_loss_metadata(self.training_stage,
                slot=slot_outputs is not None and slot_outputs["slot_valid_samples"] > 0,
                slot_enabled=getattr(getattr(self, "slot_config", None), "enabled", False),
                ar=ar_loss_count is not None and ar_loss_count > 0,
                flow=flow_outputs is not None and flow_outputs["optical_flow_loss_count"] > 0,
                fm=flow_matching_loss_count > 0))
        return result
    
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        if getattr(self, "training_stage", None) is not None:
            from utils.slot_checkpoint import save_slot_artifacts
            save_slot_artifacts(self, save_directory)
            save_flow_artifacts(self, save_directory)
        # save backbone model parameter and model config (Qwen-VL)
        self.backbone.model.save_pretrained(save_directory)
        # save backnone's processor
        self.backbone.processor.save_pretrained(save_directory)
        action_weight_path = os.path.join(save_directory, "action_expert.safetensors")
        if self.action_expert is not None:
            save_file(self.action_expert.state_dict(), action_weight_path)
        elif os.path.exists(action_weight_path):
            os.unlink(action_weight_path)
        # AR-only Stage05 checkpoints preserve the exact external config bytes so
        # the future Joint initialization contract remains independently auditable.
        resolved_action_config = json.loads(
            json.dumps(self.action_expert_config.to_dict(), ensure_ascii=True)
        )
        action_config_path = os.path.join(save_directory, "action_expert_config.json")
        is_stage05_ar = is_stage05_four_dataset_manifest(
            self.resolved_dataset_manifest, expected_loss_type="vlm"
        ) and self.loss_type == "vlm"
        is_stage05_joint = is_stage05_four_dataset_manifest(
            self.resolved_dataset_manifest, expected_loss_type="vlm_and_action"
        ) and self.loss_type == "vlm_and_action"
        config_provenance = None
        source_bytes = getattr(self, "action_expert_config_source_bytes", None)
        if getattr(self, "training_stage", None) and source_bytes is not None:
            from utils.action_expert_config import load_action_expert_config

            original_bytes = getattr(self, "action_expert_config_original_bytes", source_bytes)
            original_path = Path(save_directory) / "action_expert_source_config.json"
            original_path.write_bytes(original_bytes)
            source = load_action_expert_config(original_path,
                action_horizon_override=resolved_action_config["action_horizon"])
            if source.payload != resolved_action_config:
                raise ValueError("Action Expert source/runtime configs differ beyond the horizon")
            Path(action_config_path).write_text(json.dumps(resolved_action_config, indent=4, sort_keys=True) + "\n")
            config_provenance = {"version": 1, "source_file": original_path.name,
                "source_action_horizon": source.source_action_horizon,
                "runtime_action_horizon": source.config.action_horizon}
        elif is_stage05_ar or is_stage05_joint:
            source_bytes = getattr(self, "action_expert_config_source_bytes", None)
            if not isinstance(source_bytes, bytes):
                raise ValueError(
                    "Stage05 AR-only checkpoint requires the original Action Expert "
                    "configuration bytes"
                )
            Path(action_config_path).write_bytes(source_bytes)
        else:
            with open(action_config_path, "w") as json_file:
                json.dump(resolved_action_config, json_file, indent=4, sort_keys=True)
                json_file.write("\n")
        save_difference_query_artifacts(
            save_directory,
            enabled=self.use_difference_query,
            hidden_size=self.action_expert_config.vlm_output_embedding_dim,
            difference_query=(
                self.backbone.difference_query.weight
                if self.use_difference_query
                else None
            ),
        )
        generic_contract = None
        if self.action_expert is not None:

            source_horizon = int(
                getattr(
                    self,
                    "action_expert_source_action_horizon",
                    resolved_action_config["action_horizon"],
                )
            )
            target_horizon = int(resolved_action_config["action_horizon"])
            if config_provenance is not None:
                source_horizon = config_provenance["source_action_horizon"]
            runtime_purpose = self.checkpoint_load_purpose or "legacy"
            # A newly saved Stage05 Joint artifact is the completed target of
            # AR-to-Joint initialization, but its future consumer semantics
            # are Joint resume.  Keep that distinction explicit in the
            # generic contract instead of serializing the loader purpose.
            if is_stage05_joint:
                runtime_purpose = STAGE05_JOINT_RESUME
            runtime_contract = {
                "purpose": runtime_purpose,
                "checkpoint_kind": _checkpoint_kind_for_loss_type(self.loss_type),
                "action_horizon": target_horizon,
                "source_action_horizon": source_horizon,
                "target_action_horizon": target_horizon,
            }
            generic_contract = {
                "version": 1,
                "runtime_contract": runtime_contract,
                "config_file": "action_expert_config.json",
                "resolved_config": resolved_action_config,
                "state_dict_shapes": {
                    name: list(value.shape)
                    for name, value in self.action_expert.state_dict().items()
                },
            }
        metadata = {
            "version": CHECKPOINT_METADATA_VERSION,
            "checkpoint_kind": _checkpoint_kind_for_loss_type(self.loss_type),
            "action_expert": {
                "status": (
                    "constructed"
                    if self.action_expert is not None
                    else "not_constructed_future_joint_config_reference"
                ),
                "config_file": "action_expert_config.json",
                "weights_file": (
                    "action_expert.safetensors"
                    if self.action_expert is not None
                    else None
                ),
            },
        }
        if getattr(self, "training_stage", None):
            metadata.update(checkpoint_stage_metadata(self))
            metadata.update(slot_aux_type=self.slot_config.slot_aux_type, query_role_layout=self.query_role_layout)
            metadata.update(slot_runtime_config=self.slot_config.to_dict())
        if generic_contract is not None:
            metadata["action_expert_contract"] = generic_contract
        if config_provenance is not None:
            metadata["action_expert_config_provenance"] = config_provenance
        if is_stage05_ar or is_stage05_joint:
            metadata[STAGE05_AR_JOINT_CONTRACT_KEY] = build_stage05_ar_joint_contract(
                checkpoint_directory=save_directory,
                resolved_action_expert_config=resolved_action_config,
                resolved_dataset_manifest=self.resolved_dataset_manifest,
                vlm_hidden_size=self.backbone.model.config.text_config.hidden_size,
                num_difference_queries=self.num_difference_queries,
                checkpoint_kind="ar_only" if is_stage05_ar else "joint",
                runtime_purpose=(
                    STAGE05_AR_TO_JOINT if is_stage05_ar else STAGE05_JOINT_RESUME
                ),
            )
        with open(
            os.path.join(save_directory, CHECKPOINT_METADATA_NAME),
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(checkpoint_metadata(metadata), metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        if self.resolved_dataset_manifest is not None:
            write_resolved_dataset_manifest(
                save_directory, self.resolved_dataset_manifest
            )
    
        if getattr(self, "portable_run", False):
            from utils.portable_run import save_portable_artifacts
            save_portable_artifacts(self, Path(save_directory))

    @classmethod
    def from_pretrained(
        cls,
        save_directory,
        tune_vlm=False,
        tune_action_expert=False,
        detach_vlm_outputs_for_action_expert=False,
        use_difference_query=None,
        num_difference_queries=None,
        vlm_attention_backend=None,
        loss_type=None,
        allow_ar_warm_start=False,
        for_action_inference=False,
        checkpoint_load_purpose=None,
        action_horizon=None,
        action_expert_config_path=None,
        resume_training=False,
        training_stage=None,
        optical_flow_config=None,
        slot_config=None,
        slot_supervision_stats=None,
        init_from_checkpoint=None,
        resume_from_checkpoint=None,
        action_expert_name_or_path=None,
        action_conditioning_norm="final_rmsnorm",
    ):
        from utils.optical_flow_checkpoint import read_flow_artifacts
        flow_payload = read_flow_artifacts(save_directory)
        if not for_action_inference and checkpoint_load_purpose != INFERENCE:
            training_stage = training_stage or (flow_payload["training_stage"] if flow_payload else None)
        if training_stage is not None and not for_action_inference and checkpoint_load_purpose != INFERENCE:
            from utils.action_expert_config import load_action_expert_config
            config_path = action_expert_config_path or Path(save_directory) / "action_expert_config.json"
            horizon_argument = ("action_horizon_override" if checkpoint_load_purpose == DOWNSTREAM_FINETUNE
                                and init_from_checkpoint else "expected_action_horizon")
            arguments = {horizon_argument: action_horizon} if action_horizon is not None else {}
            expert_config = load_action_expert_config(config_path, **arguments).config
            expert_source = action_expert_name_or_path
            restore_saved_stage = (flow_payload is not None and flow_payload["training_stage"] == training_stage
                                   and not init_from_checkpoint)
            resume_training = resume_training or restore_saved_stage
            if (resume_from_checkpoint or resume_training) and training_stage in {"stage3_joint", "downstream_flow"}:
                expert_source = str(save_directory)
            return cls(str(save_directory), expert_source, expert_config,
                tune_vlm=tune_vlm, tune_action_expert=tune_action_expert,
                use_difference_query=use_difference_query, num_difference_queries=num_difference_queries,
                vlm_attention_backend=vlm_attention_backend, loss_type=loss_type,
                action_conditioning_norm=action_conditioning_norm,
                training_stage=training_stage, optical_flow_config=optical_flow_config,
                slot_config=slot_config, slot_supervision_stats=slot_supervision_stats,
                init_from_checkpoint=init_from_checkpoint, resume_from_checkpoint=resume_from_checkpoint,
                resume_training=resume_training, checkpoint_load_purpose=checkpoint_load_purpose,
                detach_vlm_outputs_for_action_expert=detach_vlm_outputs_for_action_expert,
                action_expert_config_path=action_expert_config_path)
        loss_type = loss_type or "vlm_and_action"
        validate_checkpoint_load_purpose_arguments(
            checkpoint_load_purpose, resume_training=resume_training
        )
        checkpoint_kind = _read_checkpoint_kind(save_directory)
        action_weights = Path(save_directory) / "action_expert.safetensors"
        if checkpoint_kind is None:
            checkpoint_kind = "legacy_full" if action_weights.is_file() else "legacy_ar"
            print(
                f"legacy checkpoint without {CHECKPOINT_METADATA_NAME}: "
                f"using conservative {checkpoint_kind} compatibility path"
            )
        if checkpoint_load_purpose is not None and checkpoint_load_purpose not in CHECKPOINT_LOAD_PURPOSES:
            raise ValueError(
                f"unknown checkpoint load purpose: {checkpoint_load_purpose!r}"
            )
        metadata_path = Path(save_directory) / CHECKPOINT_METADATA_NAME
        metadata_payload = {}
        if metadata_path.is_file():
            try:
                metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as error:
                raise ValueError(f"failed to read checkpoint metadata {metadata_path}: {error}") from error
        stage05_contract_present = isinstance(
            metadata_payload.get(STAGE05_AR_JOINT_CONTRACT_KEY), dict
        )
        generic_contract_present = "action_expert_contract" in metadata_payload
        stage05_identity = _checkpoint_has_stage05_identity(
            Path(save_directory).resolve(), metadata_payload
        )
        if stage05_identity and not stage05_contract_present:
            raise ValueError(
                "checkpoint has Stage05 dataset identity but its Stage05 contract is missing or damaged"
            )
        if stage05_identity and checkpoint_load_purpose is None:
            raise ValueError(
                "Stage05 checkpoint requires an explicit checkpoint_load_purpose; "
                "use stage05_ar_resume, stage05_ar_to_joint, stage05_joint_resume, downstream_finetune, "
                "or inference"
            )
        # The legacy boolean remains accepted only for generic legacy callers;
        # a Stage05 AR checkpoint can never use it as a contract bypass.
        if (
            checkpoint_load_purpose is None
            and allow_ar_warm_start
            and checkpoint_kind == "ar_only"
            and (stage05_contract_present or generic_contract_present)
        ):
            raise ValueError(
                "Stage05 AR checkpoint requires explicit "
                "checkpoint_load_purpose='stage05_ar_to_joint'"
            )
        validated_config = None
        validation_purpose = checkpoint_load_purpose
        should_validate_purpose = validation_purpose in {
            STAGE05_AR_RESUME,
            STAGE05_AR_TO_JOINT,
            STAGE05_JOINT_RESUME,
            DOWNSTREAM_FINETUNE,
            INFERENCE,
        }
        if validation_purpose == STAGE05_AR_RESUME and loss_type != "vlm":
            raise ValueError("stage05_ar_resume requires loss_type=vlm")
        if should_validate_purpose:
            validated_config = validate_stage05_checkpoint_for_purpose(
                save_directory,
                purpose=validation_purpose,
                external_config_path=action_expert_config_path,
                requested_action_horizon=action_horizon,
                resume_training=resume_training,
                expected_action_dim=64 if stage05_identity else None,
                expected_state_dim=64 if stage05_identity else None,
                expected_num_difference_queries=(
                    num_difference_queries if num_difference_queries is not None else 32
                ),
            )
            if checkpoint_load_purpose == STAGE05_AR_TO_JOINT:
                loss_type = "vlm_and_action"
            elif checkpoint_load_purpose in {STAGE05_JOINT_RESUME, DOWNSTREAM_FINETUNE, INFERENCE}:
                loss_type = "action" if for_action_inference else loss_type
        elif generic_contract_present:
            # Generic contracts are newer artifacts too.  They may retain the
            # historical no-purpose API, but must still be validated strictly
            # rather than falling through to an unchecked config load.
            validated_config = validate_generic_action_expert_contract(save_directory)
        if for_action_inference:
            if checkpoint_kind in {"ar_only", "legacy_ar"}:
                raise ValueError(
                    f"checkpoint kind {checkpoint_kind!r} cannot be used for action inference"
                )
            if checkpoint_kind not in {"action_only", "joint", "legacy_full"}:
                raise ValueError(
                    f"checkpoint kind {checkpoint_kind!r} is not action-inference compatible"
                )
            loss_type = "action"
        if loss_type == "vlm" and checkpoint_kind not in {"ar_only", "legacy_ar", "legacy_full"}:
            raise ValueError(
                f"checkpoint kind {checkpoint_kind!r} cannot be loaded for loss_type=vlm"
            )
        if loss_type == "vlm_and_action" and checkpoint_kind == "action_only":
            raise ValueError(
                "checkpoint kind 'action_only' cannot initialize loss_type=vlm_and_action"
            )
        if (
            loss_type == "vlm_and_action"
            and checkpoint_kind in {"ar_only", "legacy_ar"}
            and not allow_ar_warm_start
            and checkpoint_load_purpose != STAGE05_AR_TO_JOINT
        ):
            raise ValueError(
                "AR checkpoint to joint initialization requires "
                "allow_ar_warm_start=True"
            )
        if (
            loss_type == "action"
            and checkpoint_kind not in {"action_only", "legacy_full"}
            and not (
                checkpoint_kind == "joint"
                and (for_action_inference or checkpoint_load_purpose == DOWNSTREAM_FINETUNE)
            )
        ):
            raise ValueError(
                f"checkpoint kind {checkpoint_kind!r} cannot be loaded for loss_type=action"
            )
        if validated_config is not None:
            action_expert_config = validated_config.config
        else:
            with open(
                os.path.join(save_directory, "action_expert_config.json"), encoding="utf-8"
            ) as config_file:
                action_expert_config_json = json.load(config_file)
            action_expert_config = FlowmatchingActionHeadConfig(**action_expert_config_json)

        load_action_weights = (
            loss_type != "vlm"
            and checkpoint_kind in {"joint", "action_only", "legacy_full"}
        )
        model = cls(
            vlm_name_or_path = save_directory,
            action_expert_name_or_path=(save_directory if load_action_weights else None),
            action_expert_config = action_expert_config,
            tune_vlm = tune_vlm,
            tune_action_expert = tune_action_expert,
            detach_vlm_outputs_for_action_expert = detach_vlm_outputs_for_action_expert,
            use_difference_query = use_difference_query,
            num_difference_queries = num_difference_queries,
            vlm_attention_backend = vlm_attention_backend,
            action_conditioning_norm=action_conditioning_norm,
            loss_type=loss_type,
            checkpoint_load_purpose=checkpoint_load_purpose,
            resume_training=resume_training,
            action_expert_config_path=action_expert_config_path,
        )
        if flow_payload:
            model.source_stage_training_state = read_stage_training_state(
                flow_payload, OpticalFlowConfig(**flow_payload["config"]))
        return model

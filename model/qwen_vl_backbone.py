import torch
from types import MethodType

from torch import nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import is_flash_attn_2_available

from .difference_query import (
    DifferenceQuery,
    ResolvedDifferenceQueryConfig,
    build_difference_query_sequence,
    resolve_difference_query_config,
)


QUERY_PLACEHOLDER_TOKEN_ID = 0
QUERY_GENERATION_ERROR = (
    "autoregressive generation is not implemented with Difference Query"
)


def validate_action_conditioning_norm(mode: str, *, use_difference_query: bool = False) -> str:
    if mode not in {"final_rmsnorm", "raw"}:
        raise ValueError("action_conditioning_norm must be 'final_rmsnorm' or 'raw'")
    if mode == "raw" and use_difference_query:
        raise ValueError("raw action conditioning is incompatible with Difference Query")
    return mode


def _reject_difference_query_generation(_model, *args, **kwargs):
    raise NotImplementedError(QUERY_GENERATION_ERROR)


def _unwrap_qwen_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _get_qwen_final_text_norm(model: nn.Module) -> nn.Module:
    model = _unwrap_qwen_model(model)
    return model.model.language_model.norm


def _get_qwen_multimodal_model(model: nn.Module) -> nn.Module:
    return _unwrap_qwen_model(model).model


def _get_qwen_hidden_size(model: nn.Module) -> int:
    return int(_unwrap_qwen_model(model).config.text_config.hidden_size)


def _get_qwen_initializer_std(model: nn.Module) -> float:
    text_config = _unwrap_qwen_model(model).config.text_config
    return float(getattr(text_config, "initializer_range", 0.02))


class QwenVLBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        tune_vlm: bool,
        lora_args: dict,
        use_difference_query: bool | None = None,
        num_difference_queries: int | None = None,
        vlm_attention_backend: str | None = None,
        resolved_difference_query_config: ResolvedDifferenceQueryConfig | None = None,
        action_conditioning_norm: str = "final_rmsnorm",
    ):
        super().__init__()

        self.tune_vlm = tune_vlm
        if resolved_difference_query_config is None:
            resolved_difference_query_config = resolve_difference_query_config(
                model_name,
                None,
                use_difference_query=use_difference_query,
                num_difference_queries=num_difference_queries,
                vlm_attention_backend=vlm_attention_backend,
            )
        elif any(
            value is not None
            for value in (
                use_difference_query,
                num_difference_queries,
                vlm_attention_backend,
            )
        ):
            raise ValueError(
                "pass either resolved_difference_query_config or raw Difference Query options"
            )

        self.action_conditioning_norm = validate_action_conditioning_norm(
            action_conditioning_norm, use_difference_query=resolved_difference_query_config.enabled
        )
        if resolved_difference_query_config.attention_backend is not None:
            attn_implementation = resolved_difference_query_config.attention_backend
        elif is_flash_attn_2_available():
            attn_implementation = "flash_attention_2"
        else:
            print(
                "[Warning] Flash Attention 2 is not available. "
                "It is highly recommended to install flash-attn for faster training speed "
                "and lower GPU memory usage."
            )
            attn_implementation = "eager"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation
        )
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_name, trust_remote_code=True,
        #     attn_implementation=attn_implementation
        # )

        if lora_args and not self.tune_vlm:
            raise ValueError("you set tune_vlm=False, which means that the parameters of the VLM do not change in any way, \
                             therefore you can not set use_lora=True.")

        if lora_args and lora_args["use_lora"]:
            target_modules = [target_module.strip() for target_module in lora_args["target_modules"].split(',')]
            print("Lora target_modules:", target_modules)
            peft_config = LoraConfig(
                task_type = TaskType.CAUSAL_LM, 
                target_modules = target_modules, 
                r = lora_args["r"], 
                lora_alpha = lora_args["lora_alpha"], 
                lora_dropout = lora_args["lora_dropout"]
            )
            self.model = get_peft_model(self.model, peft_config)

        if self.tune_vlm:
            self.model.gradient_checkpointing_enable()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.use_difference_query = resolved_difference_query_config.enabled
        self.num_difference_queries = (
            resolved_difference_query_config.num_difference_queries
        )
        self.query_placeholder_token_id = QUERY_PLACEHOLDER_TOKEN_ID
        self.difference_query = None
        hidden_size = _get_qwen_hidden_size(self.model)
        expected_hidden_size = resolved_difference_query_config.expected_hidden_size
        if expected_hidden_size is not None and expected_hidden_size != hidden_size:
            raise ValueError(
                "Difference Query checkpoint hidden size does not match the VLM: "
                f"{expected_hidden_size} != {hidden_size}"
            )
        if self.use_difference_query:
            self._validate_query_placeholder_token()
            self.difference_query = DifferenceQuery(
                num_queries=self.num_difference_queries,
                hidden_size=hidden_size,
                initializer_std=_get_qwen_initializer_std(self.model),
            )
            checkpoint_tensor = resolved_difference_query_config.checkpoint_tensor
            if checkpoint_tensor is not None:
                with torch.no_grad():
                    self.difference_query.weight.copy_(checkpoint_tensor)
            self._install_generation_guard()

        actual_backend = getattr(
            _unwrap_qwen_model(self.model).config,
            "_attn_implementation",
            attn_implementation,
        )
        if self.use_difference_query and actual_backend != "sdpa":
            raise ValueError(
                f"Difference Query requires SDPA, but Qwen loaded {actual_backend!r}"
            )
        print(f"Qwen attention backend: {actual_backend}")
        self.set_trainable_parameters()

    def _validate_query_placeholder_token(self) -> None:
        tokenizer = self.processor.tokenizer
        token = tokenizer.convert_ids_to_tokens(self.query_placeholder_token_id)
        if token != "!":
            raise ValueError(
                "Difference Query placeholder token id 0 must map to the ordinary token '!'"
            )

        special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        config = _unwrap_qwen_model(self.model).config
        for attribute in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "pad_token_id",
            "eos_token_id",
            "bos_token_id",
        ):
            value = getattr(config, attribute, None)
            if isinstance(value, int):
                special_ids.add(value)
        for value in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
        ):
            if isinstance(value, int):
                special_ids.add(value)
        if self.query_placeholder_token_id in special_ids:
            raise ValueError(
                "Difference Query placeholder token id 0 must not be a special token"
            )

    def set_trainable_parameters(self):
        if not self.tune_vlm:
            self.model.requires_grad_(False)

    def _install_generation_guard(self) -> None:
        generation_models = (self.model, _unwrap_qwen_model(self.model))
        guarded_ids = set()
        for generation_model in generation_models:
            if id(generation_model) in guarded_ids or not hasattr(
                generation_model, "generate"
            ):
                continue
            generation_model.generate = MethodType(
                _reject_difference_query_generation, generation_model
            )
            guarded_ids.add(id(generation_model))

    def generate(self, *args, **kwargs):
        if self.use_difference_query:
            raise NotImplementedError(QUERY_GENERATION_ERROR)
        return self.model.generate(*args, **kwargs)
        
    def set_frozen_modules_to_eval_mode(self):
        if self.training and not self.tune_vlm:
            self.model.eval()

    def prepare_inputs(self, batch: dict):
        vl_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "image_grid_thw": batch["image_grid_thw"]
        }

        if "labels" in batch:
            vl_inputs["labels"] = batch["labels"]
        if "sub_task_flag" in batch:
            vl_inputs["sub_task_flag"] = batch["sub_task_flag"]
        
        return BatchFeature(data=vl_inputs)

    # def generate(self, vl_inputs: BatchFeature) -> BatchFeature:
    #     # TODO: we only support batch size = 1 during the inference
    #     if vl_inputs["input_ids"].shape[0] > 1:
    #         raise ValueError("we only support batch size = 1 during the inference, current batch size is: ", vl_inputs["input_ids"].shape[0])

    #     # obtain the VLM's last layer hidden states
    #     vlm_outputs = self.forward(vl_inputs)

    #     return vlm_outputs

    def extract_action_expert_cross_attn_mask_only_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: LongTensor, shape (bs, L)
        Returns: BoolTensor, shape (bs, L)

        Rule: For each row, set True for positions before and including the last occurrence (guaranteed unique) of the token ids [151644, 77091, 198], 
                and False for positions after.
        """
        # 151644 is '<|im_start|>', 77091 is 'assistant'; 198 is '\n'. 
        # They are adjacent tokens in the generation prompt
        a, b, c = 151644, 77091, 198
        bs, L = input_ids.shape
        if L < 3:
            return torch.ones((bs, L), dtype=torch.bool, device=input_ids.device)

        # Ternary neighbor matching yields a Boolean tensor of shape (bs, L-2), 
        # with True indicating that the position is the start of the ternary sequence
        matches3 = (input_ids[:, :-2] == a) & (input_ids[:, 1:-1] == b) & (input_ids[:, 2:] == c)
        
        # check which batch actually found the sub-sequence
        has_match = matches3.any(dim=1)  # (bs,) bool
        
        # Each sequence appears exactly once, use argmax to take the starting position
        start_idx = matches3.to(torch.int64).argmax(dim=1)  # (bs,)
        end_idx = start_idx + 2  # position of the last token of the ternary sequence
        
        # if not found, set end_idx for that row to L-1, so that all mask be set to True
        end_idx = end_idx.masked_fill(~has_match, L-1)
        
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        mask = pos <= end_idx.unsqueeze(1)  # (bs, L), bool
        return mask

    def extract_action_expert_cross_attn_mask_prompt_plus_subtask(self, input_ids):
        """
        input_ids: LongTensor, shape (bs, L)
        Returns: BoolTensor, shape (bs, L)

        Rule:
            For each row, set True for positions before and including the unique occurrence
            of the token id 153718, and False for positions after.

        Note:
            Token id 153718 is guaranteed to appear exactly once per sequence.
        """
        target_token = 153718 # here, 153718 is the token id of '</SUB_TASK>'
        bs, L = input_ids.shape

        # check which batch actually found the target token
        token_mask_sum = (input_ids == target_token).sum(dim=1)
        
        # Find index of the unique target token per batch (shape: bs,)
        idx = (input_ids == target_token).to(torch.int64).argmax(dim=1)
        
        # if not found, set end_idx for that row to L-1, so that all mask be set to True
        idx = idx.masked_fill(token_mask_sum == 0, L-1)

        # Generate boolean mask: True for all positions <= idx
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        mask = pos <= idx.unsqueeze(1)  # (bs, L), bool

        return mask

    def forward(
        self, vl_inputs: BatchFeature, compute_vlm_loss: bool = True
    ) -> BatchFeature:
        # set frozen module to eval
        self.set_frozen_modules_to_eval_mode()

        if self.use_difference_query:
            return self._forward_with_difference_query(
                vl_inputs, compute_vlm_loss=compute_vlm_loss
            )

        use_cache = False if self.training else True
        model_inputs = dict(vl_inputs)
        has_ar_supervision = (
            compute_vlm_loss
            and isinstance(model_inputs.get("labels"), torch.Tensor)
            and model_inputs["labels"][..., 1:].ne(-100).any()
        )
        if compute_vlm_loss and not has_ar_supervision:
            model_inputs.pop("labels", None)
        if not compute_vlm_loss:
            model_inputs.pop("labels", None)
            model_inputs["logits_to_keep"] = torch.empty(
                0,
                dtype=torch.long,
                device=model_inputs["input_ids"].device,
            )
        model_outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=use_cache,
        )

        # Transformers records Qwen3-VL decoder outputs before its final RMSNorm.
        raw_last_hidden_state = model_outputs["hidden_states"][-1]
        # Official legacy ZR-0 b1440d4 conditions its Expert on this raw state.
        embeddings = (
            raw_last_hidden_state
            if getattr(self, "action_conditioning_norm", "final_rmsnorm") == "raw"
            else _get_qwen_final_text_norm(self.model)(raw_last_hidden_state)
        )
        if compute_vlm_loss:
            vlm_loss = (
                model_outputs["loss"]
                if has_ar_supervision
                else embeddings.sum() * 0.0
            )
        else:
            vlm_loss = None

        # let the action expert only attend to the input (i.e., "image+text prompt" part) of the VLM,
        # including the generation prompt '<|im_start|>assistant\n'. Thus, during inference, we should set `add_generation_prompt` to True.
        # action_expert_cross_attn_mask = self.extract_action_expert_cross_attn_mask(vl_inputs["input_ids"])
        
        # Compute both masks
        mask_mode1 = self.extract_action_expert_cross_attn_mask_only_prompt(vl_inputs["input_ids"])
        mask_mode2 = self.extract_action_expert_cross_attn_mask_prompt_plus_subtask(vl_inputs["input_ids"])

        # Flag for mode selection: 0 -> mode1, 1 -> mode2
        sub_task_flag = vl_inputs["sub_task_flag"].unsqueeze(1)  # (bs, 1)

        # Combine masks by mode
        action_expert_cross_attn_mask = torch.where(
            sub_task_flag == 1,
            mask_mode2,
            mask_mode1
        )
        
        return BatchFeature(
            data={
                "backbone_embeddings": embeddings,
                "action_expert_cross_attn_mask": action_expert_cross_attn_mask,
                "vlm_loss": vlm_loss
            }
        )

    def _forward_with_difference_query(
        self, vl_inputs: BatchFeature, *, compute_vlm_loss: bool
    ) -> BatchFeature:
        sequence = build_difference_query_sequence(
            vl_inputs["input_ids"],
            vl_inputs["attention_mask"],
            labels=vl_inputs.get("labels"),
            num_queries=self.num_difference_queries,
            placeholder_token_id=self.query_placeholder_token_id,
        )

        inputs_embeds = self.model.get_input_embeddings()(
            sequence.auxiliary_input_ids
        )
        query_values = self.difference_query.for_batch(
            batch_size=inputs_embeds.shape[0], reference=inputs_embeds
        )
        query_mask = torch.zeros_like(
            sequence.auxiliary_input_ids, dtype=torch.bool
        )
        query_mask.scatter_(1, sequence.query_positions, True)
        inputs_embeds = inputs_embeds.masked_scatter(
            query_mask.unsqueeze(-1).expand_as(inputs_embeds),
            query_values,
        )

        qwen_model = _get_qwen_multimodal_model(self.model)
        position_ids, _ = qwen_model.get_rope_index(
            sequence.auxiliary_input_ids,
            image_grid_thw=vl_inputs.get("image_grid_thw"),
            video_grid_thw=vl_inputs.get("video_grid_thw"),
            attention_mask=sequence.valid_attention_mask,
        )
        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": sequence.attention_mask,
            "position_ids": position_ids,
        }
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        ):
            if key in vl_inputs:
                model_inputs[key] = vl_inputs[key]
        has_ar_supervision = (
            compute_vlm_loss
            and sequence.labels is not None
            and sequence.labels[..., 1:].ne(-100).any()
        )
        if has_ar_supervision:
            model_inputs["labels"] = sequence.labels

        if compute_vlm_loss:
            model_outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
            )
            raw_last_hidden_state = model_outputs["hidden_states"][-1]
            normalized_hidden_state = _get_qwen_final_text_norm(self.model)(
                raw_last_hidden_state
            )
            vlm_loss = (
                model_outputs["loss"]
                if has_ar_supervision
                else normalized_hidden_state.sum() * 0.0
            )
        else:
            model_outputs = _get_qwen_multimodal_model(self.model)(
                **model_inputs,
                return_dict=True,
                use_cache=False,
            )
            normalized_hidden_state = model_outputs.last_hidden_state
            vlm_loss = None
        gather_positions = sequence.query_positions.unsqueeze(-1).expand(
            -1, -1, normalized_hidden_state.shape[-1]
        )
        query_hidden_states = normalized_hidden_state.gather(
            dim=1, index=gather_positions
        )
        action_expert_cross_attn_mask = torch.ones(
            query_hidden_states.shape[:2],
            dtype=torch.bool,
            device=query_hidden_states.device,
        )
        return BatchFeature(
            data={
                "backbone_embeddings": query_hidden_states,
                "action_expert_cross_attn_mask": action_expert_cross_attn_mask,
                "vlm_loss": vlm_loss,
            }
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

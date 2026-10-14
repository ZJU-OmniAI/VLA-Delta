# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

# from .cross_attention_dit import DiT
from .cross_attention_dit import DiT


def distributed_masked_mean(
    elementwise_loss: torch.Tensor,
    mask: torch.Tensor,
    process_group=None,
) -> torch.Tensor:
    """Return the global valid-element mean with DDP-correct local gradients."""
    if elementwise_loss.shape != mask.shape:
        raise ValueError("elementwise_loss and mask must have identical shapes")
    local_sum, local_count = masked_loss_sum_and_count(elementwise_loss, mask)

    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size(process_group) if distributed else 1
    global_count = local_count.clone()
    global_sum = local_sum.detach().to(torch.float32).clone()
    if distributed:
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=process_group)

    has_valid = (global_count > 0).to(dtype=local_sum.dtype)
    denominator = global_count.clamp_min(1.0).to(dtype=local_sum.dtype)
    gradient_loss = local_sum * world_size / denominator
    global_mean = (global_sum / global_count.clamp_min(1.0)).to(
        dtype=local_sum.dtype
    )
    return (
        gradient_loss + (global_mean - gradient_loss.detach())
    ) * has_valid


def masked_loss_sum_and_count(
    elementwise_loss: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if elementwise_loss.shape != mask.shape:
        raise ValueError("elementwise_loss and mask must have identical shapes")
    mask = mask.to(device=elementwise_loss.device, dtype=torch.bool)
    local_sum = (elementwise_loss.float() * mask.float()).sum()
    local_count = mask.sum().detach().to(torch.float32)
    return local_sum, local_count


def swish(x):
    return x * torch.sigmoid(x)

def make_time_weights(N, R=3.0, mode="exp", device="cuda", dtype=torch.float32):
    """
    Create a monotonic increasing time-weight vector of length N, normalized to have mean 1.

    Intended use:
      - Weight per time step t=0..N-1 for sequence losses (e.g., earlier steps weight=1, later steps larger).
      - The last time step has a relative weight of approximately R compared to the first step,
        before normalization. After normalization, the overall mean becomes 1 so the global loss
        scale stays stable.

    Args:
        N (int): Number of time steps.
        R (float): Target relative weight at the final step vs the first step (R >= 1).
        mode (str): Growth shape along time. Options:
            - "linear": linear ramp from 1 to R.
            - "quad": quadratic ramp (slower at start, faster later).
            - "exp": exponential ramp (smooth, multiplicative growth).
        device (str or torch.device): Device for the returned tensor.

    Returns:
        torch.Tensor: Weights of shape [N], monotonically increasing, normalized so mean(weight)=1.
    """
    # Time indices: 0, 1, ..., N-1
    t = torch.arange(N, device=device, dtype=dtype)
    # Normalize time to [0, 1]; handle N=1 to avoid division by zero
    u = t / max(N - 1, 1)  # 0→1
    if mode == "linear":
        # Linear ramp: w(u) = 1 + (R-1) * u
        w = 1.0 + (R - 1.0) * u
    elif mode == "quad":
        # Quadratic ramp: w(u) = 1 + (R-1) * u^2 (keeps early steps closer to 1)
        w = 1.0 + (R - 1.0) * (u ** 2)
    else:  # "exp"
        # Exponential ramp: w(u) = exp( ln(R) * u ); w(0)=1, w(1)=R
        w = torch.exp(torch.log(torch.tensor(R, device=device)) * u)
    
    # Normalize to mean=1 so that overall loss scale remains comparable to the unweighted case
    w = w / w.mean()
    return w  # shape [N]

def schedule_R(training_progress, R_target=3.0, warm_ratio=0.2, mode="cos"):
    """
    Schedule the terminal weight ratio R over training progress.

    Behavior:
      - During the initial warm phase (training_progress < warm_ratio), return 1.0 (no extra emphasis).
      - After warmup, increase smoothly from 1.0 to R_target as progress goes from warm_ratio to 1.0.
      - Use either a linear or cosine ramp for smoothness.

    Args:
        training_progress (float): Global training progress in [0, 1].
        R_target (float): Target final ratio at the end of training (>= 1.0).
        warm_ratio (float): Fraction of training spent with R=1.0 (no extra weighting).
        mode (str): Ramp shape after warmup:
            - "linear": linear increase.
            - "cos": cosine easing (slower at the start/end, smoother).

    Returns:
        float: The current R value in [1.0, R_target] according to the schedule.
    """
    # Keep R=1.0 during the warmup portion
    if training_progress < warm_ratio:
        return 1.0

    # Normalize the post-warmup progress q to [0, 1]
    q = (training_progress - warm_ratio) / max(1e-8, (1 - warm_ratio))
    if mode == "linear":
        # Linear ramp from 0 to 1
        s = q
    else:  # "cos"
        # Cosine ramp: s = 0.5 * (1 - cos(pi * q)), smooth start and end
        s = 0.5 * (1 - math.cos(math.pi * q))
    return 1.0 + (R_target - 1.0) * s

class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )

        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, embedding_dim)

        return enc


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        # cat_ids.shape is (batch_size, )
        selected_W = self.W[cat_ids] # [batch_size, input_dim, hidden_dim]
        selected_b = self.b[cat_ids] # [batch_size, hidden_dim]
        # bmm, performs a batch matrix-matrix product of matrices
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        # obtain SinusoidalPositional Embedding for diffusion steps
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        
        fuse information from actions and timesteps
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x



class FlowmatchingActionHeadConfig(PretrainedConfig):
    """
    Configuration class for FlowMatching Action Head model.
    
    Args:
        add_pos_embed (bool, optional): Whether to add positional embeddings. Defaults to True.
        diffusion_transformer_cfg (dict, optional): Configuration dictionary for the diffusion transformer. 
            Defaults to a comprehensive transformer configuration.
        input_embedding_dim (int, optional): Dimension of input embeddings. Defaults to 2048.
        mlp_hidden_size (int, optional): Hidden size for MLP layers. Defaults to 1024.
        max_seq_len (int, optional): Maximum sequence length (state + action tokens). Defaults to 256.
        action_dim (int, optional): Dimension of action space. Defaults to 8.
        state_dim (int, optional): Dimension of state space. Defaults to 8.
        action_horizon (int, optional): Number of parallel action tokens. Defaults to 20.
        noise_beta_alpha (float, optional): Alpha parameter for Beta distribution (timestep sampling). Defaults to 1.5.
        noise_beta_beta (float, optional): Beta parameter for Beta distribution (timestep sampling). Defaults to 1.0.
        noise_s (float, optional): Parameter for flow matching noise Beta distribution. Defaults to 0.999.
        num_timestep_buckets (int, optional): Number of timestep discretization buckets. Defaults to 1000.
        num_inference_timesteps (int, optional): Number of denoising steps during inference. Defaults to 10.
        max_num_embodiments (int, optional): Maximum number of embodiments. Defaults to 1.
        **kwargs: Additional configuration parameters.
    """
    def __init__(
        self,
        add_pos_embed=True,
        diffusion_transformer_cfg=None,
        vlm_output_embedding_dim=2048,
        dit_output_embedding_dim=1024,
        action_or_state_token_embedding_dim=2048, # DiT hidden size
        mlp_hidden_size=512,
        max_seq_len=256,
        action_dim=64,
        state_dim=64,
        action_horizon=20,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        # max_num_embodiments=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Core model parameters
        self.add_pos_embed = add_pos_embed
        self.vlm_output_embedding_dim = vlm_output_embedding_dim
        self.action_or_state_token_embedding_dim = action_or_state_token_embedding_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.max_seq_len = max_seq_len
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        # self.max_num_embodiments = max_num_embodiments
        
        # Noise scheduling parameters
        self.noise_beta_alpha = noise_beta_alpha  # Alpha for Beta distribution
        self.noise_beta_beta = noise_beta_beta    # Beta for Beta distribution
        self.noise_s = noise_s                    # Noise scale parameter
        self.num_timestep_buckets = num_timestep_buckets  # Discretization denoise (or noise) timestep buckets
        
        # DiT architecture config
        self.diffusion_transformer_cfg = diffusion_transformer_cfg or {
            "num_attention_heads": 32, # 8
            "attention_head_dim": 64, # 64
            "output_dim": dit_output_embedding_dim, # DiT output embedding dim
            "num_layers": 9, # 12
            "dropout": 0.2,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "upcast_attention": False,
            "norm_type": "ada_norm",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-5,
            "max_num_positional_embeddings": 128,
            "positional_embeddings": None,
            "final_dropout": True,
            "interleave_self_attention": True,
            "causal_mask_in_self_attn": False
        }

        # Handle additional custom parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self, config: FlowmatchingActionHeadConfig, tune_action_expert: bool
    ):
        super().__init__()
        assert config.action_or_state_token_embedding_dim == \
            config.diffusion_transformer_cfg["num_attention_heads"] * config.diffusion_transformer_cfg["attention_head_dim"]

        # set config
        self.mlp_hidden_size = config.mlp_hidden_size
        self.vlm_output_embedding_dim = config.vlm_output_embedding_dim
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

        # initialize neural networks
        self.dit = DiT(**config.diffusion_transformer_cfg)

        self.state_encoder = CategorySpecificMLP(
            num_categories=1,
            input_dim=config.state_dim,
            hidden_dim=self.mlp_hidden_size,
            output_dim=config.action_or_state_token_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=config.action_or_state_token_embedding_dim,
            num_embodiments=1,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=1,
            input_dim=config.diffusion_transformer_cfg["output_dim"],
            hidden_dim=self.mlp_hidden_size,
            output_dim=self.action_dim,
        )

        assert config.vlm_output_embedding_dim == config.action_or_state_token_embedding_dim
        # self.vl_embed_proj = nn.Linear(
        #     in_features=config.vlm_output_embedding_dim,
        #     out_features=config.action_or_state_token_embedding_dim,
        #     bias=True
        # )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.action_or_state_token_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.tune_action_expert = tune_action_expert
        self.set_trainable_parameters()

    def set_trainable_parameters(self):
        for p in self.parameters():
            p.requires_grad = True
        if not self.tune_action_expert:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            # self.vl_embed_proj.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            self.dit.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_action_expert:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                # self.vl_embed_proj.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
                self.dit.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_inputs(self, batch: dict) -> BatchFeature:
        action_expert_inputs = {
            "observation.state": batch["observation.state"].to(self.dtype),
            "state_mask": batch["state_mask"]
        }

        if "gr1_constant_dimensions" in batch:
            action_expert_inputs["gr1_constant_dimensions"] = batch["gr1_constant_dimensions"]
        if "action" in batch:
            action_expert_inputs["action"] = batch["action"].to(self.dtype)
            action_expert_inputs["action_mask"] = batch["action_mask"]
        else:
            action_expert_inputs["infer_action_mask"] = batch["infer_action_mask"]

        # for key, inputs in action_expert_inputs.items():
        #     print(f"{key}.shape: {inputs.shape}")
        #     print(f"{key}.dtype: {inputs.dtype}")

        return BatchFeature(data=action_expert_inputs)

    def forward(self, backbone_outputs: BatchFeature, action_expert_inputs: BatchFeature, training_progress: float) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # Get vision and language embeddings
        vl_embeds = backbone_outputs["backbone_embeddings"]
        # vl_embeds = self.vl_embed_proj(vl_embeds) # (batch_size, seq_len, VLM hidden_size) -> (batch_size, seq_len, action_or_state_token_embedding_dim)
        batch_size = vl_embeds.shape[0]

        # Get embodiment ID
        cat_ids = torch.zeros(batch_size, dtype=torch.int64)

        # Embed state
        state_features = self.state_encoder(action_expert_inputs["observation.state"], cat_ids) # (B, 1, action_or_state_token_embedding_dim)

        # Embed noised action trajectory
        actions = action_expert_inputs["action"] # (B, action_horizon, action_dim)
        action_mask = action_expert_inputs["action_mask"] # (B, action_horizon, action_dim)
        M = action_mask.to(device=actions.device, dtype=actions.dtype) # (B, action_horizon, action_dim)
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype) * M

        # sample a timestep for each batch item in the batch (a single scalar per batch item)
        # NOTE: for each item in the batch, all actions share a same diffusion step during training,
        # because all actions in a data item have a same diffusion step during inference
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype) # (B, )
        t = t[:, None, None]  # shape (B,1,1) for broadcast, in order to calcualte noised trajectory

        noisy_trajectory = ((1 - t) * noise + t * actions) * M # (B, action_horizon, action_dim)
        velocity = (actions - noise) * M # (B, action_horizon, action_dim)

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long() # (B, )
        action_features = self.action_encoder(noisy_trajectory, t_discretized, cat_ids) # (B, action_horizon, action_or_state_token_embedding_dim)

        # Maybe add position embedding (action sequence position embedding)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=self.device) # (action_horizon, )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0) # (1, action_horizon, action_or_state_token_embedding_dim)
            action_features = action_features + pos_embs  # (B, action_horizon, action_or_state_token_embedding_dim)

        # Join vision, language, state and action embedding along the sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1) # (B, 1+action_horizon, action_or_state_token_embedding_dim)
        vl_attn_mask = backbone_outputs["action_expert_cross_attn_mask"]

        # print("sa_embs:", sa_embs)
        # print("t_discretized:", t_discretized)
        model_output = self.dit(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=t_discretized,
            encoder_attention_mask=vl_attn_mask,
        ) # (B, 1+action_horizon, action_or_state_token_embedding_dim)

        # print(model_output.shape)
        pred = self.action_decoder(model_output, cat_ids) # (B, 1+action_horizon, action_dim)
        pred_velocity = pred[:, -self.action_horizon :]  # (B, action_horizon, action_dim)

        # R_now = schedule_R(training_progress=training_progress, R_target=2.0, warm_ratio=1.0, mode="cos") # warm_ratio=0.5
        # w = make_time_weights(N=self.action_horizon, R=R_now, mode="exp", device=pred_velocity.device, dtype=actions.dtype) # (action_horizon)

        # # masked + time-weighted MSE
        # w_t = w.view(1, self.action_horizon, 1) # (1, action_horizon, 1)

        # diff2 = (pred_velocity - velocity).pow(2)      # [B, action_horizon, action_dim]
        # weighted_mask = M * w_t                        # [B, action_horizon, action_dim]
        # num = (diff2 * weighted_mask).sum()            # weighted loss sum
        # den = weighted_mask.sum().clamp_min(1e-8)      # weighted mask sum
        # loss = num / den

        # Calculate loss only on unmasked dimensions (where action_mask==1)
        pred_velocity_masked = pred_velocity * M
        elementwise_loss = F.mse_loss(
            pred_velocity_masked, velocity, reduction="none"
        )
        local_loss_sum, local_loss_count = masked_loss_sum_and_count(
            elementwise_loss, action_mask
        )
        loss = distributed_masked_mean(elementwise_loss, action_mask)
        # loss = F.mse_loss(pred_velocity, velocity, reduction="none") * action_mask
        # loss = loss.sum() / action_mask.sum()

        output_dict = {
            "action_expert_loss": loss,
            "flow_matching_loss_sum": local_loss_sum,
            "flow_matching_loss_count": local_loss_count,
        }
        if "gr1_constant_dimensions" in action_expert_inputs:
            from utils.downstream_contracts import ACTIVE_GR1, PLACEHOLDER_GR1
            for name, dims in (("active", ACTIVE_GR1), ("placeholder", PLACEHOLDER_GR1), ("active_constant", ACTIVE_GR1)):
                mask = action_mask[..., list(dims)].detach().bool()
                if name == "active_constant":
                    mask = mask & action_expert_inputs["gr1_constant_dimensions"][:, None, list(dims)]
                output_dict[f"gr1_{name}_sum"] = (elementwise_loss[..., list(dims)].detach().float() * mask).sum()
                output_dict[f"gr1_{name}_count"] = mask.sum().detach()
        return BatchFeature(data=output_dict)

    def get_action(self, backbone_outputs: BatchFeature,
                   action_expert_inputs: BatchFeature, num_denoised_steps: int) -> BatchFeature:
        # Get vision and language embeddings
        vl_embeds = backbone_outputs["backbone_embeddings"]
        # vl_embeds = self.vl_embed_proj(vl_embeds) # (B, seq_len, VLM hidden size) -> (B, seq_len, DiT hidden size)
        
        batch_size = vl_embeds.shape[0]
        # Get embodiment ID
        cat_ids = torch.zeros(batch_size, dtype=torch.int64)

        # Embed state
        state_features = self.state_encoder(action_expert_inputs["observation.state"], cat_ids)  # (B, 1, action_or_state_token_embedding_dim)

        infer_action_mask = action_expert_inputs["infer_action_mask"] # (B, action_horizon, action_dim)
        M = infer_action_mask.to(device=vl_embeds.device, dtype=vl_embeds.dtype)

        # Set initial actions as the sampled noise
        batch_size = vl_embeds.shape[0] # B
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=self.device,
        ) # (B, action_horizon, action_dim)
        actions = actions * M
        
        # each denoised step length is dt
        dt = 1.0 / num_denoised_steps

        # Run denoising steps
        for t in range(num_denoised_steps):
            # print(t)
            # find the timestep bucket of the current denoised step
            t_cont = t / float(num_denoised_steps)  # e.g. goes 0, 1/num_denoised_steps, 2/num_denoised_steps, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory with timestep information
            t_discretized = torch.full(size=(batch_size,), fill_value=t_discretized, device=self.device, dtype=torch.long) # (B, )
            action_features = self.action_encoder(actions * M, t_discretized, cat_ids) # (B, action_horizion, action_or_state_token_embedding_dim)
            
            # Maybe add action sequence position embedding
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=self.device) # (action_horizion, )
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0) # (1, action_horizon, action_or_state_token_embedding_dim)
                action_features = action_features + pos_embs # (B, action_horizion, action_or_state_token_embedding_dim)

            # Join vision, language, state and action embedding along sequence dimension
            sa_embs = torch.cat((state_features, action_features), dim=1)
            vl_attn_mask = backbone_outputs["action_expert_cross_attn_mask"]

            # Run model forward
            model_output = self.dit(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
            )
            pred = self.action_decoder(model_output, cat_ids) # (B, 1+action_horizon, action_dim)

            pred_velocity = pred[:, -self.action_horizon:, :] # (B, action_horizon, action_dim)

            # Update actions using euler integration
            pred_velocity = pred_velocity * M
            actions = (actions + dt * pred_velocity) * M # (B, action_horizon, action_dim)
        
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

if __name__ == "__main__":
    torch.manual_seed(0)

    action_expert_config = FlowmatchingActionHeadConfig(
        action_dim = 64, # 64 is the max padding length of action and state vectors
        state_dim = 64,
        action_horizon = 20
    )

    model = FlowmatchingActionHead(action_expert_config, False, False)
    model.eval()

    state = torch.randn(4, 1, 64)
    state_mask = torch.ones(4, 1, 64)
    action = torch.randn(4, 20, 64)
    action_mask = torch.ones(4, 20, 64)
    vlm_hidden_states = torch.randn(4, 512, 2048)
    attention_mask = torch.ones(4, 512) # seq len = 512

    # state = torch.randn(4, 1, 64)
    # state_mask = torch.ones(4, 1, 64)
    # action = torch.randn(4, 20, 64)
    # action_mask = torch.ones(4, 20, 64)
    # vlm_hidden_states = torch.cat([torch.load("random_tensor.pt"), torch.randn(4, 512, 2048)], dim=1) # random_tensor.pt is torch.randn(4, 222, 2048)
    # attention_mask = torch.cat([torch.zeros(4, 222), torch.ones(4, 512)], dim=1) # seq len = 512 + 222

    action_expert_inputs = {
        "observation.state": state.to(model.dtype),
        "encoder_attention_mask": attention_mask.to(torch.bool),
        "state_mask": state_mask,
        "action": action.to(model.dtype),
        "action_mask": action_mask
    }

    backbone_outputs = {
        "backbone_embeddings": vlm_hidden_states.to(model.dtype)
    }
    model(backbone_outputs, action_expert_inputs)

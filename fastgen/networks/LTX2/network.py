# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Optional, List, Set, Union, Tuple
import types

import torch
import torch.nn as nn
from torch import dtype
from torch.distributed.fsdp import fully_shard

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import LTXVideoTransformer3DModel, AutoencoderKLLTX2Video
from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformerBlock
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from transformers import GemmaTokenizer, Gemma3ForConditionalGeneration

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.basic_utils import str2bool
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


class LTX2TextEncoder:
    """Text encoder for LTX-2 using Gemma 3."""

    def __init__(self, model_id: str):
        self.tokenizer = GemmaTokenizer.from_pretrained(
            model_id,
            cache_dir=os.environ["HF_HOME"],
            subfolder="tokenizer",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=os.environ["HF_HOME"],
            subfolder="text_encoder",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.text_encoder.eval().requires_grad_(False)

    def encode(
        self,
        conditioning: Optional[Any] = None,
        precision: dtype = torch.float32,
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        """Encode text prompts to raw Gemma 3 embeddings."""
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        text_inputs = self.tokenizer(
            conditioning,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=text_inputs.input_ids.to(self.text_encoder.device),
                attention_mask=text_inputs.attention_mask.to(self.text_encoder.device),
                output_hidden_states=True,
                return_dict=True,
            )
            # Raw Gemma hidden states to be projected by connectors
            prompt_embeds = outputs.hidden_states[-1].to(precision)

        return prompt_embeds

    def to(self, *args, **kwargs):
        self.text_encoder.to(*args, **kwargs)
        return self


class LTX2VideoEncoder:
    """Spatio-temporal VAE encoder/decoder for LTX-2."""

    def __init__(self, model_id: str):
        self.vae: AutoencoderKLLTX2Video = AutoencoderKLLTX2Video.from_pretrained(
            model_id,
            cache_dir=os.environ["HF_HOME"],
            subfolder="vae",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.vae.eval().requires_grad_(False)

        # LTX-2 normalization constants
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 1.5305)
        self.shift_factor = getattr(self.vae.config, "shift_factor", 0.0609)

    def encode(self, real_video: torch.Tensor) -> torch.Tensor:
        """Encode videos to 3D latent space."""
        latents = self.vae.encode(real_video, return_dict=False)[0].sample()
        latents = (latents - self.shift_factor) * self.scaling_factor
        return latents

    def decode(self, latents: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latents to video frames with optional timestep conditioning."""
        latents = (latents / self.scaling_factor) + self.shift_factor
        video = self.vae.decode(latents, timestep=timestep, return_dict=False)[0]
        return video.clip(-1.0, 1.0)

    def to(self, *args, **kwargs):
        self.vae.to(*args, **kwargs)
        return self


def classify_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timesteps: torch.Tensor,
    indices: torch.Tensor,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    audio_hidden_states: Optional[torch.Tensor] = None,
    audio_encoder_attention_mask: Optional[torch.Tensor] = None,
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
    **kwargs,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Patched forward pass for LTXVideoTransformer3DModel."""
    if feature_indices is None:
        feature_indices = set()

    # 1. Text Connector Projection
    # Projects Gemma embeddings for video and audio streams
    encoder_hidden_states, audio_encoder_hidden_states = self.connectors(
        encoder_hidden_states
    )

    # 2. Time & Text Embeddings
    temb = self.time_text_embed(timesteps, encoder_hidden_states)
    
    # LTX-2 also prepares a separate audio time embedding
    # Here we simplify or derive it if necessary
    temb_audio = temb 
    
    # 3. Derive Modulation Parameters for Cross-Attention
    # These are typically linear projections of temb/temb_audio
    # Based on the user's Turn 10 snippet, these must be passed to the blocks
    video_ca_scale_shift = self.video_ca_modulation(temb)
    audio_ca_scale_shift = self.audio_ca_modulation(temb_audio)
    video_ca_a2v_gate = self.video_ca_gate(temb)
    audio_ca_v2a_gate = self.audio_ca_gate(temb_audio)

    # 4. Positional Embeddings
    video_rotary_emb = self.pos_embed(indices)
    audio_rotary_emb = None # Placeholder if audio is zeroed
    
    if audio_hidden_states is None:
        audio_hidden_states = torch.zeros_like(hidden_states[:, :0]) 
    
    idx, features = 0, []

    # 5. Dual-Stream Transformer Blocks
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                temb,
                temb_audio,
                video_ca_scale_shift,
                audio_ca_scale_shift,
                video_ca_a2v_gate,
                audio_ca_v2a_gate,
                video_rotary_emb,
                audio_rotary_emb,
                None, # ca_video_rotary_emb
                None, # ca_audio_rotary_emb
                encoder_attention_mask,
                audio_encoder_attention_mask,
            )
        else:
            hidden_states, audio_hidden_states = block(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=video_ca_scale_shift,
                temb_ca_audio_scale_shift=audio_ca_scale_shift,
                temb_ca_gate=video_ca_a2v_gate,
                temb_ca_audio_gate=audio_ca_v2a_gate,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
            )

        if idx in feature_indices:
            features.append(hidden_states.clone())
        if return_features_early and len(features) == len(feature_indices):
            return features
        idx += 1

    # 6. Final Projection
    output = self.proj_out(hidden_states, temb)

    if return_features_early:
        return features
    out = output if len(feature_indices) == 0 else [output, features]

    if return_logvar:
        # temb_dim = 4096
        logvar = self.logvar_linear(temb)
        return out, logvar

    return out


class LTX2(FastGenNetwork):
    """LTX-2 network for text-to-video generation."""

    MODEL_ID = "Lightricks/LTX-2"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        load_pretrained: bool = True,
        **model_kwargs,
    ):
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)
        self.model_id = model_id
        self._initialize_network(model_id, load_pretrained)
        self.transformer.forward = types.MethodType(classify_forward, self.transformer)

        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

    def _initialize_network(self, model_id: str, load_pretrained: bool) -> None:
        in_meta_context = self._is_in_meta_context()
        
        if load_pretrained and not in_meta_context:
            logger.info(f"Loading LTX-2 transformer and connectors from {model_id}")
            self.transformer = LTXVideoTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer"
            )
            # Load LTX2TextConnectors
            self.transformer.connectors = LTX2TextConnectors.from_pretrained(
                model_id, subfolder="connectors"
            )
        else:
            config = LTXVideoTransformer3DModel.load_config(model_id, subfolder="transformer")
            self.transformer = LTXVideoTransformer3DModel.from_config(config)
            conn_config = LTX2TextConnectors.load_config(model_id, subfolder="connectors")
            self.transformer.connectors = LTX2TextConnectors.from_config(conn_config)

        self.transformer.logvar_linear = nn.Linear(4096, 1)

    def _calculate_shift(self, sequence_length: int) -> float:
        """Resolution-dependent shift."""
        base_seq_len, max_seq_len = 1024, 4096
        base_shift, max_shift = 0.5, 1.15
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return sequence_length * m + b

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        neg_condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 4.0,
        num_steps: int = 40,
        **kwargs,
    ) -> torch.Tensor:
        """Generate video samples using Euler flow matching."""
        batch_size, _, frames, height, width = noise.shape
        mu = self._calculate_shift(frames * height * width)

        scheduler = FlowMatchEulerDiscreteScheduler(shift=mu)
        scheduler.set_timesteps(num_steps, device=noise.device)
        timesteps, latents = scheduler.timesteps, noise.clone()

        for timestep in timesteps:
            t = (timestep / 1000.0).expand(batch_size).to(dtype=noise.dtype, device=noise.device)
            t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t)

            if neg_condition is not None and guidance_scale is not None:
                noise_pred = self(
                    torch.cat([latents, latents], dim=0),
                    torch.cat([t, t], dim=0),
                    condition=torch.cat([neg_condition, condition], dim=0),
                    fwd_pred_type="flow"
                )
                uncond, cond = noise_pred.chunk(2)
                noise_pred = uncond + guidance_scale * (cond - uncond)
            else:
                noise_pred = self(latents, t, condition=condition, fwd_pred_type="flow")

            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        return latents

    def fully_shard(self, **kwargs):
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            apply_fsdp_checkpointing(
                self.transformer, check_fn=lambda b: isinstance(b, LTX2VideoTransformerBlock)
            )
        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)
        fully_shard(self.transformer, **kwargs)

    def init_preprocessors(self):
        self.text_encoder = LTX2TextEncoder(self.model_id)
        self.vae = LTX2VideoEncoder(self.model_id)

    def _prepare_video_indices(self, F, H, W, device, dtype):
        t_coords = torch.arange(F, device=device, dtype=dtype)
        h_coords = torch.arange(H, device=device, dtype=dtype)
        w_coords = torch.arange(W, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(t_coords, h_coords, w_coords, indexing="ij"), dim=-1).reshape(-1, 3)

    def _pack_latents(self, x):
        b, c, f, h, w = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(b, -1, c)

    def _unpack_latents(self, x, f, h, w):
        b, _, c = x.shape
        return x.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)

    def forward(self, x_t, t, condition=None, **kwargs):
        b, c, f, h, w = x_t.shape
        indices = self._prepare_video_indices(f, h, w, x_t.device, x_t.dtype)
        model_outputs = self.transformer(
            hidden_states=self._pack_latents(x_t),
            encoder_hidden_states=condition,
            timesteps=t,
            indices=indices,
            **kwargs
        )
        # Logvar handling
        if kwargs.get("return_logvar", False):
            out, logvar = model_outputs
            out = self._unpack_latents(out, f, h, w)
            return out, logvar
        
        # Standard unpack
        out = model_outputs if not isinstance(model_outputs, list) else model_outputs[0]
        return self._unpack_latents(out, f, h, w)
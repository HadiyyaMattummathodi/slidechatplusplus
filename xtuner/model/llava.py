# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AddedToken, AutoConfig, CLIPImageProcessor,
                          CLIPVisionModel, LlamaForCausalLM,
                          LlamaTokenizerFast, LlavaConfig,
                          LlavaForConditionalGeneration, LlavaProcessor)
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.registry import BUILDER
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)

from .torchscale.model.LongNet import make_longnet_from_name
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

print("[DEBUG] LLaVAModel is being imported from:", __file__, flush=True)

def convert_state_dict_to_hf(state_dict, mapping):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('.inv_freq'):
            continue
        for key_to_modify, new_key in mapping.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict

class AdaptiveAvgPool1dLayer(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool1dLayer, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_avg_pool1d(x, self.output_size)
    
class LLaVAModel(BaseModel):

    def __init__(self,
                 llm,
                 freeze_llm=True,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 encoder_name= None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 max_position_embeddings=None,
                 hidden_size=512, # This is the WSI feature dim (C_visual)
                 train_stage='2',
                 enable_long_net=True,
                 use_focus=False,
                 vision_feature_dim=512): # This is the WSI feature dim
        super().__init__()

        # --- FOCUS Token Compression Parameters ---
        self.sim_threshold = 0.7
        self.window_size = 32
        self.L_max = 512
        self.use_focus = use_focus
        self.vision_feature_dim = vision_feature_dim # 512
        
        # self.atok_proj will be defined later after self.llm is built

        self.encoder_name = encoder_name
        self.torch_dtype = torch.float16
        self.to(torch.float16)
        
        # Determine initial LongNet freeze status
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = True
        self.freeze_long_net = True
        
        if train_stage == '1':
            print('train_stage == 1')
            self.freeze_llm = True
            self.freeze_long_net = False # Freeze LLM, train LongNet/Projector
        elif train_stage == '2':
            print('train_stage == 2')
            self.freeze_llm = False # Train LLM, LongNet, and Projector
            self.freeze_long_net = False 

        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)
            self.llm = self._build_from_cfg_or_module(llm)
        
        # --- CRITICAL FIX 1: Define atok_proj after self.llm is built ---
        # This projects visual features (512) to LLM hidden size (e.g., 3584)
        self.llm_hidden_size = self.llm.config.hidden_size 
        
        if self.use_focus and self.vision_feature_dim != self.llm_hidden_size:
            self.atok_proj = nn.Linear(
                self.vision_feature_dim,
                self.llm_hidden_size, 
                bias=True
            )
        else:
            # Use identity if FOCUS is disabled or dims already match
            self.atok_proj = nn.Identity()
        
        # Ensure atok_proj is on the correct initial dtype
        self.atok_proj.to(self.torch_dtype)


        self.enable_long_net = enable_long_net
        if enable_long_net:
            print('enable long net')
        else:
            print('disable long net')

        # --- LongNet Setup ---
        # 1. Initialize LongNet Encoder
        self.LongNet_encoder = make_longnet_from_name(self.encoder_name).to(self.torch_dtype)
        
        # 2. Reduction layer: Map LongNet output (e.g., 1024) back to WSI input dim (512)
        LONGNET_OUTPUT_DIM = hidden_size
        self.longnet_reduce = nn.Linear(LONGNET_OUTPUT_DIM, hidden_size, bias=True).to(self.torch_dtype)
        
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        self.projector_depth = projector_depth
        
        # --- Projector Setup ---
        projector_config = ProjectorConfig(
            visual_hidden_size=hidden_size, # Input to projector is 512 (after LongNet+Reduction)
            llm_hidden_size=self.llm.config.hidden_size,
            depth=self.projector_depth)        

        self.projector = ProjectorModel(projector_config).to(
            self.llm.dtype)
        
        # --- Freezing Layers ---
        if self.freeze_llm:
            print('freeze_llm')
            self.llm.requires_grad_(False)        
        if self.freeze_long_net:
            print('freeze_long_net')
            self.LongNet_encoder.requires_grad_(False)
            self.longnet_reduce.requires_grad_(False)
        

# File: xtuner/model/llava.py (around line 167-175 in your latest version)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
                
            self.projector.enable_input_require_grads()
            # self.atok_proj.enable_input_require_grads()  <-- REMOVE THIS LINE
            self.gradient_checkpointing_enable()

        self.use_llm_lora =  None
        self.use_visual_encoder_lora =  None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}',
                      'current')

        self.visual_select_layer = visual_select_layer
        self._is_init = True
        self.is_first_iter = True

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder (omitted/not used)
        
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        
        # Step 4. LongNet_encoder
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'LongNet_encoder.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'longnet_reduce.' in k})
        
        # Step 5. atok_proj (for FOCUS)
        if not isinstance(self.atok_proj, nn.Identity):
             to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'atok_proj.' in k})
                 
        return to_return

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg,
                                           max_position_embeddings):

        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {'factor': 1}

        orig_rope_scaling_factor = orig_rope_scaling[
            'factor'] if 'factor' in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(
                    math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {
                    'type': 'linear',
                    'factor': scaling_factor
                }

        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = ('LlamaConfig', 'GemmaConfig', 'MistralConfig',
                             'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig',
                             'Starcoder2Config', 'Starcoder2Config',
                             'Phi3Config')
        SUPPORT_FLASH_ATTN2 = ('InternLM2Config', 'LlamaConfig', 'GemmaConfig',
                               'MistralConfig', 'MixtralConfig', 'Qwen2Config',
                               'Qwen2MoeConfig', 'Starcoder2Config',
                               'Starcoder2Config', 'Phi3Config')

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        if getattr(cfg, 'attn_implementation', None) is not None:
            if cfg.attn_implementation == 'flash_attention_2':
                cfg.torch_dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_qlora_zero3(cfg):
        if (not is_deepspeed_zero3_enabled()) or (not hasattr(
                cfg, 'quantization_config')):
            return cfg

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        cfg.torch_dtype = torch_dtype
        quantization_config = cfg.quantization_config
        quantization_config.bnb_4bit_compute_dtype = torch_dtype
        quantization_config.bnb_4bit_quant_storage = torch_dtype

        return cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        cfg = self._prepare_for_qlora_zero3(cfg)
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(
                cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError
            
    def forward(self, data, data_samples=None, mode='loss'):
        print(f"[DEBUG] Entered LLaVAModel.forward, mode={mode}", flush=True)
        print(f"[DEBUG] forward() data keys: {list(data.keys())}", flush=True)

        if self.is_first_iter:
            self.to(data['input_ids'].device)
            self.is_first_iter = False

        if 'pixel_values' in data:
            x = data['pixel_values']
            print("\n================ IMAGE INPUT DEBUG ================")
            print(f"[pixel_values] shape: {tuple(x.shape)}")
            print(f"[pixel_values] dtype:  {x.dtype}")
            print(f"[pixel_values] device: {x.device}")
            print("===================================================\n")

        feat_to_proj = None

        # -------------------------------
        # FOCUS path (token compression)
        # -------------------------------
        if self.use_focus and 'pixel_values' in data:
            # 1) Text features (for text-guided selection)
            input_ids = data["input_ids"]
            text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, H_llm]
            text_features = text_embeds.mean(dim=1)  # [B, H_llm]

            # 2) Raw visual tokens
            feat_to_proj = data['pixel_values'].to(self.llm.dtype)  # [B, N, 512]
            print(f"Step 3_feat_to_proj: {feat_to_proj.squeeze(0).shape}")

            # 3) Adaptive token selection (global, text-guided)
            selected_features, _ = self.adaptive_token_selection(
                feat_to_proj.squeeze(0),      # [N, 512]
                text_features.squeeze(0)      # [H_llm]
            )
            print(f"Step 3_selected_features: {selected_features.shape}")

            # 4) Spatial token compression (local redundancy removal)
            compressed_features = self.spatial_token_compression(
                selected_features,
                text_features.squeeze(0)
            )

            assert compressed_features.dim() == 2, \
                f"compressed_features must be 2D [T, C], got {compressed_features.shape}"
            assert compressed_features.size(1) == self.vision_feature_dim, \
                f"Expected feature dim {self.vision_feature_dim}, got {compressed_features.size(1)}"

            # 5) Pad / trim to fixed length T_expected = L_max
            T_expected = self.L_max
            T_cur = compressed_features.size(0)

            if T_cur < T_expected:
                pad_len = T_expected - T_cur
                pad = torch.zeros(
                    pad_len,
                    compressed_features.size(1),
                    device=compressed_features.device,
                    dtype=compressed_features.dtype,
                )
                compressed_features = torch.cat([compressed_features, pad], dim=0)
            elif T_cur > T_expected:
                compressed_features = compressed_features[:T_expected]

            # 6) Build batch-first features
            feat_to_proj = compressed_features.unsqueeze(0)  # [1, T_expected, 512]

            print("\n================ FOCUS DEBUG ================")
            print(f"[FOCUS] feat_to_proj (batch first) shape: {tuple(feat_to_proj.shape)}")

            # 7) LongNet expects [T, B, C]
            feat_to_proj_longnet = feat_to_proj.permute(1, 0, 2).to(self.torch_dtype)  # [T, 1, 512]
            print(f"[FOCUS] feat_to_proj for LongNet (seq first) shape: {tuple(feat_to_proj_longnet.shape)}")
            print("=============================================\n")

            # 8) Pass through LongNet encoder if enabled
            if self.enable_long_net:
                long_net_output = self.LongNet_encoder(
                    src_tokens=None,
                    token_embeddings=feat_to_proj_longnet
                )["encoder_out"]  # [T, 1, 1024]

                feat_to_proj = long_net_output.permute(1, 0, 2)  # [1, T, 1024]

                # 🔥 Reduce 1024 → 512 before projector
                feat_to_proj = self.longnet_reduce(feat_to_proj)  # [1, T, 512]

                print(f"[FOCUS] after reduction feat_to_proj shape: {tuple(feat_to_proj.shape)}")

        # -------------------------------
        # Non-FOCUS path (original)
        # -------------------------------
        elif 'pixel_values' in data:
            feat_to_proj = data['pixel_values'].to(self.llm.dtype)  # [B, N, 512]
            if self.enable_long_net:
                long_net_output = self.LongNet_encoder(
                    src_tokens=None,
                    token_embeddings=feat_to_proj.permute(1, 0, 2).to(self.torch_dtype)
                )["encoder_out"]  # [T, B, 1024]
                
                feat_to_proj = long_net_output.permute(1, 0, 2)  # [B, T, 1024]
                
                # Apply reduction for the non-FOCUS path as well
                feat_to_proj = self.longnet_reduce(feat_to_proj) # [B, T, 512]


        # -------------------------------
        # Project visual tokens & prepare multimodal inputs
        # -------------------------------
        if 'pixel_values' in data:
            pixel_values = self.projector(feat_to_proj.to(self.llm.dtype))
            data['pixel_values'] = pixel_values
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        # -------------------------------
        # LLaVA forward modes
        # -------------------------------
        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):
        outputs = self.llm(**data)
        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
        
    def encode_image(self, image: torch.Tensor, input_ids: torch.Tensor = None):

        """
        Encode WSI features into pixel_values, optionally with FOCUS.
        Args:
            image: [B, N, 512] WSI patch features
            input_ids: [B, L] text tokens
        Returns:
            pixel_values: [B, T, hidden_size] visual tokens for the LLM
        """

        print("\n================ ENCODE_IMAGE DEBUG ================", flush=True)
        print(f"[ENC] raw image (WSI features) shape: {tuple(image.shape)}", flush=True)

        B, N, C = image.shape
        assert C == self.vision_feature_dim, f"Expected image feat dim {self.vision_feature_dim}, got {C}"

        # -------------------------------------------------
        # A) FOCUS path
        # -------------------------------------------------
        if self.use_focus and input_ids is not None:
            # 1) Text features (for text-guided selection)
            text_embeds = self.llm.get_input_embeddings()(input_ids)    # [B, L, H_llm]
            text_features = text_embeds.mean(dim=1)                     # [B, H_llm]

            # 2) Start from WSI tokens
            feat_to_proj = image.to(self.llm.dtype)                     # [B, N, 512]
            assert B == 1, "Current FOCUS implementation assumes batch size = 1"

            # 3) Adaptive token selection
            selected_features, _ = self.adaptive_token_selection(
                feat_to_proj.squeeze(0),          # [N, 512]
                text_features.squeeze(0)          # [H_llm]
            )
            print(f"[FOCUS] selected_features shape: {tuple(selected_features.shape)}",
                  flush=True)

            # 4) Spatial token compression
            compressed_features = self.spatial_token_compression(
                selected_features,
                text_features.squeeze(0)          # [H_llm]
            )                                     # [T', 512]
            print(f"[FOCUS] compressed_features shape: {tuple(compressed_features.shape)}",
                  flush=True)

            # 5) Pad / trim to L_max
            T_expected = self.L_max
            T_cur = compressed_features.size(0)

            if T_cur < T_expected:
                pad_len = T_expected - T_cur
                pad = torch.zeros(
                    pad_len,
                    compressed_features.size(1),
                    device=compressed_features.device,
                    dtype=compressed_features.dtype,
                )
                compressed_features = torch.cat([compressed_features, pad], dim=0)
            elif T_cur > T_expected:
                compressed_features = compressed_features[:T_expected]

            # 6) Batch-first before LongNet
            feat_to_proj = compressed_features.unsqueeze(0)             # [1, T_expected, 512]

        else:
            # -------------------------------------------------
            # B) Non-FOCUS path (original SlideChat)
            # -------------------------------------------------
            feat_to_proj = image.to(self.llm.dtype)                     # [B, N, 512]
            T_expected = feat_to_proj.size(1)

        # -------------------------------------------------
        # LongNet expects [T, B, C]
        token_embeddings = feat_to_proj.permute(1, 0, 2)                # [T, B, 512]
        print(f"[ENC] token_embeddings (LongNet INPUT after FOCUS/WSI) shape: "
              f"{tuple(token_embeddings.shape)}", flush=True)

        # -------------------------------------------------
        # C) LongNet + projector (common to both paths)
        # -------------------------------------------------
        if self.enable_long_net:
              long_out = self.LongNet_encoder(
                  src_tokens=None,
                  token_embeddings=token_embeddings.to(self.torch_dtype)
              )["encoder_out"]  # [T, B, 1024]
                
              feat_to_proj = long_out.permute(1, 0, 2)  # [B, T, 1024]
              print(f"LongNet output shape: {feat_to_proj.shape}")

              # 🔥 Reduce LongNet output dim → 512
              feat_to_proj = self.longnet_reduce(feat_to_proj)  # [B, T, 512]

              print(f"[ENC] after reduction feat_to_proj shape: "
                    f"{tuple(feat_to_proj.shape)}", flush=True)

        else:
            print("[ENC] LongNet disabled, skipping LongNet_encoder", flush=True)

        pixel_values = self.projector(feat_to_proj.to(self.llm.dtype))  # [B, T, hidden_size]
        print(f"[ENC] pixel_values shape (after projector): "
              f"{tuple(pixel_values.shape)}", flush=True)
        print("====================================================\n", flush=True)

        return pixel_values


    def to_hf(self,
              cfg,
              save_dir,
              fp32=False,
              save_pretrained_kwargs={},
              save_format='xtuner',
              **kwargs):
        if save_format == 'xtuner':
            self.to_xtuner_llava(cfg, save_dir, fp32, save_pretrained_kwargs)
        elif save_format == 'huggingface':
            self.to_huggingface_llava(cfg, save_dir, fp32,
                                      save_pretrained_kwargs)
        elif save_format == 'official':
            self.to_official_llava(cfg, save_dir, fp32, save_pretrained_kwargs)
        else:
            raise NotImplementedError
        
    

    def to_xtuner_llava(self,
                        cfg,
                        save_dir,
                        fp32=False,
                        save_pretrained_kwargs={}):
        # LLM
        self.llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            self.llm.half()
        if self.use_llm_lora:
            llm_path = osp.join(save_dir, 'llm_adapter')
            print_log(f'Saving LLM adapter to {llm_path}', 'current')
            self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
        elif not self.freeze_llm:
            llm_path = save_dir
            print_log(f'Saving LLM tokenizer to {llm_path}', 'current')
            tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(llm_path, **save_pretrained_kwargs)
            print_log(f'Saving LLM to {llm_path}', 'current')
            self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
        self.llm.config.use_cache = False

        # Projector
        projector_path = osp.join(save_dir, 'projector')
        print_log(f'Saving projector to {projector_path}', 'current')
        self.projector.save_pretrained(projector_path,
                                       **save_pretrained_kwargs)

        # LongNet_encoder
        LongNet_encoder_path = osp.join(save_dir, 'LongNet_encoder')
        print_log(f'Saving LongNet_encoder to {LongNet_encoder_path}', 'current')
        self.LongNet_encoder.save_pretrained(LongNet_encoder_path,
                                       **save_pretrained_kwargs)
        
        # LongNet Reduce
        longnet_reduce_path = osp.join(save_dir, 'longnet_reduce')
        print_log(f'Saving longnet_reduce to {longnet_reduce_path}', 'current')
        torch.save(self.longnet_reduce.state_dict(), osp.join(longnet_reduce_path, 'pytorch_model.bin'))
        
        # atok_proj
        if not isinstance(self.atok_proj, nn.Identity):
            atok_proj_path = osp.join(save_dir, 'atok_proj')
            print_log(f'Saving atok_proj to {atok_proj_path}', 'current')
            torch.save(self.atok_proj.state_dict(), osp.join(atok_proj_path, 'pytorch_model.bin'))


    def to_huggingface_llava(self,
                             cfg,
                             save_dir,
                             fp32=False,
                             save_pretrained_kwargs={}):

        LLM_MAPPING = {
            'model': 'language_model.model',
            'lm_head': 'language_model.lm_head',
        }
        VIT_MAPPING = {
            'vision_model': 'vision_tower.vision_model',
        }
        PROJECTOR_MAPPING = {
            'model.0': 'multi_modal_projector.linear_1',
            'model.2': 'multi_modal_projector.linear_2',
        }
        LONGNET_MAPPING = {
            'layers.0': 'LongNet_encoder.layers.0',
            'layers.1': 'LongNet_encoder.layers.1',
            'layer_norm': 'LongNet_encoder.layer_norm'
        }
        LONGNET_REDUCE_MAPPING = {
            'weight': 'longnet_reduce.weight',
            'bias': 'longnet_reduce.bias'
        }
        ATOK_PROJ_MAPPING = {
            'weight': 'atok_proj.weight',
            'bias': 'atok_proj.bias'
        }


        assert getattr(self.llm, 'hf_quantizer', None) is None, \
            'This conversion format does not support quantized LLM.'

        # get state_dict
        llm = self.llm
        if self.use_llm_lora:
            llm = self.llm.merge_and_unload()
        llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            llm.half()

        assert isinstance(llm, LlamaForCausalLM), \
            'This conversion format only supports LlamaForCausalLM.'
        llm_state_dict = llm.state_dict()
        llm_state_dict = convert_state_dict_to_hf(llm_state_dict, LLM_MAPPING)

        visual_encoder_state_dict = {}

        projector_state_dict = self.projector.state_dict()
        projector_state_dict = convert_state_dict_to_hf(
            projector_state_dict, PROJECTOR_MAPPING)

        LongNet_encoder_state_dict = self.LongNet_encoder.state_dict()
        LongNet_encoder_state_dict = convert_state_dict_to_hf(
            LongNet_encoder_state_dict, LONGNET_MAPPING)
            
        longnet_reduce_state_dict = self.longnet_reduce.state_dict()
        longnet_reduce_state_dict = convert_state_dict_to_hf(
            longnet_reduce_state_dict, LONGNET_REDUCE_MAPPING)
            
        atok_proj_state_dict = {}
        if not isinstance(self.atok_proj, nn.Identity):
             atok_proj_state_dict = self.atok_proj.state_dict()
             atok_proj_state_dict = convert_state_dict_to_hf(
                atok_proj_state_dict, ATOK_PROJ_MAPPING)


        state_dict = {
            **projector_state_dict,
            **llm_state_dict,
            **visual_encoder_state_dict,
            **LongNet_encoder_state_dict,
            **longnet_reduce_state_dict,
            **atok_proj_state_dict
        }

        # init model
        text_config = llm.config
        
        vision_config = AutoConfig.from_pretrained(
            'openai/clip-vit-large-patch14', trust_remote_code=True)


        config = LlavaConfig(
            text_config=text_config,
            vision_config=vision_config,
            attn_implementation='eager')

        with init_empty_weights():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*non-meta.*', category=UserWarning)
                model = LlavaForConditionalGeneration(config)
        model.load_state_dict(state_dict, strict=True, assign=True)

        # processor
        cfg.tokenizer.type = LlamaTokenizerFast.from_pretrained
        tokenizer = BUILDER.build(cfg.tokenizer)

        tokenizer.add_tokens(
            AddedToken(DEFAULT_IMAGE_TOKEN, special=True, normalized=False),
            special_tokens=True)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

        image_processor = BUILDER.build(cfg.image_processor)
        assert isinstance(image_processor, CLIPImageProcessor),\
            'This conversion format only supports CLIPImageProcessor.'

        processor = LlavaProcessor(
            tokenizer=tokenizer, image_processor=image_processor)

        # Pad to 64 for performance reasons
        pad_shape = 64

        pre_expansion_embeddings = \
            model.language_model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T
                 @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5 * sigma)

        # We add an image token so we need to resize the model
        ori_vocab_size = config.text_config.vocab_size
        tokenizer_vocab_size = tokenizer.encode('<pad>')[-1]
        added_token = tokenizer_vocab_size - ori_vocab_size

        if added_token > 0:
            model.resize_token_embeddings(ori_vocab_size + added_token,
                                          pad_shape)
            model.language_model.model.embed_tokens.weight.data[
                ori_vocab_size:] = torch.stack(
                    tuple(
                        dist.sample()
                        for _ in range(model.language_model.model.embed_tokens.
                                       weight.data[ori_vocab_size:].shape[0])),
                    dim=0,
                )
            model.language_model.lm_head.weight.data[
                ori_vocab_size:] = torch.stack(
                    tuple(dist.sample()
                          for _ in range(model.language_model.lm_head.weight.
                                         data[ori_vocab_size:].shape[0])),
                    dim=0,
                )
        model.config.image_token_index = tokenizer.encode(
            DEFAULT_IMAGE_TOKEN)[-1]
        model.config.pad_token_id = tokenizer.encode('<pad>')[-1]

        # save
        print_log(f'Saving to {save_dir}', 'current')
        model.save_pretrained(save_dir, **save_pretrained_kwargs)
        processor.save_pretrained(save_dir, **save_pretrained_kwargs)

    def to_official_llava(self,
                          cfg,
                          save_dir,
                          fp32=False,
                          save_pretrained_kwargs={}):

        VIT_MAPPING = {
            'vision_model': 'model.vision_tower.vision_tower.vision_model',
        }
        PROJECTOR_MAPPING = {
            'model.0': 'model.mm_projector.0',
            'model.2': 'model.mm_projector.2',
        }
        LONGNET_MAPPING = {
            'layers.0': 'LongNet_encoder.layers.0',
            'layers.1': 'LongNet_encoder.layers.1',
            'layer_norm': 'LongNet_encoder.layer_norm'
        }
        LONGNET_REDUCE_MAPPING = {
            'weight': 'longnet_reduce.weight',
            'bias': 'longnet_reduce.bias'
        }
        ATOK_PROJ_MAPPING = {
            'weight': 'atok_proj.weight',
            'bias': 'atok_proj.bias'
        }


        try:
            from llava.model import LlavaConfig, LlavaLlamaForCausalLM
        except ImportError:
            raise ImportError(
                'Please install llava with '
                '`pip install git+https://github.com/haotian-liu/LLaVA.git '
                '--no-deps`.')

        assert getattr(self.llm, 'hf_quantizer', None) is None, \
            'This conversion format does not support quantized LLM.'

        # get state_dict
        llm = self.llm
        if self.use_llm_lora:
            llm = self.llm.merge_and_unload()
        llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            llm.half()

        assert isinstance(llm, LlamaForCausalLM), \
            'This conversion format only supports LlamaForCausalLM.'
        llm_state_dict = llm.state_dict()

        visual_encoder_state_dict = {}

        projector_state_dict = self.projector.state_dict()
        projector_state_dict = convert_state_dict_to_hf(
            projector_state_dict, PROJECTOR_MAPPING)

        LongNet_encoder_state_dict = self.LongNet_encoder.state_dict()
        LongNet_encoder_state_dict = convert_state_dict_to_hf(
            LongNet_encoder_state_dict, LONGNET_MAPPING)
            
        longnet_reduce_state_dict = self.longnet_reduce.state_dict()
        longnet_reduce_state_dict = convert_state_dict_to_hf(
            longnet_reduce_state_dict, LONGNET_REDUCE_MAPPING)

        atok_proj_state_dict = {}
        if not isinstance(self.atok_proj, nn.Identity):
             atok_proj_state_dict = self.atok_proj.state_dict()
             atok_proj_state_dict = convert_state_dict_to_hf(
                atok_proj_state_dict, ATOK_PROJ_MAPPING)


        state_dict = {
            **projector_state_dict,
            **llm_state_dict,
            **visual_encoder_state_dict,
            **LongNet_encoder_state_dict,
            **longnet_reduce_state_dict,
            **atok_proj_state_dict
        }

        # init model
        tokenizer = BUILDER.build(cfg.tokenizer)
        image_processor = BUILDER.build(cfg.image_processor)
        assert isinstance(image_processor, CLIPImageProcessor),\
            'This conversion format only supports CLIPImageProcessor.'

        vision_config = AutoConfig.from_pretrained(
            'openai/clip-vit-large-patch14', trust_remote_code=True)


        llava_config_dict = llm.config.__dict__.copy()
        llava_config_dict.update(
            dict(
                image_aspect_ratio='pad',
                mm_hidden_size=vision_config.hidden_size,
                mm_projector_type=f'mlp{self.projector_depth}x_gelu',
                mm_use_im_patch_token=False,
                mm_use_im_start_end=False,
                mm_vision_select_feature='patch',
                mm_vision_select_layer=self.visual_select_layer,
                mm_vision_tower=vision_config.name_or_path,
                unfreeze_mm_vision_tower=need_visual_encoder,
                model_type='llava',
                use_cache=True,
                use_mm_proj=True))

        llava_config = LlavaConfig(**llava_config_dict)

        with init_empty_weights():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*non-meta.*', category=UserWarning)
                model = LlavaLlamaForCausalLM(llava_config)

        model.load_state_dict(state_dict, strict=True, assign=True)

        # save
        print_log(f'Saving to {save_dir}', 'current')

        model.save_pretrained(save_dir, **save_pretrained_kwargs)
        image_processor.save_pretrained(save_dir, **save_pretrained_kwargs)
        tokenizer.save_pretrained(save_dir, **save_pretrained_kwargs)



# FOCUS Redundancy Reduction Injection 


    def compute_patch_similarity(self, x, window_size):
        """Compute similarity between patches within sliding windows"""
        N, D = x.shape
        x_norm = F.normalize(x, p=2, dim=-1)
        
        similarities = []
        selected_indices = []
        
        for i in range(0, N, window_size):
            window = x_norm[i:i+window_size]
            
            # Skip if window is too small
            if len(window) < 2:
                selected_indices.append(torch.arange(i, min(i+window_size, N), device=x.device))
                continue
                
            # Local similarity computation
            window_sim = torch.mm(window, window.t())
            
            # Adaptive thresholding
            if window_sim.numel() > 1:
                threshold = window_sim.mean() + window_sim.std(unbiased=False)
            else:
                threshold = window_sim.mean()
            
            # Select non-redundant patches
            redundant = window_sim.mean(1) > threshold
            keep_indices = torch.where(~redundant)[0] + i
            
            if len(keep_indices) == 0:
                keep_indices = torch.tensor([i], device=x.device)
                
            selected_indices.append(keep_indices)
            similarities.append(window_sim)
        
        if not selected_indices:
            return [], torch.arange(N, device=x.device)
            
        return similarities, torch.cat(selected_indices)


    def adaptive_token_selection(self, features, text_features):
        """Select tokens based on text relevance and local structure"""
        N, D = features.shape

        # 1) Compute local similarities & initial indices
        similarities, indices = self.compute_patch_similarity(features, self.window_size)

        # 2) Use a common dtype/device (match the LLM)
        common_dtype = getattr(self.llm, "dtype", features.dtype)
        device = features.device
        features = features.to(device=device, dtype=common_dtype)
        text_features = text_features.to(device=device, dtype=common_dtype)

        # 3) Project to text dim if needed
        # features.shape[-1] (512) vs text_features.shape[-1] (e.g., 3584)
        if features.shape[-1] != text_features.shape[-1]:
            # Use the pre-defined self.atok_proj, ensure device/dtype match
            self.atok_proj.to(device=device, dtype=common_dtype)
            features_projected = self.atok_proj(features) # [N, 512] -> [N, H_llm]
        else:
            features_projected = features

        # 4) Text-guided importance: [N, H_llm] * [H_llm] -> [N, H_llm] -> sum(dim=-1) -> [N]
        # FIX APPLIED: Dimensions now match: (N, H_llm) * (H_llm) is fine due to broadcasting and the sum
        text_relevance = (features_projected * text_features).sum(dim=-1)

        # 5) Build importance mask & pick top tokens
        importance_mask = torch.zeros(N, device=device, dtype=common_dtype)
        importance_mask[indices] = text_relevance[indices]

        num_tokens = min(self.L_max, N)
        _, selected_indices = torch.topk(importance_mask, k=num_tokens)
        selected_indices, _ = torch.sort(selected_indices)

        selected_features = features[selected_indices]
        return selected_features, selected_indices


    def spatial_token_compression(self, features, text_features):
        """Compress tokens while preserving important information"""
        N, D = features.shape
        
        chunk_size = 8
        compressed_chunks = []
        
        for i in range(0, N, chunk_size):
            chunk = features[i:i+chunk_size]
            if len(chunk) == 1:
                compressed_chunks.append(chunk)
                continue
                
            # Compute chunk similarities
            chunk_norm = F.normalize(chunk, p=2, dim=-1)
            sim = F.cosine_similarity(
                chunk_norm[:-1],
                chunk_norm[1:],
                dim=-1
            )
            
            # Keep first token and dissimilar tokens
            keep_mask = sim < self.sim_threshold
            kept_tokens = torch.cat([
                chunk[:1],
                chunk[1:][keep_mask]
            ])
            compressed_chunks.append(kept_tokens)
        
        compressed_features = torch.cat(compressed_chunks)
        
        # Ensure we don't exceed max length
        if len(compressed_features) > self.L_max:
            compressed_features = compressed_features[:self.L_max]
            
        return compressed_features
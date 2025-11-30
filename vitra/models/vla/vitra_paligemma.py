import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Callable
import copy
import numpy as np
import json

from PIL import Image
from functools import partial

from vitra.utils.tensor_utils import move_masked_to_left, get_mask_of_last_masked_index, move_masked_to_left_ids
from vitra.models.vlm_builder import build_vlm

class VITRA_Paligemma(nn.Module):
    def __init__(
        self,
        configs,
        train_setup_configs=None,
        act_model_configs=None,
        fwd_pred_next_n=1,
        repeated_diffusion_steps:int = 8,
        use_state='DiT',
        use_fov=True,
        use_bf16=False,
        **kwargs,
    ):
        super().__init__()

        self.configs = configs
        self.train_setup_configs = train_setup_configs
        self.act_model_configs = act_model_configs
        self.use_state = use_state
        self.use_fov = use_fov
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self.past_action_window_size = 0
        # chunk_size for action prediction
        self.chunk_size = self.configs.get("fwd_pred_next_n", 16)
        self.future_action_window_size = self.chunk_size-1
        self.state_mask_prob = self.configs.get("state_mask_prob", 0.1)
        self.action_type = self.configs['train_dataset'].get("action_type", "angle")
        self.use_state = use_state
        self.use_fov = use_fov
        self.use_bf16 = use_bf16
        if self.action_type == 'angle':
            self.hand_dim = 51
        elif self.action_type == 'keypoints':
            self.hand_dim = 69

        # Initialize the tokenizer and VLM backbone
        self.tokenizer, self.backbone = self._init_backbone()
        if self.train_setup_configs is not None and self.train_setup_configs.get("reinit", False):
            initialize_param(self.backbone)

        self.act_model = self._init_act_model()

        if self.use_state == 'VLM':
            self.state_and_mask_dim = 2 * self.configs["state_encoder"]["state_dim"]
            self.vlm_state_encoder = self._init_state_encoder()

        if self.use_fov:
            self.fov_encoder = self._init_fov_encoder()

        # The `cognition_token_id` is set to an unused token ID.
        self.cognition_token_id = self.configs.get("cognition_token_id", 10)
        untied_cognition_token = self.configs.get("untied_cognition_token", True)

        # Use a separately learned `cognition_token_embedding` that does not share parameters with the word embedding matrix
        # if `untied_cognition_token` is True. We initialize this separate `cognition_token_embedding` 
        # with the word embedding parameter corresponding to the specified `cognition_token_id`. 

        if untied_cognition_token:
            print(f"Using separate cognition token")
            print(f"Cognition token id: {self.cognition_token_id}")
            init_id = self.configs.get("cognition_token_init_id", None)
            if init_id is None:
                init_id = self.cognition_token_id
            print(f"Init cognition token with id={init_id}")
            ebd = self.model.get_input_embeddings().weight.data[init_id]
            self.cognition_token = nn.Parameter(ebd.clone())
        else:
            self.cognition_token = None

    def _init_backbone(self):
        processor, model = build_vlm(self.configs["vlm"])
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        return self.tokenizer, model

    def _init_fov_encoder(self):
        from vitra.utils.nn_utils import MLPProjector
        fov_dim = 2 # fov_x, fov_y
        mlp = MLPProjector(fov_dim, self.hidden_size)
        nn.init.normal_(mlp.projector[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[0].bias, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].bias, mean=0.0, std=0.02)
        return mlp

    def _init_state_encoder(self):
        from vitra.utils.nn_utils import MLPProjector
        mlp = MLPProjector(self.state_and_mask_dim, self.hidden_size)
        nn.init.normal_(mlp.projector[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[0].bias, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].bias, mean=0.0, std=0.02)
        return mlp

    def _init_act_model(self):
        from vitra.models.action_model.diffusion_policy import DiffusionPolicy
        action_head = DiffusionPolicy(
            model_type = self.act_model_configs.get("model_type", 'DiT-B'),
            token_size = self.act_model_configs.get("token_size", -1),
            in_channels = self.act_model_configs.get("action_dim", 192),
            future_action_window_size = self.future_action_window_size,
            past_action_window_size = self.past_action_window_size,
            use_state = self.use_state,
            action_type = self.configs['train_dataset'].get("action_type", "angle"),
            state_dim = self.configs["state_encoder"]["state_dim"] if self.use_state=='DiT' else None,
            loss_type = self.configs.get("loss_type", "human"),
        )

        for param in action_head.parameters():
            assert param.dtype == torch.float32, f"Loaded diffusion action model parameter not in full precision: {param}"

        return action_head

    def trainable_params_setup(self):
        model = self.model
        model.config.use_cache = False

        if self.train_setup_configs.get("freeze_option", "full_finetune") == "full_finetune":
            model.requires_grad_(True)
            self.vision_tower.requires_grad_(True)
            self.word_embedding.requires_grad_(True)

        if self.train_setup_configs.get("freeze_option", "only_head_and_token") == "only_head_and_token":
            model.requires_grad_(False)
            self.vision_tower.requires_grad_(False)
            self.word_embedding.requires_grad_(False)

        if self.train_setup_configs.get("freeze_option", "freeze_vision_encoder") == "freeze_vision_encoder":
            model.requires_grad_(True)
            self.vision_tower.requires_grad_(False)
            self.word_embedding.requires_grad_(True)

        if self.act_model is not None:
            self.act_model.requires_grad_(True)

        if self.use_state == 'VLM':
            self.vlm_state_encoder.requires_grad_(True)
        
        if self.use_fov:
            self.fov_encoder.requires_grad_(True)

        if self.cognition_token is not None:
            self.cognition_token.requires_grad_(True)

    @property
    def image_processor(self):
        return self.model.processor

    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size

    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens

    @property
    def text_tower(self):
        return self.model.language_model.model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def model(self):
        return self.backbone

    def _forward_act_model(
        self,
        vlm_features: torch.Tensor,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        mode: str = "train",
        repeated_diffusion_steps: int = 1,
        cfg_scale: float = 5.0,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        **kwargs,
    ):
        
        actions = None
        action_loss = None

        B = vlm_features.shape[0]
        action_features = self.extract_cognition_token(vlm_features, attention_mask) #[B, D]
        model_dtype = next(self.act_model.net.parameters()).dtype
        action_features = action_features.to(model_dtype)
        
        action_features_repeated = action_features.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
        action_masks_repeated = action_masks.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)

        action_features_repeated = action_features_repeated.view(B*repeated_diffusion_steps, 1, action_features.shape[-1])
        action_masks_repeated = action_masks_repeated.view(B*repeated_diffusion_steps, action_masks.shape[1], action_masks.shape[2])

        if self.use_state == 'DiT':
            current_state_repeated = current_state.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_repeated = current_state_repeated.view(B*repeated_diffusion_steps, 1, current_state.shape[1])
            current_state_mask_repeated = current_state_mask.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_mask_repeated = current_state_mask_repeated.view(B*repeated_diffusion_steps, 1, current_state_mask.shape[1])
        else:
            current_state_repeated = None
            current_state_mask_repeated = None

        if mode == "train":
            actions_repeated = action_labels.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
            actions_repeated = actions_repeated.view(B*repeated_diffusion_steps, action_labels.shape[1], action_labels.shape[2])
            if self.use_state == 'DiT':
                action_loss = self.act_model.loss(actions_repeated, action_features_repeated, action_masks_repeated, current_state_repeated, current_state_mask_repeated)
            else:
                action_loss = self.act_model.loss(actions_repeated, action_features_repeated, action_masks_repeated)
            return actions, action_loss
        else:
            # evaluate mode
            # sample multiple action chunks
            actions = self.act_model.sample(
                action_features_repeated,
                cfg_scale,
                current_state_repeated,      #[B*sample_times, 1, D]
                current_state_mask_repeated, #[B*sample_times, 1, D]
                use_ddim,
                num_ddim_steps,
                action_masks_repeated, # ori x_mask
            )

            return actions, action_loss

    def extract_cognition_token(self, output_hs, attention_mask):
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, output_hs.size(-1))
        action_features = output_hs.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]
        return action_features

    def prepare_vlm_input_embeddings(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        B = input_ids.shape[0]
        word_embeds = self.model.get_input_embeddings()(input_ids)
        input_ids_mask = attention_mask 

        cog_ids = torch.ones_like(input_ids[:, 0:1]) * self.cognition_token_id # [B, 1]
        cog_embeds = self.model.get_input_embeddings()(cog_ids)
        cog_ids_mask = torch.ones_like(cog_ids, dtype=torch.bool)

        if hasattr(self, 'cognition_token') and self.cognition_token is not None:
            assert self.cognition_token.shape[0] == cog_embeds.shape[-1], f"cognition token shape {self.cognition_token.shape} does not match cog embeds shape {cog_embeds.shape}"
            cog_embeds = self.cognition_token.unsqueeze(0).unsqueeze(0).expand(B, -1, -1) 

        # Build the list of embeddings and masks to concatenate
        embeds_list = [word_embeds]
        masks_list = [input_ids_mask]
        num_additional_tokens = 0

        if self.use_state == 'VLM':
            current_state = current_state * current_state_mask.to(current_state.dtype)
            state_embeds = self.state_encoder(torch.cat([current_state, current_state_mask.to(current_state.dtype)], dim=1))
            state_ids_mask = torch.ones((B, 1), dtype=torch.bool).to(input_ids_mask.device)
        
            embeds_list.append(state_embeds.unsqueeze(1))
            masks_list.append(state_ids_mask)
            num_additional_tokens += 1

        if self.use_fov:
            fov_embeds = self.fov_encoder(fov)
            fov_ids_mask = torch.ones((B, 1), dtype=torch.bool).to(input_ids_mask.device)

            embeds_list.append(fov_embeds.unsqueeze(1))
            masks_list.append(fov_ids_mask)
            num_additional_tokens += 1

        # Always append cognition token at the end
        embeds_list.append(cog_embeds)
        masks_list.append(cog_ids_mask)
        num_additional_tokens += 1

        # Concatenate all
        inputs_embeds = torch.cat(embeds_list, dim=1)
        inputs_masks = torch.cat(masks_list, dim=1)

        # Note: Here we only use `self.cognition_token_id` as a placeholder for the token corresponding to the FOV or states (if any) input. 
        # In practice, the embedding passed to the LLM will be replaced with the actual FOV or states (if any) embedding.
        additional_tokens = torch.full((B, num_additional_tokens), self.cognition_token_id, dtype=input_ids.dtype, device=input_ids.device)

        inputs_embeds, attention_mask = move_masked_to_left(inputs_embeds, inputs_masks)
        input_ids = torch.cat([input_ids, additional_tokens], dim=1)
        input_ids, inputs_masks = move_masked_to_left_ids(input_ids, inputs_masks)

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed
        # Merge text and images
        if pixel_values is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                image_features = self.model.get_image_features(pixel_values)

            special_image_mask = (input_ids == self.model.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        causal_mask = self.model._update_causal_mask(
            attention_mask, None, None, cache_position, input_ids, inputs_embeds, False
        )
        return {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
        }, inputs_masks

    def prepare_vlm_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):

        vlm_inputs, inputs_masks = self.prepare_vlm_input_embeddings(
                pixel_values,
                input_ids,
                attention_mask,
                current_state_mask,
                current_state,
                fov,
                **kwargs,
            )

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            outputs = self.model.language_model(
                past_key_values=None,
                use_cache=use_cache,
                output_hidden_states=True,
                num_logits_to_keep=0, # can be modified
                **vlm_inputs
            )

        output_hs = outputs.hidden_states[-1]
        return output_hs, inputs_masks

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        mode="train",
        **kwargs,
    ):

        assert mode == "train", f"mode {mode} not supported in the forward function."

        loss = {}

        output_hs, inputs_masks = self.prepare_vlm_features(
            pixel_values,
            input_ids,
            attention_mask,
            current_state_mask,
            current_state,
            fov,
            use_cache,
            **kwargs,
        )

        _, action_loss = self._forward_act_model(
            vlm_features = output_hs, 
            action_labels = action_labels, 
            attention_mask = inputs_masks, 
            action_masks = action_masks, 
            current_state = current_state, 
            current_state_mask = current_state_mask, 
            mode = mode,
            repeated_diffusion_steps = self.repeated_diffusion_steps,
        )

        self._update_loss(loss, action_loss)

        return loss

    def image_preprocess(self, image: Image, size: Tuple[int, int] = (224, 224)) -> Image:
        width, height = image.size

        return image
        

    def predict_action(
        self, 
        image, 
        instruction: str, 
        current_state, 
        current_state_mask, 
        use_ddim=True, 
        num_ddim_steps=10, 
        cfg_scale=5.0, 
        action_mask_torch=None, 
        fov=None, 
        sample_times=1, 
        use_cache=False
    ) -> np.ndarray:
        """
        support B = 1 only for now
        instruction: str
        current_state: normalized current robot action, [B, D]
        current_state_mask: [B, D]
        action_mask_torch: [B, T, D]
        fov: [B, 2]
        return: predicted normalized robot action, [sample_times, T, D]
        """

        B = current_state.shape[0]
        assert B == 1, f"Batch size {B} not supported in predict_action for now."

        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
            elif image.ndim == 4:
                image = [Image.fromarray(im) for im in image]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        prefix = '<image>'
        model_inputs = self.processor(text=prefix + instruction, images=image, return_tensors="pt").to('cuda')
        pixel_value = model_inputs['pixel_values']
        input_ids = model_inputs['input_ids']

        # Preprocess Image
        if isinstance(pixel_value, torch.Tensor):
            pixel_value = pixel_value.to('cuda')
        elif isinstance(pixel_value, dict):
            pixel_value = {
                k: torch.stack([pixel_value[idx][k] for idx in range(len(input_ids))]).to('cuda') for k in pixel_value[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_value)}")
        if pixel_value.dim() == 5:
            pixel_value = pixel_value.view(-1, *pixel_value.shape[2:])

        if action_mask_torch is None:
            x_mask = torch.zeros(B, self.chunk_size, self.act_model.in_channels, device=input_ids.device)
            # x_mask[:, :, :102] = 1.0 # predict dual hand actions
            x_mask[:, :, 51:102] = 1.0 # predict right hand actions
            # x_mask[:, :, 0:51] = 1.0 # predict left hand actions
        else:
            x_mask = action_mask_torch.to(input_ids.device)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)
        current_state_mask = current_state_mask.to(input_ids.device)
        current_state = current_state.to(input_ids.device)
        fov = fov.to(input_ids.device) if fov is not None else None

        output_hs, inputs_masks = self.prepare_vlm_features(
            pixel_value,
            input_ids,
            attention_mask,
            current_state_mask,
            current_state,
            fov,
            use_cache=use_cache,
        )
        # handle multiple samples for one input
        samples, _ = self._forward_act_model(
            vlm_features = output_hs,
            attention_mask = inputs_masks,
            action_masks = x_mask,
            current_state = current_state,
            current_state_mask = current_state_mask,
            mode = "eval",
            repeated_diffusion_steps = sample_times,
            cfg_scale = cfg_scale,
            use_ddim = use_ddim,
            num_ddim_steps = num_ddim_steps,
        )
        action_np = samples.cpu().numpy() * x_mask.cpu().numpy()    # sample_times x T x D
        return action_np

    def _format_loss(self, loss):
        # for visualization and loss backward in pytorch
        _loss = 0
        _keys = list(loss.keys())

        for k in _keys:
            if "loss" in k:
                _loss += loss[k]

        loss["loss"] = _loss
        return loss

    @staticmethod
    def _update_loss(loss, new_loss, suffix=None):
        """
        use new_loss to update loss.
            * if suffix is not None, the key from new_loss will be reformatted as: key|suffix
            * otherwise, if the key from new_loss is not in loss, it will be directly used: key
            * otherwise, the key from the new_loss will be reformatted as: key|index, where index is
                searched from 0->+inf so that key|index is not in loss.

        """

        def get_key(k, d):
            if suffix is not None:
                new_k = f"{k}_{suffix}"
                assert new_k not in d
                return new_k

            ind = 0
            while True:
                if ind == 0:
                    new_k = k
                else:
                    new_k = f"{k}_{ind}"
                if new_k not in d:
                    return new_k
                ind += 1

        for k in new_loss:
            new_k = get_key(k, loss)
            loss[new_k] = new_loss[k]

        return loss
from typing import ClassVar, List, Optional
from math import ceil, sqrt

import torch
from torch import nn
from transformers.models.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from colpali_engine.utils.torch_utils import find_min_max_indices


class PatchMerger(nn.Module):
    def __init__(self, hidden_size: int, dim: int, merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.merge_size = merge_size
        self.ln_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size*merge_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(x.size(0), -1, self.hidden_size*self.merge_size))
        return x


class AvgPoolingMerger(nn.Module):
    def __init__(self, merge_size) -> None:
        super().__init__()
        kernel_size = int(sqrt(merge_size))
        self.conv = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
        self.max_tokens = ceil(780//merge_size)

    def forward(
        self,
        hidden_states,
        attention_mask,
        image_grid_thw,
    ):
        outputs = torch.zeros((hidden_states.size(0), self.max_tokens, hidden_states.size(-1)), device=hidden_states.device, dtype=hidden_states.dtype)
        outputs_attention = torch.zeros((attention_mask.size(0), self.max_tokens), device=attention_mask.device, dtype=attention_mask.dtype)
        
        image_token_nums = (image_grid_thw[:,1]*image_grid_thw[:,2])//4
        for i, (image_token_num, hidden_state) in enumerate(zip(image_token_nums, hidden_states)):
            hidden_state = hidden_state[:image_token_num]
            hidden_state_reshaped = hidden_state.view(image_grid_thw[i, 1]//2, image_grid_thw[i, 2]//2, hidden_state.size(-1)).permute(2, 0, 1)
            hidden_state_reshaped = self.conv(hidden_state_reshaped).view(hidden_state.size(-1), -1).permute(1, 0)
            outputs[i, :hidden_state_reshaped.size(0)] = hidden_state_reshaped
            outputs_attention[i, :hidden_state_reshaped.size(0)] = 1
        return outputs, outputs_attention


class ColQwen2(Qwen2VLForConditionalGeneration):
    """
    ColQwen2 model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self, 
        config: Qwen2VLConfig,
        pooling_strategy: Optional[str]=None,
        pool_size: int=10,
        dim: int=128,
    ):
        super().__init__(config=config)
        self.dim = dim
        self.pool_size = pool_size
        self.pooling_strategy = pooling_strategy
        if self.pooling_strategy=="channelwise-pooling":
            self.custom_image_proj = PatchMerger(self.model.config.hidden_size, self.dim, self.pool_size)
        if self.pooling_strategy=="post_proj_2dpool":
            self.custom_image_proj = AvgPoolingMerger(self.pool_size)
        
        self.custom_text_proj = nn.Linear(self.model.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.post_init()


    def inner_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if pixel_values is not None and self.pooling_strategy=="pre_llm_flatten":
            pad_length = ceil(attention_mask.size(1)/self.pool_size)*self.pool_size-attention_mask.size(1)
            attention_paddings = torch.zeros_like(attention_mask)[:, :pad_length]
            attention_mask = torch.cat([attention_mask, attention_paddings], dim=1)
            attention_mask = attention_mask.view(attention_mask.size(0), attention_mask.size(1)//self.pool_size, self.pool_size)
            attention_mask = torch.any(attention_mask, dim=2).long()

            embeds_paddings = torch.zeros_like(inputs_embeds)[:, :pad_length]
            inputs_embeds = torch.cat([inputs_embeds, embeds_paddings], dim=1)
            inputs_embeds = inputs_embeds.view(inputs_embeds.size(0), inputs_embeds.size(1)//self.pool_size, self.pool_size, inputs_embeds.size(2))
            inputs_embeds = inputs_embeds.mean(dim=2)

            position_ids = None

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if output_attentions:
            hidden_states, self_attns = outputs[0], outputs[-1]
            return hidden_states, self_attns
        else:
            hidden_states = outputs[0]
            return hidden_states


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)


        # The following code is a hack to make sure the scatter in DDP is done correctly when training on multiple GPUs
        if "pixel_values" in kwargs:
            # compute pixel_values offsets
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pv[:o] for pv, o in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        position_ids, rope_deltas = self.get_rope_index(
            input_ids=kwargs["input_ids"],
            image_grid_thw=kwargs.get("image_grid_thw", None),
            video_grid_thw=None,
            attention_mask=kwargs.get("attention_mask", None),
        )
        if kwargs.get("output_attentions", False):
            last_hidden_states, self_attns = self.inner_forward(*args,
                                    **kwargs,
                                    position_ids=position_ids,
                                    use_cache=False,
                                    output_hidden_states=True)  # (batch_size, sequence_length, hidden_size)
        else:
            last_hidden_states = self.inner_forward(*args,
                                    **kwargs,
                                    position_ids=position_ids,
                                    use_cache=False,
                                    output_hidden_states=True)  # (batch_size, sequence_length, hidden_size)

        # We only consider image tokens.
        if "pixel_values" in kwargs:
            if self.pooling_strategy:
                last_hidden_states = self.extract_image_features(last_hidden_states, kwargs)

            if self.pooling_strategy is not None and self.pooling_strategy!="post_proj_2dpool":
                # pad_length = ceil(kwargs["attention_mask"].size(1)/self.pool_size)*self.pool_size-kwargs["attention_mask"].size(1)
                pad_length = 780 - kwargs["attention_mask"].size(1)
                attention_paddings = torch.zeros_like(kwargs["attention_mask"])[:, :pad_length]
                kwargs["attention_mask"] = torch.cat([kwargs["attention_mask"], attention_paddings], dim=1)
                kwargs["attention_mask"] = kwargs["attention_mask"].view(kwargs["attention_mask"].size(0), kwargs["attention_mask"].size(1)//self.pool_size, self.pool_size)
                kwargs["attention_mask"] = torch.any(kwargs["attention_mask"], dim=2).long()
            
            if self.pooling_strategy in ["pre_proj_flatten", "channelwise-pooling"]:
                output_paddings = torch.zeros_like(last_hidden_states)[:, :pad_length]
                last_hidden_states = torch.cat([last_hidden_states, output_paddings], dim=1)
                if self.pooling_strategy=="pre_proj_flatten":
                    last_hidden_states = last_hidden_states.view(last_hidden_states.size(0), last_hidden_states.size(1)//self.pool_size, self.pool_size, last_hidden_states.size(2))
                    last_hidden_states = last_hidden_states.mean(dim=2)
        
        if "pixel_values" in kwargs and self.pooling_strategy=="channelwise-pooling":
            proj = self.custom_image_proj(last_hidden_states)
        else:
            proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs:
            if self.pooling_strategy=="post_proj_flatten":
                output_paddings = torch.zeros_like(proj)[:, :pad_length]
                proj = torch.cat([proj, output_paddings], dim=1)
                proj = proj.view(proj.size(0), proj.size(1)//self.pool_size, self.pool_size, proj.size(2))
                proj = proj.mean(dim=2)
            if self.pooling_strategy=="post_proj_2dpool":
                proj, kwargs["attention_mask"] = self.custom_image_proj(
                    proj,
                    kwargs["attention_mask"],
                    kwargs["image_grid_thw"]
                )
            # Copy from https://github.com/dvlab-research/VisionZip/blob/main/visionzip/clip_encoder.py
            if self.pooling_strategy=="post_proj_cluster":
                target_indices = torch.arange(0, proj.shape[1], self.pool_size, device=proj.device)
                target_tokens = proj[:, target_indices, :]
                tokens_to_merge = proj[:, ~torch.isin(torch.arange(proj.shape[1], device=proj.device), target_indices), :]
                similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
                assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], len(target_indices), dtype=proj.dtype, device=proj.device)
                assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
                counts = assign_one_hot.sum(dim=1).unsqueeze(-1)
                proj = (torch.bmm(assign_one_hot.transpose(1, 2), tokens_to_merge)+target_tokens) / (counts+1)
        
        # L2 normalization
        proj = (proj+1e-10) / (proj.norm(dim=-1, keepdim=True)+1e-10)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        print(proj.size())

        if kwargs.get("output_attentions", False):
            return proj, self_attns
        else:
            return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size

    def extract_image_features(self, last_hidden_states, kwargs):
        min_max_indices = find_min_max_indices(kwargs["input_ids"], self.config.image_token_id)
        max_length = max(min_max_indices[:, 1] - min_max_indices[:, 0])+1

        image_features = torch.zeros((last_hidden_states.size(0), max_length, last_hidden_states.size(2)), device=last_hidden_states.device, dtype=last_hidden_states.dtype)
        attention_mask = torch.zeros((kwargs["attention_mask"].size(0), max_length), device=kwargs["attention_mask"].device, dtype=kwargs["attention_mask"].dtype)
        for i, (min_index, max_index) in enumerate(min_max_indices):
            image_features[i, :(max_index-min_index+1)] = last_hidden_states[i, min_index:(max_index+1)]
            attention_mask[i, :(max_index-min_index+1)] = 1
        kwargs["attention_mask"] = attention_mask

        return image_features
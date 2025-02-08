from typing import ClassVar, List, Optional
from math import ceil, sqrt

import torch
from torch import nn
from transformers.models.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from colpali_engine.utils.torch_utils import find_min_max_indices, pool_embeddings


class AvgPoolingMerger(nn.Module):
    def __init__(self, merge_size) -> None:
        super().__init__()
        self.factor = 28
        self.kernel_size = int(sqrt(merge_size))

    def forward(
        self,
        hidden_states,
        attention_mask,
        image_grid_thw,
        patch_range_list: Optional[torch.Tensor]=None,
        patch_indices_list_list: Optional[torch.Tensor]=None,
        remove_index_list_list: Optional[torch.Tensor]=None,
    ):
        max_tokens = 150
        outputs = torch.zeros((hidden_states.size(0), max_tokens, hidden_states.size(-1)), device=hidden_states.device, dtype=hidden_states.dtype)
        outputs_attention = torch.zeros((attention_mask.size(0), max_tokens), device=attention_mask.device, dtype=attention_mask.dtype)
        
        if remove_index_list_list is None:
            remove_index_list_list = [None for _ in range(hidden_states.size(0))]
        for i, (patch_range, hidden_state, patch_indices_list, remove_index_list) in enumerate(zip(patch_range_list, hidden_states, patch_indices_list_list, remove_index_list_list)):
            start, end = patch_range
            hidden_state = hidden_state[start:(end+1)]

            mask = (patch_indices_list!=-1)
            if remove_index_list is not None:
                remove_index_list = remove_index_list[remove_index_list!=-1]
                mask = mask & ~torch.isin(patch_indices_list, remove_index_list)
            patch_indices_list = patch_indices_list[torch.any(mask, dim=-1)]
            mask = mask[torch.any(mask, dim=-1)]

            hidden_state_reshaped = torch.sum(hidden_state[patch_indices_list], dim=1)/mask.sum(dim=1).unsqueeze(-1)
            print("hidden_state: {}; hidden_state_reshaped: {}; patch_indices_list: {}".format(
                hidden_state.size(), hidden_state_reshaped.size(), patch_indices_list.size())
            )
            # outputs[i, -token_num:] = hidden_state_reshaped
            outputs[i, -hidden_state_reshaped.size(0):] = hidden_state_reshaped
            outputs_attention[i, -hidden_state_reshaped.size(0):] = 1
        return outputs, outputs_attention


class ColQwen2(Qwen2VLForConditionalGeneration):
# class ColQwen2(Qwen2_5_VLForConditionalGeneration):
    """
    ColQwen2 model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self, 
        config: Qwen2VLConfig,
        # config: Qwen2_5_VLConfig,
        pooling_strategy: Optional[str]=None,
        pool_size: int=10,
        dim: int=128,
    ):
        super().__init__(config=config)
        self.dim = dim
        self.pool_size = pool_size
        self.pooling_strategy = pooling_strategy
        if self.pooling_strategy in ["post_proj_2dpool", "pre_llm_2dpool"]:
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
            patch_indices_list: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
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

        if pixel_values is not None and self.pooling_strategy=="pre_llm_2dpool":
            patch_ranges = find_min_max_indices(input_ids, self.config.image_token_id)
            vision_end_token_index = find_min_max_indices(input_ids, self.config.vision_end_token_id)[0, 0]
            eos_token_index = find_min_max_indices(input_ids, self.config.eos_token_id)[0, 0]
            inputs_embeds_tail = inputs_embeds[:, (vision_end_token_index+1):(eos_token_index+1)]
            attention_mask_tail = torch.ones((inputs_embeds_tail.size(0), inputs_embeds_tail.size(1)), device=inputs_embeds_tail.device, dtype=inputs_embeds_tail.dtype)

            inputs_embeds, attention_mask = self.custom_image_proj(
                inputs_embeds,
                attention_mask,
                image_grid_thw,
                patch_range_list=patch_ranges,
                patch_indices_list_list=patch_indices_list,
            )
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_tail], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask_tail], dim=1)
            position_ids = None

        if pixel_values is not None and self.pooling_strategy=="pre_llm_cluster":
            patch_ranges = find_min_max_indices(input_ids, self.config.image_token_id)
            vision_end_token_index = find_min_max_indices(input_ids, self.config.vision_end_token_id)[0, 0]
            eos_token_index = find_min_max_indices(input_ids, self.config.eos_token_id)[0, 0]

            inputs_embeds_tail = inputs_embeds[:, (vision_end_token_index+1):(eos_token_index+1)]
            attention_mask_tail = torch.ones((inputs_embeds_tail.size(0), inputs_embeds_tail.size(1)), device=inputs_embeds_tail.device, dtype=inputs_embeds_tail.dtype)

            max_tokens = ceil(attention_mask.size(1)/self.pool_size)
            new_inputs_embeds = torch.zeros((inputs_embeds.size(0), max_tokens, inputs_embeds.size(-1)), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            attention_mask = torch.zeros((inputs_embeds.size(0), max_tokens), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            for i, (patch_range, p_embed) in enumerate(zip(patch_ranges, inputs_embeds)):
                p_embed = p_embed[patch_range[0]:patch_range[1]+1]
                p_embed, _ = pool_embeddings(p_embed, self.pool_size)
                new_inputs_embeds[i, -p_embed.size(0):] = p_embed
                attention_mask[i, -p_embed.size(0):] = 1
            new_inputs_embeds = torch.cat([new_inputs_embeds, inputs_embeds_tail], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask_tail], dim=1)
            inputs_embeds = new_inputs_embeds
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
            return hidden_states, attention_mask


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        # patch_indices_list_list = kwargs.pop("patch_indices_list", None)
        remove_index_list_list = kwargs.pop("remove_index_list", None)

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
            last_hidden_states, kwargs["attention_mask"] = self.inner_forward(*args,
                                    **kwargs,
                                    position_ids=position_ids,
                                    use_cache=False,
                                    output_hidden_states=True)  # (batch_size, sequence_length, hidden_size)

        # We only consider image tokens.
        if "pixel_values" in kwargs:
            patch_ranges = find_min_max_indices(kwargs["input_ids"], self.config.image_token_id)
            vision_end_token_index = find_min_max_indices(kwargs["input_ids"], self.config.vision_end_token_id)[0, 0]
            eos_token_index = find_min_max_indices(kwargs["input_ids"], self.config.eos_token_id)[0, 0]

            if self.pooling_strategy is not None and self.pooling_strategy not in ["post_proj_2dpool", "pre_llm_cluster", "pre_llm_2dpool"]:
                pad_length = ceil((vision_end_token_index+1)/self.pool_size)*self.pool_size-(vision_end_token_index+1)
                zero_paddings = torch.zeros_like(kwargs["attention_mask"])[:, :pad_length]
                one_paddings = torch.ones_like(kwargs["attention_mask"])[:, vision_end_token_index:eos_token_index]
                kwargs["attention_mask"] = torch.cat([zero_paddings, kwargs["attention_mask"][:, :(vision_end_token_index+1)]], dim=1)
                kwargs["attention_mask"] = kwargs["attention_mask"].view(kwargs["attention_mask"].size(0), kwargs["attention_mask"].size(1)//self.pool_size, self.pool_size)
                kwargs["attention_mask"] = torch.any(kwargs["attention_mask"], dim=2).long()
                kwargs["attention_mask"] = torch.cat([kwargs["attention_mask"], one_paddings], dim=1)
            
            if self.pooling_strategy in ["pre_proj_flatten"]:
                output_paddings = torch.zeros_like(last_hidden_states)[:, :pad_length]
                last_hidden_states = torch.cat([last_hidden_states, output_paddings], dim=1)
                if self.pooling_strategy=="pre_proj_flatten":
                    last_hidden_states = last_hidden_states.view(last_hidden_states.size(0), last_hidden_states.size(1)//self.pool_size, self.pool_size, last_hidden_states.size(2))
                    last_hidden_states = last_hidden_states.mean(dim=2)
        
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs:
            if self.pooling_strategy=="post_proj_selected":
                from torch.nn.utils.rnn import pad_sequence
                indices_list = list()
                for patch_range in find_min_max_indices(kwargs["input_ids"], self.config.image_token_id):
                    k = min(ceil((vision_end_token_index+1)/self.pool_size), patch_range[1]+1-patch_range[0])
                    indices = torch.randperm(patch_range[1]-patch_range[0]+1, device=weight.device)[:k] + patch_range[0]
                    indices_list.append(indices)
                indices_list = pad_sequence(indices_list, batch_first=True, padding_value=0).to(weights.device)
                indices_list = torch.sort(indices_list, dim=1).values
                proj_tail = proj[:, (vision_end_token_index+1):(eos_token_index+1)]
                proj = proj[torch.arange(proj.size(0), device=proj.device).unsqueeze(-1), indices_list]
                proj = torch.cat([proj, proj_tail], dim=1)
                # adapt attention_mask to the size of proj
                kwargs["attention_mask"] = torch.cat([(indices_list!=0).long(), torch.ones_like(proj_tail)[..., -0]], dim=1)

            if self.pooling_strategy=="post_proj_flatten":
                output_paddings = torch.zeros_like(proj)[:, :pad_length]
                proj_tail = proj[:, (vision_end_token_index+1):(eos_token_index+1)]
                proj = torch.cat([output_paddings.clone(), proj[:, :vision_end_token_index+1]], dim=1)
                proj = proj.view(proj.size(0), proj.size(1)//self.pool_size, self.pool_size, proj.size(2))

                proj = proj.view(proj.size(0), proj.size(1)//self.pool_size, self.pool_size, proj.size(2)).mean(dim=2)
                proj = torch.cat([proj, proj_tail], dim=1)

            if self.pooling_strategy=="post_proj_2dpool":
                proj_tail = proj[:, (vision_end_token_index+1):(eos_token_index+1)]
                attention_mask_tail = kwargs["attention_mask"][:, (vision_end_token_index+1):(eos_token_index+1)].clone()
                proj, kwargs["attention_mask"] = self.custom_image_proj(
                    proj,
                    kwargs["attention_mask"],
                    kwargs["image_grid_thw"],
                    patch_range_list=patch_ranges,
                    patch_indices_list_list=kwargs["patch_indices_list"],
                    remove_index_list_list=remove_index_list_list
                )
                proj = torch.cat([proj, proj_tail], dim=1)
                kwargs["attention_mask"] = torch.cat([kwargs["attention_mask"], attention_mask_tail], dim=1)
            
            if self.pooling_strategy=="post_proj_cluster":
                proj_tail = proj[:, (vision_end_token_index+1):(eos_token_index+1)]
                attention_mask_tail = torch.ones((proj_tail.size(0), proj_tail.size(1)), device=proj_tail.device, dtype=proj_tail.dtype)

                max_tokens = ceil(proj.size(1)/self.pool_size)+10
                new_proj = torch.zeros((proj.size(0), max_tokens, proj.size(-1)), device=proj.device, dtype=proj.dtype)
                attention_mask = torch.zeros((proj.size(0), max_tokens), device=proj.device, dtype=proj.dtype)
                for i, (p_embed, patch_range) in enumerate(zip(proj, patch_ranges)):
                    p_embed = p_embed[patch_range[0]:patch_range[1]+1]
                    p_embed, _ = pool_embeddings(p_embed, self.pool_size)
                    new_proj[i, -p_embed.size(0):] = p_embed
                    attention_mask[i, -p_embed.size(0):] = 1
                new_proj = torch.cat([new_proj, proj_tail], dim=1)
                kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask_tail], dim=1)
                proj = new_proj
        
        # L2 normalization
        proj = (proj+1e-10) / (proj.norm(dim=-1, keepdim=True)+1e-10)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        print(proj.size(), proj.device)

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
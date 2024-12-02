from typing import ClassVar, List, Optional
from math import ceil, sqrt

import torch
from torch import nn
from transformers.models.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration


class ColQwen2(Qwen2VLForConditionalGeneration):
    """
    ColQwen2 model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config=config)
        self.dim = 128
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
        last_hidden_states = self.inner_forward(*args,
                                  **kwargs,
                                  position_ids=position_ids,
                                  use_cache=False,
                                  output_hidden_states=True)  # (batch_size, sequence_length, hidden_size)

        if "pixel_values" in kwargs and self.pooling_strategy is not None:
            pad_length = ceil(kwargs["attention_mask"].size(1)/self.pool_size)*self.pool_size-kwargs["attention_mask"].size(1)
            attention_paddings = torch.zeros_like(kwargs["attention_mask"])[:, :pad_length]
            kwargs["attention_mask"] = torch.cat([kwargs["attention_mask"], attention_paddings], dim=1)
            kwargs["attention_mask"] = kwargs["attention_mask"].view(kwargs["attention_mask"].size(0), kwargs["attention_mask"].size(1)//self.pool_size, self.pool_size)
            kwargs["attention_mask"] = torch.any(kwargs["attention_mask"], dim=2).long()
        
        if "pixel_values" in kwargs and self.pooling_strategy=="pre_proj_flatten":
            output_paddings = torch.zeros_like(last_hidden_states)[:, :pad_length]
            last_hidden_states = torch.cat([last_hidden_states, output_paddings], dim=1)
            last_hidden_states = last_hidden_states.view(last_hidden_states.size(0), last_hidden_states.size(1)//self.pool_size, self.pool_size, last_hidden_states.size(2))
            last_hidden_states = last_hidden_states.mean(dim=2)
        
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)
        if "pixel_values" in kwargs and self.pooling_strategy=="post_proj_flatten":
            output_paddings = torch.zeros_like(proj)[:, :pad_length]
            proj = torch.cat([proj, output_paddings], dim=1)
            proj = proj.view(proj.size(0), proj.size(1)//self.pool_size, self.pool_size, proj.size(2))
            proj = proj.mean(dim=2)
        
        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        print(proj.size())
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
    

    # def get_patch_ids(self, h_len, w_len):
    #     self.pool_size = 9
    #     pool_length = int(sqrt(self.pool_size))
    #     w_len = int(w_len/self.spatial_merge_size)
    #     h_len = int(h_len/self.spatial_merge_size)

    #     g_w = ceil(w_len/pool_length)
    #     g_h = ceil(h_len/pool_length)

    #     patch_ids_list = list()
    #     for i in range(g_w*g_h):
    #         g_x, g_y = i%g_w, i//g_w
    #         patch_ids = list()
    #         for y in range(pool_length*g_y, min(pool_length*(g_y+1), h_len)):
    #             for x in range(pool_length*g_x, min(pool_length*(g_x+1), w_len)):
    #                 patch_ids.append(x+y*w_len)
    #         patch_ids_list.append(
    #             torch.tensor(patch_ids)
    #         )
    #     return nn.utils.rnn.pad_sequence(patch_ids_list, batch_first=True, padding_value=-1)

import os
import json
import torch
import asyncio
import openai
from openai import AsyncOpenAI

from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
from draw_and_locate_grid.utils import draw_grid, encode_image_to_base64


MIN_TOKENS = 256
MAX_TOKENS = 1280
FACTOR = 28


def load_model_and_processor(model_name_or_path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    min_pixels = MIN_TOKENS*FACTOR*FACTOR
    max_pixels = MAX_TOKENS*FACTOR*FACTOR
    processor = AutoProcessor.from_pretrained(model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor


def generate_vlm(model, processor, prompt_messages, image_path, question):
    messages = prompt_messages + [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": "Question: {}\nGrid Numbers:".format(question)
                }
            ]
        }
    ]


    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text[0]


async def generate_api_batch(prompt_messages, image_path_list, question_list):
    from copy import deepcopy
    messages_list = [
        deepcopy(prompt_messages) + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{}".format(encode_image_to_base64(image_path))},
                    },
                    {
                        "type": "text",
                        "text": "Question: {}\nGrid Numbers:".format(question)
                    }
                ]
            }
        ] for image_path, question in zip(image_path_list, question_list)
    ]
    async_responses = [
        client.chat.completions.create(
            model=model_name_or_path,
            messages=messages,
            max_tokens=128,
            temperature=0.0
        ) for messages in messages_list
    ]
    return await asyncio.gather(*async_responses, return_exceptions=True)


if __name__ == "__main__":
    model_name_or_path = "gpt-4o"
    # model_name_or_path = "/mnt/hwfile/mllm/huangqidong/lvlm/Qwen2-VL-72B-Instruct"
    grid_image_dir = "./grid_image"
    batch_size = 4

    if "gpt-4" in model_name_or_path:
        client = AsyncOpenAI(api_key="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")
        prompt_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Given an image, a question about this image. Please point out the numbers of all grids that are relevant to the question, such as the grids including the keywords in the question or the answer. The number of pointed grids should be moderate, not too many or too few. 1-5 grids are recommended."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{}".format(encode_image_to_base64("./exp_grid_1.png"))},
                    },
                    {
                        "type": "text",
                        "text": "Question: What percent of Pinterest users are women?Grid Number:\n"
                    },
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "[31, 33, 34, 37]"
                    },
                ],
            }
        ]
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            min_pixels=MIN_TOKENS*FACTOR*FACTOR,
            max_pixels=MAX_TOKENS*FACTOR*FACTOR
        )
        prompt_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Given an image, a question about this image. Please point out the numbers of all grids that are relevant to the question, such as the grids including the keywords in the question or the answer. The number of pointed grids should be moderate, not too many or too few. 1-5 grids are recommended. And your answer should be simply a list."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "./exp_grid_1.png",
                    },
                    {
                        "type": "text",
                        "text": "Question: What percent of Pinterest users are women?Grid Number:\n"
                    },
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "[31, 33, 34, 37]"
                    },
                ],
            }
        ]
    
    dataset = load_dataset("/mnt/petrelfs/mayubo/colpali/data_dir/colpali_train_set")['train']
    dataset = dataset.shuffle(seed=42).select(range(1000))

    if not os.path.exists(grid_image_dir):
        os.makedirs(grid_image_dir)

    res_list = list()
    for curr in tqdm(range(0, len(dataset), batch_size)):
        batch_data = dataset[curr:curr+batch_size]
        question_list = [query.split('\n')[0] for query in batch_data['query']]
        answer_list = batch_data['answer']

        new_hw_list = list()
        grid_image_path_list = list()
        for i, (image, source) in enumerate(zip(batch_data['image'], batch_data['source'])):
            image.save("./temp_{}.png".format(i))
            grid_image_path = os.path.join(grid_image_dir, "./{}_{}.png".format(source, curr+i))
            grid_image_path_list.append(grid_image_path)
            new_height, new_width = draw_grid("./temp_{}.png".format(i), 4*FACTOR, grid_image_path)
            new_hw_list.append((new_height, new_width))
        
        if "gpt-4" in model_name_or_path:
            response_list = asyncio.run(
                generate_api_batch(prompt_messages, grid_image_path_list, question_list)
            )
            output_text_list = [
                response.choices[0].message.content 
                    if (isinstance(response, openai.types.chat.ChatCompletion) and hasattr(response.choices[0].message, "content")) else "OpenAI Output Error" for response in response_list
            ]
        else:
            pass
            # output_text = generate_vlm(model, processor, prompt_messages, grid_image_path, question)
    
        res_list.extend(
            [
                {
                    "image_path": grid_image_path,
                    "h_w": h_w,
                    "question": question,
                    "answer": answer,
                    "grid_num": output_text,
                } for (grid_image_path, h_w, question, answer, output_text) in zip(grid_image_path_list, new_hw_list, question_list, answer_list, output_text_list)
            ]
        )

        with open("./res_list_{}.json".format(model_name_or_path.split("/")[-1]), 'w') as f:
            json.dump(res_list, f)

    
    # for i, data in enumerate(tqdm(dataset)):
    #     question = data['query'].split('\n')[0]
    #     data['image'].save("./temp.png")
    #     grid_image_path = os.path.join(grid_image_dir, "./{}_{}.png".format(data['source'], i))
    #     new_height, new_width = draw_grid("./temp.png", 4*FACTOR, grid_image_path)

    #     if "gpt-4" in model_name_or_path:
    #         output_text = generate_api(prompt_messages, grid_image_path, question)
    #     else:
    #         output_text = generate_vlm(model, processor, prompt_messages, grid_image_path, question)
    #     res_list.append(
    #         {
    #             "image_path": grid_image_path,
    #             "h_w": (new_height, new_width),
    #             "question": question,
    #             "answer": data['answer'],
    #             "grid_num": output_text,
    #         }
    #     )

    #     with open("./res_list_{}.json".format(model_name_or_path.split("/")[-1]), 'w') as f:
    #         json.dump(res_list, f)
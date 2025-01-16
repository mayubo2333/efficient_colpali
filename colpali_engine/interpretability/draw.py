import os
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
import json
import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
    plot_one_similarity_map
)
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device


if __name__=="__main__":
    torch.manual_seed(42)

    model_name = "/mnt/hwfile/mllm/mayubo/Qwen2-VL-2B-Instruct"
    peft_name = "/mnt/petrelfs/mayubo/colpali/models/colqwen2-ba256-ckpt-5e-nopad_1207/checkpoint-2310"
    dataset_name = "/mnt/petrelfs/mayubo/colpali/data_dir/docvqa_test_subsampled"

    if not os.path.exists(
        os.path.join("/mnt/petrelfs/mayubo/colpali/colpali_engine/interpretability", dataset_name.split("/")[-1])
    ):
        os.makedirs(os.path.join("/mnt/petrelfs/mayubo/colpali/colpali_engine/interpretability", dataset_name.split("/")[-1]))

    device = get_torch_device("auto")
    # Load the model
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    model.load_adapter(peft_name)

    if "custom_text_proj.pt" in os.listdir(peft_name):
        proj_state_dict = torch.load(os.path.join(peft_name, "custom_text_proj.pt"))
        model.custom_text_proj.base_layer.weight = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.base_layer.weight'])
        model.custom_text_proj.base_layer.bias = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.base_layer.bias'])

    model.pooling_strategy = None

    # Load the processor
    processor = ColQwen2Processor.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)['test']
    
    query_list = list()
    for data in tqdm(dataset):
        # Load the image and query
        image, query = data["image"], data["query"]
        query_id = data["questionId"] if "questionId" in data else data['image_filename'].split('/')[-1].split(".")[0]
        query_list.append([query_id, query])

        with open(os.path.join("/mnt/petrelfs/mayubo/colpali/colpali_engine/interpretability", dataset_name.split("/")[-1], "query.json"), 'w') as f:
            json.dump(query_list, f)

        # Preprocess inputs
        batch_images = processor.process_images([image]).to(device)
        batch_queries = processor.process_queries([query]).to(device)

        # Forward passes
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            query_embeddings = model.forward(**batch_queries)

        # Get the number of image patches
        n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size, spatial_merge_size=2)

        # Get the tensor mask to filter out the embeddings that are not related to the image
        image_mask = processor.get_image_mask(batch_images)

        # Generate the similarity maps
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        # Get the similarity map for our (only) input image
        similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)

        # Tokenize the query
        query_tokens = processor.tokenizer.tokenize(query)

        # # Plot and save the similarity maps for each query token
        # plots = plot_all_similarity_maps(
        #     image=image,
        #     query_tokens=query_tokens,
        #     similarity_maps=similarity_maps,
        #     show_colorbar=True
        # )
        # for idx, (fig, ax) in enumerate(plots):
        #     fig.savefig(f"similarity_map_{idx}.png")

        # Plot and save the similarity maps for each query token
        fig, ax = plot_one_similarity_map(
            image=image,
            similarity_maps=similarity_maps,
            figsize=(8, 10)
        )
        fig.savefig(
            os.path.join("/mnt/petrelfs/mayubo/colpali/colpali_engine/interpretability", dataset_name.split("/")[-1], "{}.png".format(query_id))
        )
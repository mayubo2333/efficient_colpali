import os
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
import json
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

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
    peft_name = "/mnt/petrelfs/mayubo/colpali/models/colqwen2-ba256-ckpt-5e-nopad_1207"
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
    model.pooling_strategy = None
    batch_size = 8

    # Load the processor
    processor = ColQwen2Processor.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)['test']

    image_embeddings = list()
    query_embeddings = list()
    # for curr in tqdm(range(0, len(dataset), batch_size)):
    for curr in tqdm(range(0, 100, batch_size)):
        # Load the image and query
        images = dataset[curr:(curr+batch_size)]['image']
        queries = dataset[curr:(curr+batch_size)]['query']
        # Preprocess inputs
        batch_images = processor.process_images(images).to(device)
        batch_queries = processor.process_queries(queries).to(device)

        # Forward passes
        with torch.no_grad():
            batch_image_embeddings = model.forward(**batch_images)
            batch_query_embeddings = model.forward(**batch_queries)
        image_embeddings.append(batch_image_embeddings)
        query_embeddings.append(batch_query_embeddings)
    

    counter_list = list()
    for batch_image_embeddings in tqdm(image_embeddings):
        similarity_matrix_list = list()
        for batch_query_embeddings in query_embeddings:
            similarity_matrix = torch.einsum(
                "bnl,sml->bsnm", batch_image_embeddings, batch_query_embeddings
            ).sum(dim=3)
            similarity_matrix_list.append(similarity_matrix)

        for i in range(batch_image_embeddings.size(0)):
            rank_batch = torch.cat([similarity_matrix[i].topk(dim=1, k=10).indices for similarity_matrix in similarity_matrix_list], dim=0)
            counter_list.append(
                Counter(rank_batch.view(-1).cpu().tolist()).most_common()
            )
    import ipdb; ipdb.set_trace()
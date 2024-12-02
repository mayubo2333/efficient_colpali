import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")

from pathlib import Path

import configue
import typer

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.gpu_stats import print_gpu_utilization


def main(config_file: Path) -> None:
    print_gpu_utilization()
    print("Loading config")
    config = configue.load(config_file, sub_path="config")
    print("Creating Setup")
    if isinstance(config, ColModelTrainingConfig):
        app = ColModelTraining(config)
    else:
        raise ValueError("Config must be of type ColModelTrainingConfig")

    if config.run_train:
        print("Training model")
        if not app.config.proj_lora:
            app.model.custom_text_proj.weight.requires_grad = True
            app.model.custom_text_proj.bias.requires_grad = True
        app.train()
        app.save(config_file=config_file)
    if config.run_eval:
        if app.model.device.type=='cpu':
            print("attach model to cuda")
            app.model.to("cuda")
        print("Running evaluation")
        app.eval()
    print("Done!")


if __name__ == "__main__":
    # config_file = "./scripts/configs/qwen2/inference_post_proj_flatten.yaml"
    # config_file = "./scripts/configs/qwen2/inference_pre_proj_flatten.yaml"
    config_file = "./scripts/configs/qwen2/train_colqwen2_model_proj-nolora_post_proj_flatten.yaml"
    main(config_file)
    # typer.run(main)

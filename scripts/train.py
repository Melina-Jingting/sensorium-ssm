import time
import copy
import json
import argparse
from pathlib import Path
from pprint import pprint
from importlib.machinery import SourceFileLoader
import importlib

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from argus.utils import deep_to
import wandb
import numpy as np

from src.datasets import TrainMouseVideoDataset, ValMouseVideoDataset, ConcatMiceVideoDataset
from src.utils import get_lr, init_weights, get_best_model_path, save_model_to_wandb
from src.responses import get_responses_processor
from src.inputs import get_inputs_processor
from src.metrics import CorrelationMetric
from src.indexes import IndexesGenerator
from src.models.dwiseneurossm import DwiseNeuroSSM
from src.data import get_mouse_data
from src.mixers import CutMix
from src import constants
from src.losses import MicePoissonLoss



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str)
    parser.add_argument("-f", "--folds", default="0", type=str)
    return parser.parse_args()

def train_mouse(config: dict, save_dir: Path, train_splits: list[str], val_splits: list[str]):
    config = copy.deepcopy(config)
    
    # Split module and class name
    module_name, class_name = config["model_class"].rsplit(".", 1)

    # Dynamically import module and get model class
    model_module = importlib.import_module(module_name)
    model_class = getattr(model_module, class_name)
    
    model = model_class(**config["nn_module"][1])
    model.to(config["device"])
    
    
    # Dataset processing
    indexes_generator = IndexesGenerator(**config["frame_stack"])
    inputs_processor = get_inputs_processor(*config["inputs_processor"])
    responses_processor = get_responses_processor(*config["responses_processor"])
    cutmix = CutMix(**config["cutmix"])

    # Build training dataset
    train_datasets = []
    mouse_epoch_size = config["train_epoch_size"] // constants.num_mice
    for mouse in constants.mice:
        train_datasets.append(
            TrainMouseVideoDataset(
                mouse_data=get_mouse_data(mouse=mouse, splits=train_splits),
                indexes_generator=indexes_generator,
                inputs_processor=inputs_processor,
                responses_processor=responses_processor,
                epoch_size=mouse_epoch_size,
                mixer=cutmix,
            )
        )
    train_dataset = ConcatMiceVideoDataset(train_datasets)

    # Build validation dataset
    val_datasets = []
    for mouse in constants.mice:
        val_datasets.append(
            ValMouseVideoDataset(
                mouse_data=get_mouse_data(mouse=mouse, splits=val_splits),
                indexes_generator=indexes_generator,
                inputs_processor=inputs_processor,
                responses_processor=responses_processor,
            )
        )
    val_dataset = ConcatMiceVideoDataset(val_datasets)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_dataloader_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] // config["iter_size"],
        shuffle=False,
        num_workers=config["num_dataloader_workers"],
    )
    
    # Optimizer, scheduler, and loss
    optimizer = optim.Adam(model.parameters(), lr=config["base_lr"])
    total_iterations = len(train_loader) * sum(config["num_epochs"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iterations, eta_min=get_lr(config["min_base_lr"], config["batch_size"])
    )
    loss_fn = MicePoissonLoss()  # Replace with the correct loss function if different
    correlation_metric = CorrelationMetric()

    # Training loop
    num_total_epochs = sum(config["num_epochs"])
    global_step = 0
    iter_size = config.get("iter_size", 1)  # Gradient accumulation
    grad_scaler = torch.amp.GradScaler("cuda",enabled=True)  # Mixed precision
    
    wandb.init(project="sensorium_ssm", config=config)

    for num_epochs, stage in zip(config["num_epochs"], config["stages"]):
        
        num_iterations = (len(train_dataset) // config["batch_size"]) * num_epochs
        if stage == "warmup":
            scheduler = LambdaLR(optimizer, lr_lambda=lambda x: x / num_iterations)
        elif stage == "train":
            scheduler = CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=get_lr(config["min_base_lr"], config["batch_size"]))

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                inputs, target = deep_to(batch, device=config["device"], non_blocking=True)

                with torch.amp.autocast('cuda', enabled=True):                
                    prediction = model(inputs)
                    loss = loss_fn(prediction, target) / iter_size  # Scale loss for accumulation

                grad_scaler.scale(loss).backward()
                epoch_loss += loss.item() * iter_size

                if (i + 1) % iter_size == 0:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    wandb.log({"train_loss": loss.item() * iter_size, "lr": optimizer.param_groups[0]["lr"], "epoch": epoch + 1, "global_step": global_step})

            # Validation step
            model.eval()
            val_loss = 0.0
            correlation_metric.reset()
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs, target = batch
                    inputs, target = deep_to(batch, config["device"], non_blocking=True)
                    prediction = model(inputs)
                    loss = loss_fn(prediction, target)
                    val_loss += loss.item()
                    correlation_metric.update({"prediction": prediction, "target": target})
            
            val_loss /= len(val_loader)
            val_corr = correlation_metric.compute()
            avg_corr = np.mean(list(val_corr.values()))  # Get overall mean correlation
            
            print(f"Epoch {epoch+1}/{num_total_epochs} - Train Loss: {epoch_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f} - Corr: {avg_corr:.4f}")
            epoch_metrics = {
                "epoch_train_loss": epoch_loss / len(train_loader),
                "epoch_val_loss": val_loss,
                "epoch_correlation": avg_corr,
                "epoch": epoch + 1
            }
            for mouse_index, mouse_corr in val_corr.items():
                epoch_metrics[f"val_corr_mouse_{mouse_index}"] = mouse_corr
            
            wandb.log(epoch_metrics)
            save_model_to_wandb(model, epoch, save_dir)



if __name__ == "__main__":
    args = parse_arguments()
    print("Experiment:", args.experiment)

    config_path = constants.configs_dir / f"{args.experiment}.py"
    if not config_path.exists():
        raise RuntimeError(f"Config '{config_path}' is not exists")

    train_config = SourceFileLoader(args.experiment, str(config_path)).load_module().config
    print("Experiment config:")
    pprint(train_config, sort_dicts=False)

    experiment_dir = constants.experiments_dir / args.experiment
    print("Experiment dir:", experiment_dir)
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder '{experiment_dir}' already exists.")

    with open(experiment_dir / "train.py", "w") as outfile:
        outfile.write(open(__file__).read())

    with open(experiment_dir / "config.json", "w") as outfile:
        json.dump(train_config, outfile, indent=4)

    if args.folds == "all":
        folds_splits = constants.folds_splits
    else:
        folds_splits = [f"fold_{fold}" for fold in args.folds.split(",")]

    for fold_split in folds_splits:
        fold_experiment_dir = experiment_dir / fold_split

        val_folds_splits = [fold_split]
        train_folds_splits = sorted(set(constants.folds_splits) - set(val_folds_splits))

        print(f"Val fold: {val_folds_splits}, train folds: {train_folds_splits}")
        print(f"Fold experiment dir: {fold_experiment_dir}")
        train_mouse(train_config, fold_experiment_dir, train_folds_splits, val_folds_splits)

        torch.cuda.empty_cache()
        time.sleep(12)

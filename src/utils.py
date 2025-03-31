import re
import math
import time
import random
from pathlib import Path

import numpy as np
from torch import nn
import wandb
import os
import torch


def set_random_seed(index: int):
    seed = int(time.time() * 1000.0) + index
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


def get_lr(base_lr: float, batch_size: int, base_batch_size: int = 4) -> float:
    return base_lr * (batch_size / base_batch_size)


def get_best_model_path(dir_path, return_score=False, more_better=True):
    dir_path = Path(dir_path)
    model_scores = []
    for model_path in dir_path.glob('*.pth'):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', str(model_path))
        if score is not None:
            score = float(score.group(0)[1:-4])
            model_scores.append((model_path, score))

    if not model_scores:
        if return_score:
            return None, -np.inf if more_better else np.inf
        else:
            return None

    model_score = sorted(model_scores, key=lambda x: x[1], reverse=more_better)
    best_model_path = model_score[0][0]
    if return_score:
        best_score = model_score[0][1]
        return best_model_path, best_score
    else:
        return best_model_path


def init_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            fan_out = math.prod(m.kernel_size) * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            fan_out = m.weight.size(0)
            fan_in = 0
            init_range = 1.0 / math.sqrt(fan_in + fan_out)
            nn.init.uniform_(m.weight, -init_range, init_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def get_length_without_nan(array: np.ndarray):
    nan_indexes = np.argwhere(np.isnan(array)).ravel()
    if nan_indexes.shape[0]:
        return nan_indexes[0]
    else:
        return array.shape[0]

def save_model_to_wandb(model, epoch, save_dir):
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        
    artifact = wandb.Artifact("trained_model", type="model")
    model_path = os.path.join(save_dir, f"model_epoch_{epoch+1:03d}.pth")
    
    torch.save(model.state_dict(), model_path)  # Save locally
    artifact.add_file(model_path)  # Attach to artifact
    
    wandb.log_artifact(artifact)  # Log artifact to wandb
    
    old_model_path = os.path.join(save_dir, f"model_epoch_{epoch:03d}.pth")
    if os.path.exists(old_model_path):
        os.remove(old_model_path)
    
    
def count_trainable_layers(model):
    trainable_layers = set()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_name = ".".join(name.split(".")[:-1])  # Get layer name without weight/bias
            trainable_layers.add(layer_name)

    return len(trainable_layers)
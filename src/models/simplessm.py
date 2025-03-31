import math
import functools
from typing import Callable, List

import torch
from torch import nn
from mamba_ssm import Mamba


class SimpleSSM(nn.Module):
    def __init__(self,
                 frame_height: int,
                 frame_width: int,
                 behavior_channels: int,
                 pupil_channels: int,
                 mlp_dim: int,
                 mamba_dim: int,
                 readout_outputs: List[int],  # List of output dimensions for each readout
                 mamba_state_dim: int = 16,
                 mamba_conv_dim: int = 4,
                 mamba_expand: int = 2,
                 softplus_beta: float = 1.0,
                 drop_rate: float = 0.0):
        super().__init__()
        self.frame_flatten = nn.Flatten(start_dim=2)  # Flatten time, height, width
        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_height * frame_width, mlp_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp_dim)
        )
        self.behavior_mlp = nn.Sequential(
            nn.Linear(behavior_channels, mlp_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp_dim)
        )
        self.pupil_mlp = nn.Sequential(
            nn.Linear(pupil_channels, mlp_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp_dim)
        )
        self.mamba = Mamba(
            d_model=mamba_dim,
            d_state=mamba_state_dim,
            d_conv=mamba_conv_dim,
            expand=mamba_expand
        )
        self.projection = nn.Linear(mlp_dim, mamba_dim) # Project MLP output to Mamba dimension

        self.readouts = nn.ModuleList()
        for readout_output in readout_outputs:
            self.readouts.append(
                Readout(
                    in_features=mamba_dim,  # Input to readout is the Mamba output
                    out_features=readout_output,
                    softplus_beta=softplus_beta,
                    drop_rate=drop_rate,
                )
            )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], index: int | None = None) -> List[torch.Tensor] | torch.Tensor:
        # frames: (batch, time, height, width)
        # behavior: (batch, channels, time) -> (batch, time, channels)
        # pupil_center: (batch, channels, time) -> (batch, time, channels)
        frames, behavior, pupil_center = inputs
        batch_size, time_steps, height, width = frames.shape
        behavior = behavior.transpose(1, 2)
        pupil_center = pupil_center.transpose(1, 2)

        flattened_frames = self.frame_flatten(frames).transpose(1, 2) # (batch, time, height * width)
        frame_features = self.frame_mlp(flattened_frames) # (batch, time, mlp_dim)
        behavior_features = self.behavior_mlp(behavior)     # (batch, time, mlp_dim)
        pupil_features = self.pupil_mlp(pupil_center)   # (batch, time, mlp_dim)

        # Sum the MLP outputs
        combined_features = frame_features + behavior_features + pupil_features # (batch, time, mlp_dim)

        # Project to Mamba dimension
        mamba_input = self.projection(combined_features) # (batch, time, mamba_dim)

        # Run through Mamba
        mamba_output = self.mamba(mamba_input) # (batch, time, mamba_dim)

        if index is None:
            return [readout(mamba_output) for readout in self.readouts]
        else:
            return self.readouts[index](mamba_output)

class Readout(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 softplus_beta: float = 1.0,
                 drop_rate: float = 0.):
        super().__init__()
        self.out_features = out_features
        self.layer = nn.Sequential(
            nn.Dropout1d(p=drop_rate),
            nn.Linear(in_features, out_features),
        )
        self.gate = nn.Softplus(beta=softplus_beta)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, time, in_features)
        x = self.layer(x)
        x = self.gate(x)
        return x

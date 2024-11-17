import json
from pathlib import Path

from torch import nn
import torch
import numpy as np

from utils.download import download, default_checkpoints


NORMS = {
    'layer': nn.LayerNorm,
    'batch': nn.BatchNorm1d,
    'none': nn.Identity
}

ACT = {
    'gelu': nn.GELU(),
    'relu': nn.ReLU()
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StandardMLP(nn.Module):
    def __init__(self, dim_in, dim_out, widths, norm='layer', act='relu'):
        super(StandardMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.norm = NORMS[norm]
        self.act = ACT[act]
        self.widths = widths
        self.linear_in = nn.Linear(self.dim_in, self.widths[0])
        self.linear_out = nn.Linear(self.widths[-1], self.dim_out)
        self.layers = []
        self.layer_norms = []
        for i in range(len(self.widths) - 1):
            self.layers.append(nn.Linear(self.widths[i], self.widths[i + 1]))
            self.layer_norms.append(self.norm(widths[i + 1]))

        self.layers = nn.ModuleList(self.layers)
        self.layernorms = nn.ModuleList(self.layer_norms)

    def forward(self, x):
        z = self.linear_in(x)
        for layer, norm in zip(self.layers, self.layer_norms):
            z = norm(z)
            z = self.act(z)
            z = layer(z)

        out = self.linear_out(z)

        return out


class BottleneckMLP(nn.Module):
    def __init__(
        self, dim_in, dim_out, block_dims, norm='layer', 
        checkpoint=None, name=None, checkpoint_path='./checkpoints/'
    ):
        super(BottleneckMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.block_dims = block_dims
        self.norm = NORMS[norm]
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path

        self.name = name
        self.linear_in = nn.Linear(self.dim_in, self.block_dims[0][1])
        self.linear_out = nn.Linear(self.block_dims[-1][1], self.dim_out)
        blocks = []
        layernorms = []

        for block_dim in self.block_dims:
            wide, thin = block_dim
            blocks.append(BottleneckBlock(thin=thin, wide=wide))
            layernorms.append(self.norm(thin))

        self.blocks = nn.ModuleList(blocks)
        self.layernorms = nn.ModuleList(layernorms)

        if self.checkpoint is not None:
            self.load()

    def forward(self, x):
        x = self.linear_in(x)

        for block, norm in zip(self.blocks, self.layernorms):
            x = x + block(norm(x))

        out = self.linear_out(x)

        return out

    def load(self, name='in21k_cifar10'):
        """
        Load the model weights from a checkpoint.
        """
        checkpoint_dir = Path(self.checkpoint_path)
        checkpoint_name = f"{self.name}_{name}"
        config_path = checkpoint_dir / checkpoint_name / "config.txt"
        weight_path = checkpoint_dir / checkpoint_name / f"epoch_{self.checkpoint}"

        # Load config
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        # Load weights
        if weight_path.exists():
            print(f"Loading weights from {weight_path}")
            params = torch.load(weight_path, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(params, strict=False)
            print(f"Loaded checkpoint with missing keys: {missing_keys} and unexpected keys: {unexpected_keys}")
        else:
            raise FileNotFoundError(f"Weight file not found at {weight_path}")

class BottleneckBlock(nn.Module):
    def __init__(self, thin, wide, act=nn.GELU()):
        super(BottleneckBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(thin, wide), act, nn.Linear(wide, thin)
        )

    def forward(self, x):
        out = self.block(x)

        return out


def B_12_Wi_1024(dim_in, dim_out, checkpoint=None):
    block_dims = [[4 * 1024, 1024] for _ in range(12)]
    return BottleneckMLP(dim_in=dim_in, dim_out=dim_out, norm='layer', block_dims=block_dims, checkpoint=checkpoint,
                         name='B_' + str(len(block_dims)) + '-Wi_' + str(block_dims[0][1]) + '_res_' + str(int(np.sqrt(dim_in/3))))


def B_12_Wi_512(dim_in, dim_out, checkpoint=None):
    block_dims = [[4 * 512, 512] for _ in range(12)]
    return BottleneckMLP(dim_in=dim_in, dim_out=dim_out, norm='layer', block_dims=block_dims, checkpoint=checkpoint,
                         name='B_' + str(len(block_dims)) + '-Wi_' + str(block_dims[0][1]) + '_res_' + str(int(np.sqrt(dim_in/3))))


def B_6_Wi_1024(dim_in, dim_out, checkpoint=None, checkpoint_path='./checkpoints/'):
    block_dims = [[4 * 1024, 1024] for _ in range(6)]
    return BottleneckMLP(dim_in=dim_in, dim_out=dim_out, norm='layer', block_dims=block_dims, checkpoint=checkpoint,
                            checkpoint_path=checkpoint_path,
                            name='B_' + str(len(block_dims)) + '-Wi_' + str(block_dims[0][1]) + '_res_' + str(int(np.sqrt(dim_in/3))))


def B_6_Wi_512(dim_in, dim_out, checkpoint=None, checkpoint_path='./checkpoints/'):
    block_dims = [[4 * 512, 512] for _ in range(6)]
    return BottleneckMLP(dim_in=dim_in, dim_out=dim_out, norm='layer', block_dims=block_dims, checkpoint=checkpoint,
                            checkpoint_path=checkpoint_path,
                            name='B_' + str(len(block_dims)) + '-Wi_' + str(block_dims[0][1]) + '_res_' + str(int(np.sqrt(dim_in/3))))


model_list = {
    'B_12-Wi_1024': B_12_Wi_1024,
    'B_12-Wi_512': B_12_Wi_512,
    'B_6-Wi_1024': B_6_Wi_1024,
    'B_6-Wi_512': B_6_Wi_512
}


def get_model(architecture, checkpoint, resolution, num_classes, checkpoint_path='./checkpoints/'):
    return model_list[architecture](
        dim_in=resolution**2 * 3, dim_out=num_classes, 
        checkpoint=checkpoint, 
        checkpoint_path=checkpoint_path
    )

import logging
import os
from tempfile import NamedTemporaryFile
from typing import Optional

import torch as tr
import torch.nn.functional as F
import yaml
from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def linear_interpolate_dim(
    x: T, n: int, dim: int = -1, align_corners: bool = True
) -> T:
    n_dim = x.ndim
    assert 0 < n_dim <= 3
    if dim < 0:
        dim = n_dim + dim
    assert 0 <= dim < n_dim
    if x.size(dim) == n:
        return x

    swapped_dims = False
    if n_dim == 1:
        x = x.view(1, 1, -1)
    elif n_dim == 2:
        assert dim != 0  # TODO(cm)
        x = x.unsqueeze(1)
    elif x.ndim == 3:
        assert dim != 0  # TODO(cm)
        if dim == 1:
            x = x.swapaxes(1, 2)
            swapped_dims = True

    x = F.interpolate(x, n, mode="linear", align_corners=align_corners)
    if n_dim == 1:
        x = x.view(-1)
    elif n_dim == 2:
        x = x.squeeze(1)
    elif swapped_dims:
        x = x.swapaxes(1, 2)
    return x


def extract_model_and_synth_from_config(
    config_path: str, ckpt_path: Optional[str] = None
) -> (nn.Module, nn.Module):
    assert os.path.isfile(config_path)
    with open(config_path, "r") as in_f:
        config = yaml.safe_load(in_f)
    del config["ckpt_path"]

    tmp_config_file = NamedTemporaryFile()
    with open(tmp_config_file.name, "w") as out_f:
        yaml.dump(config, out_f)
        from cli import CustomLightningCLI

        cli = CustomLightningCLI(
            args=["-c", tmp_config_file.name],
            trainer_defaults=CustomLightningCLI.trainer_defaults,
            run=False,
        )
    lm = cli.model

    if ckpt_path is not None:
        log.info(f"Loading checkpoint from {ckpt_path}")
        assert os.path.isfile(ckpt_path)
        ckpt_data = tr.load(ckpt_path, map_location=tr.device("cpu"))
        lm.load_state_dict(ckpt_data["state_dict"])

    return lm.model, lm.synth_hat

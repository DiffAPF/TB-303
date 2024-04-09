import logging
import os
from typing import Optional, List, Tuple, Dict

import torch as tr
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F

from feature_extraction import LogMelSpecFeatureExtractor

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class Spectral2DCNNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int],
        b_dil: int,
        t_dil: int,
        pool_size: Tuple[int, int],
        use_ln: bool,
    ):
        super().__init__()
        self.use_ln = use_ln
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=(1, 1),
            dilation=(b_dil, t_dil),
            padding="same",
        )
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.act = nn.PReLU(num_parameters=out_ch)

    def forward(self, x: T) -> T:
        assert x.ndim == 4
        n_bin = x.size(2)
        n_frame = x.size(3)
        if self.use_ln:
            # TODO(cm): parameterize eps
            x = F.layer_norm(x, [n_bin, n_frame])
        x = self.conv(x)
        x = self.pool(x)
        x = self.act(x)
        return x


class Spectral2DCNN(nn.Module):
    global_param_names: List[str]

    def __init__(
        self,
        fe: LogMelSpecFeatureExtractor,
        in_ch: int = 1,
        kernel_size: Tuple[int, int] = (5, 7),
        out_channels: Optional[List[int]] = None,
        bin_dilations: Optional[List[int]] = None,
        temp_dilations: Optional[List[int]] = None,
        pool_size: Tuple[int, int] = (2, 1),
        use_ln: bool = True,
        n_temp_params: int = 1,
        temp_params_name: str = "mod_sig",
        temp_params_act_name: Optional[str] = "sigmoid",
        global_param_names: Optional[List[str]] = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.fe = fe
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        assert pool_size[1] == 1
        self.pool_size = pool_size
        self.use_ln = use_ln
        self.n_temporal_params = n_temp_params
        self.temporal_params_name = temp_params_name
        self.temp_params_act_name = temp_params_act_name
        self.dropout = dropout

        # Define default params
        if out_channels is None:
            out_channels = [64] * 5
        self.out_channels = out_channels
        if bin_dilations is None:
            bin_dilations = [1] * len(out_channels)
        self.bin_dilations = bin_dilations
        if temp_dilations is None:
            temp_dilations = [2**idx for idx in range(len(out_channels))]
        self.temp_dilations = temp_dilations
        assert len(out_channels) == len(bin_dilations) == len(temp_dilations)
        if global_param_names is None:
            global_param_names = []
        self.global_param_names = global_param_names
        if pool_size[1] == 1:
            log.info(
                f"Temporal receptive field: "
                f"{self.calc_receptive_field(kernel_size[1], temp_dilations)}"
            )

        # Define CNN
        layers = []
        curr_n_bins = fe.n_bins
        for out_ch, b_dil, t_dil in zip(out_channels, bin_dilations, temp_dilations):
            layers.append(
                Spectral2DCNNBlock(
                    in_ch, out_ch, kernel_size, b_dil, t_dil, pool_size, use_ln
                )
            )
            in_ch = out_ch
            curr_n_bins = curr_n_bins // pool_size[0]
        self.cnn = nn.Sequential(*layers)

        # Define temporal params
        self.out_temp = nn.Linear(out_channels[-1], n_temp_params)

        # Define global params
        self.out_global = nn.ModuleDict()
        for param_name in global_param_names:
            self.out_global[param_name] = nn.Sequential(
                nn.Linear(out_channels[-1], out_channels[-1] // 2),
                nn.Dropout(p=dropout),
                nn.PReLU(num_parameters=out_channels[-1] // 2),
                nn.Linear(out_channels[-1] // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: T) -> Dict[str, T]:
        assert x.ndim == 3
        out_dict = {}

        # Extract features
        log_spec = self.fe(x)
        out_dict["log_spec_wet"] = log_spec

        # Calc latent
        x = self.cnn(log_spec)
        x = tr.mean(x, dim=-2)
        latent = x.swapaxes(1, 2)
        out_dict["latent"] = latent

        # Calc temporal params
        x = self.out_temp(latent)
        if self.temp_params_act_name is None:
            out_temp = x
        elif self.temp_params_act_name == "sigmoid":
            out_temp = tr.sigmoid(x)
        elif self.temp_params_act_name == "tanh":
            out_temp = tr.tanh(x)
        elif self.temp_params_act_name == "clamp":
            out_temp = tr.clamp(x, min=0.0, max=1.0)
        # elif self.temp_params_act_name == "magic_clamp":
        #     out_temp = magic_clamp(x, min_value=0.0, max_value=1.0)
        else:
            raise ValueError(f"Unknown activation: {self.temp_params_act_name}")
        out_dict[self.temporal_params_name] = out_temp

        # Calc global params
        x = tr.mean(latent, dim=-2)
        for param_name, mlp in self.out_global.items():
            p_val_hat = mlp(x).squeeze(-1)
            out_dict[param_name] = p_val_hat

        return out_dict

    @staticmethod
    def calc_receptive_field(kernel_size: int, dilations: List[int]) -> int:
        """Compute the receptive field in samples."""
        assert dilations
        assert dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = kernel_size
        for dil in dilations[1:]:
            rf = rf + ((kernel_size - 1) * dil)
        return rf

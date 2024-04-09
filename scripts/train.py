import logging
import os
# Prevents a bug with PyTorch and CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from acid_ddsp.cli import CustomLightningCLI
from acid_ddsp.paths import CONFIGS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    config_name = "abstract_303/train.yml"

    config_path = os.path.join(CONFIGS_DIR, config_name)
    cli = CustomLightningCLI(args=["fit", "-c", config_path],
                             trainer_defaults=CustomLightningCLI.trainer_defaults)

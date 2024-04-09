import logging
import os

from torch import Tensor as T
from torch import nn
from torchaudio.transforms import MFCC

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MFCCL1(nn.Module):
    def __init__(
        self,
        sr: int,
        log_mels: bool = True,
        n_fft: int = 2048,
        hop_len: int = 512,
        n_mels: int = 128,
    ):
        super().__init__()
        self.mfcc = MFCC(
            sample_rate=sr,
            log_mels=log_mels,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_len,
                "n_mels": n_mels,
            },
        )
        self.l1 = nn.L1Loss()

    def forward(self, x_hat: T, x: T) -> T:
        return self.l1(self.mfcc(x_hat), self.mfcc(x))

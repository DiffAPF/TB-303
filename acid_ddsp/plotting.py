import io
import logging
import os
from typing import Optional, List

import PIL
import librosa
import librosa.display
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor as T
from torchvision.transforms import ToTensor

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def fig2img(fig: Figure, format: str = "png", dpi: int = 120) -> T:
    """Convert a matplotlib figure to tensor."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close("all")
    return image


def plot_waveforms_stacked(
    waveforms: List[T],
    sr: float,
    title: Optional[str] = None,
    waveform_labels: Optional[List[str]] = None,
    show: bool = False,
) -> Figure:
    assert waveforms
    if waveform_labels is None:
        waveform_labels = [None] * len(waveforms)
    assert len(waveform_labels) == len(waveforms)

    fig, axs = plt.subplots(
        nrows=len(waveforms),
        sharex="all",
        sharey="all",
        figsize=(7, 2 * len(waveforms)),
        squeeze=False,
    )
    axs = axs.squeeze(1)

    for idx, (ax, w, label) in enumerate(zip(axs, waveforms, waveform_labels)):
        assert 0 < w.ndim <= 2
        if w.ndim == 2:
            assert w.size(0) == 1
            w = w.squeeze(0)
        w = w.detach().float().cpu().numpy()
        if idx == len(waveforms) - 1:
            axis = "time"
        else:
            axis = None
        librosa.display.waveshow(w, axis=axis, sr=sr, label=label, ax=ax)
        ax.set_title(label)
        ax.grid(color="lightgray", axis="x")
        # ax.set_xticks([])
        # ax.set_yticks([])

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    # fig.savefig(os.path.join(OUT_DIR, f"3.svg"))

    if show:
        fig.show()
    return fig

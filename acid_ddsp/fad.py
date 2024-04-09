import logging
import os
import shutil
from typing import Optional

import torch as tr
import torchaudio
from torch import Tensor as T

import fadtk
from fadtk import (
    FrechetAudioDistance,
    cache_embedding_files,
)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def save_and_concat_fad_audio(
    sr: int,
    audio: T,
    dir_path: str,
    fade_n_samples: Optional[int] = None,
) -> None:
    assert audio.ndim == 2
    os.makedirs(dir_path, exist_ok=False)
    if fade_n_samples is not None:
        assert fade_n_samples > 0
        assert fade_n_samples < audio.size(-1)
        audio[:, :fade_n_samples] *= tr.linspace(0.0, 1.0, fade_n_samples)
        audio[:, -fade_n_samples:] *= tr.linspace(1.0, 0.0, fade_n_samples)
    audio = tr.flatten(audio).unsqueeze(0)
    torchaudio.save(os.path.join(dir_path, "audio.wav"), audio, sr)


def calc_fad(
    fad_model_name: str,
    baseline_dir: str,
    eval_dir: str,
    clean_up_baseline: bool = False,
    clean_up_eval: bool = False,
    workers: int = 0,
) -> float:
    assert os.path.isdir(baseline_dir)
    assert os.path.isdir(eval_dir)

    fad_models = {m.name: m for m in fadtk.get_all_models()}
    fad_model = fad_models[fad_model_name]
    cache_embedding_files(baseline_dir, fad_model, workers)
    cache_embedding_files(eval_dir, fad_model, workers)

    fad = FrechetAudioDistance(fad_model, audio_load_worker=workers, load_model=False)
    score = fad.score(baseline_dir, eval_dir)
    if clean_up_baseline:
        shutil.rmtree(baseline_dir)
    if clean_up_eval:
        shutil.rmtree(eval_dir)
    return score

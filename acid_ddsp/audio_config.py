import logging
import os

import torch as tr
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AudioConfig:
    def __init__(
        self,
        sr: int = 48000,
        buffer_size_seconds: float = 0.125,
        note_on_duration: float = 0.100,
        min_f0_hz: float = 32.70,
        max_f0_hz: float = 523.25,
        min_attack: float = 0.001,
        max_attack: float = 0.001,
        min_decay: float = 0.099,
        max_decay: float = 0.099,
        min_sustain: float = 0.5,
        max_sustain: float = 0.5,
        min_release: float = 0.025,
        max_release: float = 0.025,
        min_alpha: float = 1.0,
        max_alpha: float = 1.0,
        min_w_hz: float = 100.0,
        max_w_hz: float = 8000.0,
        min_q: float = 0.7071,
        max_q: float = 8.0,
        min_dist_gain: float = 1.0,
        max_dist_gain: float = 1.0,
        min_osc_shape: float = 1.0,
        max_osc_shape: float = 1.0,
        min_osc_gain: float = 1.0,
        max_osc_gain: float = 1.0,
        min_learned_alpha: float = 1.0,
        max_learned_alpha: float = 1.0,
        stability_eps: float = 0.001,
    ):
        self.sr = sr
        self.buffer_size_seconds = buffer_size_seconds
        self.note_on_duration = note_on_duration
        self.min_f0_hz = min_f0_hz
        self.max_f0_hz = max_f0_hz
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.min_sustain = min_sustain
        self.max_sustain = max_sustain
        self.min_release = min_release
        self.max_release = max_release
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_w_hz = min_w_hz
        self.max_w_hz = max_w_hz
        self.min_q = min_q
        self.max_q = max_q
        self.min_dist_gain = min_dist_gain
        self.max_dist_gain = max_dist_gain
        self.min_osc_shape = min_osc_shape
        self.max_osc_shape = max_osc_shape
        self.min_osc_gain = min_osc_gain
        self.max_osc_gain = max_osc_gain
        self.min_learned_alpha = min_learned_alpha
        self.max_learned_alpha = max_learned_alpha
        self.stability_eps = stability_eps

        self.n_samples = int(sr * buffer_size_seconds)
        self.min_w = self.calc_w(min_w_hz)
        self.max_w = self.calc_w(max_w_hz)

        self.min_vals = {
            "f0_hz": min_f0_hz,
            "attack": min_attack,
            "decay": min_decay,
            "sustain": min_sustain,
            "release": min_release,
            "alpha": min_alpha,
            "w_hz": min_w_hz,
            "w": self.min_w,
            "q": min_q,
            "dist_gain": min_dist_gain,
            "osc_shape": min_osc_shape,
            "osc_gain": min_osc_gain,
            "learned_alpha": min_learned_alpha,
        }
        self.max_vals = {
            "f0_hz": max_f0_hz,
            "attack": max_attack,
            "decay": max_decay,
            "sustain": max_sustain,
            "release": max_release,
            "alpha": max_alpha,
            "w_hz": max_w_hz,
            "w": self.max_w,
            "q": max_q,
            "dist_gain": max_dist_gain,
            "osc_shape": max_osc_shape,
            "osc_gain": max_osc_gain,
            "learned_alpha": max_learned_alpha,
        }
        for param_name in self.min_vals.keys():
            assert self.min_vals[param_name] <= self.max_vals[param_name]

    def calc_w(self, w_hz: float) -> float:
        return 2 * tr.pi * w_hz / self.sr

    def is_fixed(self, param_name: str) -> bool:
        return self.min_vals[param_name] == self.max_vals[param_name]

    def convert_from_0to1(self, param_name: str, val: T) -> T:
        assert val.min() >= 0.0
        assert val.max() <= 1.0
        return ((val * (self.max_vals[param_name] - self.min_vals[param_name]))
                + self.min_vals[param_name])

    def convert_to_0to1(self, param_name: str, val: T) -> T:
        assert val.min() >= self.min_vals[param_name]
        assert val.max() <= self.max_vals[param_name]
        if self.is_fixed(param_name):
            return tr.zeros_like(val)
        return ((val - self.min_vals[param_name])
                / (self.max_vals[param_name] - self.min_vals[param_name]))

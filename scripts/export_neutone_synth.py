import logging
import os
import pathlib
from typing import Dict, List, Tuple

import librosa
import torch as tr
import torch.nn as nn
from neutone_sdk import (
    WaveformToWaveformBase,
    NeutoneParameter,
    ContinuousNeutoneParameter,
)
from neutone_sdk.utils import save_neutone_model
from torch import Tensor as T

from audio_config import AudioConfig
from paths import OUT_DIR
from synths import AcidSynthLPBiquad, AcidSynthLPBiquadFSM

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def make_envelope(
    x: T, note_on_samples: int, curr_env_val: float
) -> Tuple[T, int, float]:
    n_samples = x.size(-1)
    n_env = (2 * n_samples) // note_on_samples
    n_env = max(n_env, 2)
    env = tr.linspace(1.0, 0.0, note_on_samples)
    env = env.repeat(n_env)

    out = tr.zeros((1, n_samples))

    # Find first non-zero index
    non_zero_indices = tr.nonzero(x.squeeze())
    first_nonzero_idx = -1
    if non_zero_indices.size(0) > 0:
        first_nonzero_idx = non_zero_indices[0, 0].item()

    # If all silence and no envelope is being continued, return silence
    if first_nonzero_idx == -1 and curr_env_val == 0.0:
        return out, 0, curr_env_val

    # Calc envelope continuation index
    cont_env_start_idx = 0
    if curr_env_val != 0.0:
        cont_env_start_idx = int(round((1.0 - curr_env_val) * (note_on_samples - 1)))
        # cont_env_start_idx = int(round(cont_env_start_idx))
        cont_env_start_idx = min(cont_env_start_idx, note_on_samples - 1)
        cont_env_start_idx = max(cont_env_start_idx, 0)

    # If there is no silence, return the envelope
    if first_nonzero_idx == 0:
        out[:, 0:n_samples] = env[cont_env_start_idx : cont_env_start_idx + n_samples]
        curr_env_val = out[0, -1].item()
        return out, first_nonzero_idx, curr_env_val

    # Continue the envelope if required
    cont_env_len = 0
    if curr_env_val != 0.0:
        cont_env_len = min(n_samples, note_on_samples - cont_env_start_idx)
        cont_env_end_idx = cont_env_start_idx + cont_env_len
        out[:, 0:cont_env_len] = env[cont_env_start_idx:cont_env_end_idx]

    # If all silence
    if first_nonzero_idx == -1:
        curr_env_val = out[0, -1].item()
        return out, 0, curr_env_val

    first_nonzero_idx = max(first_nonzero_idx, cont_env_len)
    n_non_zero_samples = n_samples - first_nonzero_idx
    out[:, first_nonzero_idx:n_samples] = env[0:n_non_zero_samples]
    curr_env_val = out[0, -1].item()
    return out, first_nonzero_idx, curr_env_val


class AcidSynth(nn.Module):
    def __init__(
        self,
        min_midi_f0: int = 30,
        max_midi_f0: int = 60,
        min_alpha: float = 0.2,
        max_alpha: float = 3.0,
        min_w_hz: float = 100.0,
        max_w_hz: float = 8000.0,
        min_q: float = 0.7071,
        max_q: float = 8.0,
        sr: int = 48000,
        note_on_duration: float = 0.125,
        osc_shape: float = 1.0,
        osc_gain: float = 0.5,
        dist_gain: float = 1.0,
        stability_eps: float = 0.001,
        use_fs: bool = False,
        win_len: int = 128,
        overlap: float = 0.75,
        oversampling_factor: int = 1,
    ):
        super().__init__()
        self.ac = AudioConfig(
            sr=sr,
            min_w_hz=min_w_hz,
            max_w_hz=max_w_hz,
            min_q=min_q,
            max_q=max_q,
            stability_eps=stability_eps,
        )
        if use_fs:
            self.synth = AcidSynthLPBiquadFSM(
                self.ac,
                win_len=win_len,
                overlap=overlap,
                oversampling_factor=oversampling_factor,
            )
        else:
            self.synth = AcidSynthLPBiquad(self.ac)
            self.synth.toggle_scriptable(is_scriptable=True)

        self.min_midi_f0 = min_midi_f0
        self.max_midi_f0 = max_midi_f0
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_w_hz = min_w_hz
        self.max_w_hz = max_w_hz
        self.min_q = min_q
        self.max_q = max_q
        self.sr = sr
        self.register_buffer("note_on_duration", tr.full((1,), note_on_duration))
        self.register_buffer("osc_shape", tr.full((1,), osc_shape))
        self.register_buffer("osc_gain", tr.full((1,), osc_gain))
        self.register_buffer("dist_gain", tr.full((1,), dist_gain))
        self.use_fs = use_fs
        self.win_len = win_len
        self.overlap = overlap
        self.hop_len = int(win_len * (1 - overlap))
        assert win_len % self.hop_len == 0, "Hop length must divide into window length."
        self.oversampling_factor = oversampling_factor

        self.note_on_samples = int(note_on_duration * self.sr)
        self.curr_env_val = 1.0
        self.register_buffer("phase", tr.zeros((1, 1), dtype=tr.double))
        self.register_buffer("zi", tr.zeros((1, 2)))
        self.midi_f0_to_hz = {
            idx: tr.tensor(librosa.midi_to_hz(idx)).view(1).float()
            for idx in range(min_midi_f0, max_midi_f0 + 1)
        }

    def reset(self) -> None:
        self.curr_env_val = 1.0
        self.phase.zero_()
        self.zi.zero_()

    def forward(
        self,
        x: T,
        midi_f0_0to1: T,
        alpha_0to1: T,
        w_mod_sig: T,
        q_mod_sig: T,
    ) -> T:
        n_samples = x.size(-1)
        alpha = alpha_0to1 * (self.max_alpha - self.min_alpha) + self.min_alpha
        env, _, new_env_val = make_envelope(x, self.note_on_samples, self.curr_env_val)
        self.curr_env_val = new_env_val
        if alpha != 1.0:
            tr.pow(env, alpha, out=env)

        midi_f0 = (
            midi_f0_0to1 * (self.max_midi_f0 - self.min_midi_f0) + self.min_midi_f0
        )
        midi_f0 = midi_f0.round().int().item()
        f0_hz = self.midi_f0_to_hz[midi_f0]

        filter_args = {
            "w_mod_sig": w_mod_sig,
            "q_mod_sig": q_mod_sig,
            "zi": self.zi,
        }
        global_params = {
            "osc_shape": self.osc_shape,
            "osc_gain": self.osc_gain,
            "dist_gain": self.dist_gain,
            "learned_alpha": alpha,
        }
        synth_out = self.synth(
            n_samples=n_samples,
            f0_hz=f0_hz,
            note_on_duration=self.note_on_duration,
            phase=self.phase,
            filter_args=filter_args,
            global_params=global_params,
            envelope=env,
        )
        wet = synth_out["wet"]

        period_completion = (n_samples / (self.sr / f0_hz.double())) % 1.0
        tr.add(self.phase, 2 * tr.pi * period_completion, out=self.phase)
        if not self.use_fs:
            y_a = synth_out["y_a"]
            self.zi[:, :] = y_a[:, -2:]
        return wet


class AcidSynthWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        if self.model.use_fs:
            return f"acid_synth_lp_fs_{self.model.win_len}"
        else:
            return "acid_synth_lp_td"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "Low-pass biquad TB-303 DDSP implementation."

    def get_model_long_description(self) -> str:
        return "Low-pass biquad TB-303 DDSP implementation for 'Differentiable All-pole Filters for Time-varying Audio Systems'."

    def get_technical_description(self) -> str:
        return "Wrapper for a TB-303 DDSP implementation consisting of a sawtooth or square wave oscillator, time-varying low-pass biquad filter, and hyperbolic tangent distortion."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            # "Paper": "tbd",
            "Code": "https://github.com/DiffAPF/TB-303",
        }

    def get_tags(self) -> List[str]:
        return ["subtractive synth", "acid", "TB-303"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            ContinuousNeutoneParameter(
                "midi_f0",
                f"Oscillator pitch quantized to the nearest midi pitch [f{self.model.min_midi_f0}, f{self.model.max_midi_f0}]",
                default_value=0.5,
            ),
            ContinuousNeutoneParameter(
                "alpha",
                f"Decaying envelope generator exponent [f{self.model.min_alpha}, f{self.model.max_alpha}]",
                default_value=0.5,
            ),
            ContinuousNeutoneParameter(
                "w_mod_sig",
                f"Filter cutoff frequency [f{self.model.min_w_hz} Hz, f{self.model.max_w_hz} Hz]",
                default_value=1.0,
            ),
            ContinuousNeutoneParameter(
                "q_mod_sig",
                f"Filter resonance Q-factor [f{self.model.min_q}, f{self.model.max_q}]",
                default_value=0.5,
            ),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return True

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return True

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [self.model.sr]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        if self.model.use_fs:
            return [
                bs
                for bs in range(
                    self.model.win_len,
                    max(self.model.win_len + 1, 10000),
                    self.model.hop_len,
                )
            ]
        else:
            return []

    @tr.jit.export
    def reset_model(self) -> bool:
        self.model.reset()
        return True

    def do_forward_pass(self, x: T, params: Dict[str, T]) -> T:
        n_samples = x.size(-1)
        midi_f0_0to1 = params["midi_f0"]
        w_mod_sig = params["w_mod_sig"].unsqueeze(0)
        q_mod_sig = params["q_mod_sig"].unsqueeze(0)
        w_mod_sig = w_mod_sig.expand(-1, n_samples)
        q_mod_sig = q_mod_sig.expand(-1, n_samples)
        alpha_0to1 = params["alpha"]
        x = x.unsqueeze(1)
        y = self.model(x, midi_f0_0to1, alpha_0to1, w_mod_sig, q_mod_sig)
        y = y.squeeze(1)
        return y


if __name__ == "__main__":
    model = AcidSynth(use_fs=False)
    wrapper = AcidSynthWrapper(model)
    root_dir = pathlib.Path(
        os.path.join(OUT_DIR, "neutone_models", wrapper.get_model_name())
    )
    save_neutone_model(
        wrapper,
        root_dir,
        submission=False,
        dump_samples=False,
        test_offline_mode=False,
        speed_benchmark=False,
    )

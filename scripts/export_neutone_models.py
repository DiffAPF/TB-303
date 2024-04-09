import logging
import os
import pathlib
from typing import Dict, List

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

import util
from models import Spectral2DCNN
from paths import OUT_DIR, MODELS_DIR
from scripts.export_neutone_synth import make_envelope
from synths import (
    AcidSynthBase,
    AcidSynthLearnedBiquadCoeff,
    AcidSynthLPBiquad,
    AcidSynthLSTM,
)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidSynthModel(nn.Module):
    def __init__(
        self,
        model: Spectral2DCNN,
        synth: AcidSynthBase,
        min_midi_f0: int = 30,
        max_midi_f0: int = 60,
    ):
        super().__init__()
        self.model = model
        self.synth = synth
        self.min_midi_f0 = min_midi_f0
        self.max_midi_f0 = max_midi_f0

        self.is_coeff_synth = isinstance(synth, AcidSynthLearnedBiquadCoeff)
        self.is_rnn_synth = isinstance(synth, AcidSynthLSTM)
        self.is_td_synth = type(synth) in {
            AcidSynthLearnedBiquadCoeff,
            AcidSynthLPBiquad,
        }
        if self.is_td_synth:
            self.synth.toggle_scriptable(True)
        self.ac = synth.ac
        self.sr = synth.ac.sr
        self.register_buffer(
            "note_on_duration", tr.full((1,), self.ac.note_on_duration)
        )
        self.note_on_samples = self.ac.n_samples
        self.curr_env_val = 1.0
        self.register_buffer("phase", tr.zeros((1, 1), dtype=tr.double))
        self.register_buffer("zi", tr.zeros((1, 2)))
        self.register_buffer("h_n", tr.empty(0))
        self.register_buffer("c_n", tr.empty(0))
        self.midi_f0_to_hz = {
            idx: tr.tensor(librosa.midi_to_hz(idx)).view(1).float()
            for idx in range(min_midi_f0, max_midi_f0 + 1)
        }

    def reset(self) -> None:
        self.curr_env_val = 1.0
        self.phase.zero_()
        self.zi.zero_()
        self.h_n = tr.empty(0)
        self.c_n = tr.empty(0)

    def forward(self, x: T, midi_f0_0to1: T) -> T:
        n_samples = x.size(-1)

        midi_f0 = (
            midi_f0_0to1 * (self.max_midi_f0 - self.min_midi_f0) + self.min_midi_f0
        )
        midi_f0 = midi_f0.round().int().item()
        f0_hz = self.midi_f0_to_hz[midi_f0]

        model_out = self.model(x)
        alpha = self.ac.convert_from_0to1(
            "learned_alpha", model_out["learned_alpha_0to1"]
        )
        env, _, new_env_val = make_envelope(x, self.note_on_samples, self.curr_env_val)
        self.curr_env_val = new_env_val
        if alpha != 1.0:
            tr.pow(env, alpha, out=env)

        if self.is_coeff_synth:
            logits = model_out["logits"]
            filter_args = {"logits": logits}
        else:
            w_mod_sig = model_out["w_mod_sig"]
            q_mod_sig = model_out["q_0to1"].unsqueeze(1)
            filter_args = {
                "w_mod_sig": w_mod_sig,
                "q_mod_sig": q_mod_sig,
            }
        if self.is_td_synth:
            filter_args["zi"] = self.zi
        if self.is_rnn_synth:
            filter_args["h_n"] = self.h_n
            filter_args["c_n"] = self.c_n
        global_params = {
            "osc_shape": self.ac.convert_from_0to1(
                "osc_shape", model_out["osc_shape_0to1"]
            ),
            "osc_gain": self.ac.convert_from_0to1(
                "osc_gain", model_out["osc_gain_0to1"]
            ),
            "dist_gain": self.ac.convert_from_0to1(
                "dist_gain", model_out["dist_gain_0to1"]
            ),
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
        if self.is_td_synth:
            y_a = synth_out["y_a"]
            self.zi[:, :] = y_a[:, -2:]
        if self.is_rnn_synth:
            self.h_n = synth_out["h_n"]
            self.c_n = synth_out["c_n"]
        return wet


class AcidSynthModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        if self.model.is_coeff_synth:
            prefix = "acid_synth_model_coeff"
        elif self.model.is_rnn_synth:
            prefix = "acid_synth_model_rnn"
        else:
            prefix = "acid_synth_model_lp"
        if self.model.is_td_synth:
            suffix = "_td"
        else:
            suffix = f"_fs"
        return f"{prefix}{suffix}"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "TB-303 DDSP and control model implementation."

    def get_model_long_description(self) -> str:
        return "TB-303 DDSP and control model implementation for 'Differentiable All-pole Filters for Time-varying Audio Systems'."

    def get_technical_description(self) -> str:
        return "Wrapper for a TB-303 DDSP and control model implementation consisting of an analysis CNN that generates the parameters for a differentiable synth."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            # "Paper": "tbd",
            "Code": "https://github.com/DiffAPF/TB-303",
        }

    def get_tags(self) -> List[str]:
        return ["subtractive synth", "acid", "TB-303", "sound matching"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            ContinuousNeutoneParameter(
                "midi_f0",
                f"Oscillator pitch quantized to the nearest midi pitch [f{self.model.min_midi_f0}, f{self.model.max_midi_f0}]",
                default_value=0.50,
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
        return [self.model.ac.n_samples]

    def do_forward_pass(self, x: T, params: Dict[str, T]) -> T:
        midi_f0_0to1 = params["midi_f0"]
        x = x.unsqueeze(1)
        y = self.model(x, midi_f0_0to1)
        y = y.squeeze(1)
        return y


if __name__ == "__main__":
    model_dir = MODELS_DIR
    model_name = "cnn_mss_coeff_td__abstract_303_48k__6k__4k_min__epoch_199_step_1200"
    # model_name = "cnn_mss_coeff_fs_128__abstract_303_48k__6k__4k_min__epoch_143_step_864"
    # model_name = "cnn_mss_coeff_fs_256__abstract_303_48k__6k__4k_min__epoch_159_step_960"
    # model_name = "cnn_mss_coeff_fs_512__abstract_303_48k__6k__4k_min__epoch_191_step_1152"
    # model_name = "cnn_mss_coeff_fs_1024__abstract_303_48k__6k__4k_min__epoch_167_step_1008"
    # model_name = "cnn_mss_coeff_fs_2048__abstract_303_48k__6k__4k_min__epoch_159_step_960"
    # model_name = "cnn_mss_coeff_fs_4096__abstract_303_48k__6k__4k_min__epoch_167_step_1008"
    # model_name = "cnn_mss_lp_td__abstract_303_48k__6k__4k_min__epoch_183_step_1104"
    # model_name = "cnn_mss_lp_fs_128__abstract_303_48k__6k__4k_min__epoch_183_step_1104"
    # model_name = "cnn_mss_lp_fs_256__abstract_303_48k__6k__4k_min__epoch_183_step_1104"
    # model_name = "cnn_mss_lp_fs_512__abstract_303_48k__6k__4k_min__epoch_183_step_1104"
    # model_name = "cnn_mss_lp_fs_1024__abstract_303_48k__6k__4k_min__epoch_135_step_816"
    # model_name = "cnn_mss_lp_fs_2048__abstract_303_48k__6k__4k_min__epoch_135_step_816"
    # model_name = "cnn_mss_lp_fs_4096__abstract_303_48k__6k__4k_min__epoch_135_step_816"
    # model_name = "cnn_mss_lstm_64__abstract_303_48k__6k__4k_min__epoch_175_step_1056"

    config_path = os.path.join(model_dir, model_name, "config.yaml")
    ckpt_path = os.path.join(model_dir, model_name, "checkpoints", f"{model_name}.ckpt")
    cnn, synth = util.extract_model_and_synth_from_config(config_path, ckpt_path)

    model = AcidSynthModel(cnn, synth)
    wrapper = AcidSynthModelWrapper(model)
    root_dir = pathlib.Path(
        os.path.join(
            OUT_DIR, "neutone_models", f"{wrapper.get_model_name()}__{model_name}"
        )
    )
    save_neutone_model(
        wrapper,
        root_dir,
        submission=False,
        dump_samples=False,
        test_offline_mode=False,
        speed_benchmark=False,
    )

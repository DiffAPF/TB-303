import logging
import os

import torch as tr
import yaml
from torch.utils import benchmark

from audio_config import AudioConfig
from paths import CONFIGS_DIR
from synths import make_synth

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

ac_config_path = os.path.join(CONFIGS_DIR, "abstract_303", "audio_config.yml")
with open(ac_config_path, "r") as f:
    ac_kwargs = yaml.safe_load(f)["init_args"]
ac = AudioConfig(**ac_kwargs)

batch_size = 34
win_lens = [128, 256, 512, 1024, 2048, 4096]
overlaps = [0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875]
oversampling_factor = 1
n_hidden = 64

num_threads = 1


def main() -> None:
    tr.manual_seed(42)

    assert ac.sr == 48000
    n_samples = 6000
    f0_hz = tr.full((batch_size,), 220.0)
    note_on_duration = tr.full((batch_size,), 0.125)
    phase = (tr.rand((batch_size, 1)) * 2 * tr.pi) - tr.pi

    osc_shape = tr.rand((batch_size,))
    osc_gain = tr.rand((batch_size,))
    dist_gain = tr.rand((batch_size,))
    learned_alpha = tr.rand((batch_size,))

    w_mod_sig = tr.rand((batch_size, 188))
    q_mod_sig = tr.rand((batch_size, 188))
    logits = tr.rand((batch_size, 188, 5))

    osc_shape.requires_grad_(True)
    osc_gain.requires_grad_(True)
    dist_gain.requires_grad_(True)
    learned_alpha.requires_grad_(True)
    w_mod_sig.requires_grad_(True)
    q_mod_sig.requires_grad_(True)
    logits.requires_grad_(True)

    filter_args = {"logits": logits, "w_mod_sig": w_mod_sig, "q_mod_sig": q_mod_sig}
    global_params = {
        "osc_shape": osc_shape,
        "osc_gain": osc_gain,
        "dist_gain": dist_gain,
        "learned_alpha": learned_alpha,
    }

    globals = {
        "n_samples": n_samples,
        "f0_hz": f0_hz,
        "note_on_duration": note_on_duration,
        "phase": phase,
        "filter_args": filter_args,
        "global_params": global_params,
    }
    results = []

    globals["synth"] = make_synth("AcidSynthLearnedBiquadCoeff", ac)
    results.append(
        benchmark.Timer(
            stmt="y = synth(n_samples, f0_hz, note_on_duration, phase, filter_args, global_params); loss = y['wet'].sum(); loss.backward()",
            globals=globals,
            sub_label="coeff_biquad",
            description="td",
            num_threads=num_threads,
        ).blocked_autorange(min_run_time=1)
    )
    for win_len, overlap in zip(win_lens, overlaps):
        synth_kwargs = {
            "win_len": win_len,
            "overlap": overlap,
            "oversampling_factor": oversampling_factor,
        }
        globals["synth"] = make_synth(
            "AcidSynthLearnedBiquadCoeffFSM",
            ac,
            **synth_kwargs,
        )
        results.append(
            benchmark.Timer(
                stmt="y = synth(n_samples, f0_hz, note_on_duration, phase, filter_args, global_params); loss = y['wet'].sum(); loss.backward()",
                globals=globals,
                sub_label=f"coeff_biquad",
                description=f"fs_{win_len}",
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=1)
        )

    globals["synth"] = make_synth("AcidSynthLPBiquad", ac)
    results.append(
        benchmark.Timer(
            stmt="y = synth(n_samples, f0_hz, note_on_duration, phase, filter_args, global_params); loss = y['wet'].sum(); loss.backward()",
            globals=globals,
            sub_label="lp_biquad",
            description="td",
            num_threads=num_threads,
        ).blocked_autorange(min_run_time=1)
    )
    for win_len, overlap in zip(win_lens, overlaps):
        synth_kwargs = {
            "win_len": win_len,
            "overlap": overlap,
            "oversampling_factor": oversampling_factor,
        }
        globals["synth"] = make_synth(
            "AcidSynthLPBiquadFSM",
            ac,
            **synth_kwargs,
        )
        results.append(
            benchmark.Timer(
                stmt="y = synth(n_samples, f0_hz, note_on_duration, phase, filter_args, global_params); loss = y['wet'].sum(); loss.backward()",
                globals=globals,
                sub_label=f"lp_biquad",
                description=f"fs_{win_len}",
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=1)
        )

    globals["synth"] = make_synth("AcidSynthLSTM", ac, n_hidden=n_hidden)
    results.append(
        benchmark.Timer(
            stmt="y = synth(n_samples, f0_hz, note_on_duration, phase, filter_args, global_params); loss = y['wet'].sum(); loss.backward()",
            globals=globals,
            sub_label=f"lstm{n_hidden}",
            description="td",
            num_threads=num_threads,
        ).blocked_autorange(min_run_time=1)
    )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()

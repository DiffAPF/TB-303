import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict

import wandb
from auraloss.time import ESRLoss
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor as T, nn

import util
from acid_ddsp.plotting import fig2img, plot_waveforms_stacked
from lightning import AcidDDSPLightingModule

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ConsoleLRMonitor(LearningRateMonitor):
    # TODO(cm): enable every n steps
    def on_train_epoch_start(self, trainer: Trainer, *args: Any, **kwargs: Any) -> None:
        super().on_train_epoch_start(trainer, *args, **kwargs)
        if self.logging_interval != "step":
            interval = "epoch" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)
            latest_stat_str = {k: f"{v:.8f}" for k, v in latest_stat.items()}
            if latest_stat:
                log.info(f"\nCurrent LR: {latest_stat_str}")


class LogModSigAndSpecCallback(Callback):
    def __init__(self, n_examples: int = 5) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.out_dicts = {}
        self.esr = ESRLoss()
        self.l1 = nn.L1Loss()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AcidDDSPLightingModule,
        out_dict: Dict[str, T],
        batch: Dict[str, T],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx != 0:  # TODO(cm): tmp
            return
        example_idx = batch_idx // trainer.accumulate_grad_batches
        if example_idx < self.n_examples:
            if example_idx not in self.out_dicts:
                if "x_hat" in out_dict and out_dict["x_hat"] is not None:
                    x_hat = out_dict["x_hat"]
                    assert x_hat.ndim == 2
                    x_hat = x_hat.unsqueeze(1)
                    log_spec_x_hat = pl_module.spectral_visualizer(x_hat).squeeze(1)
                    out_dict["log_spec_x_hat"] = log_spec_x_hat
                out_dict = {
                    k: v.detach().cpu() for k, v in out_dict.items() if v is not None
                }
                # TODO(cm): tmp
                # self.out_dicts[example_idx] = out_dict
                for idx in range(self.n_examples):
                    idx_out_dict = {
                        k: v[idx : idx + 1] for k, v in out_dict.items() if v.ndim > 0
                    }
                    self.out_dicts[idx] = idx_out_dict

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: AcidDDSPLightingModule
    ) -> None:
        images = []
        for example_idx in range(self.n_examples):
            if example_idx not in self.out_dicts:
                log.warning(f"example_idx={example_idx} not in out_dicts")
                continue

            out_dict = self.out_dicts[example_idx]
            envelope = out_dict.get("envelope")
            log_spec_wet = out_dict.get("log_spec_wet")
            log_spec_wet_hat = out_dict.get("log_spec_wet_hat")

            temp_params = out_dict.get("temp_params")
            temp_params_hat = out_dict["temp_params_hat"]
            mod_sig_esr = -1
            mod_sig_l1 = -1
            if pl_module.temp_params_name == pl_module.temp_params_name_hat:
                if temp_params is not None and temp_params_hat is not None:
                    assert temp_params.ndim == 3 and temp_params_hat.ndim == 3
                    if temp_params.size(2) == 1 and temp_params_hat.size(2) == 1:
                        mod_sig_esr = self.esr(
                            temp_params[0], temp_params_hat[0]
                        ).item()
                        mod_sig_l1 = self.l1(temp_params[0], temp_params_hat[0]).item()

            q = out_dict.get("q", [-1])
            q_hat = out_dict.get("q_hat", [-1])
            dist_gain = out_dict.get("dist_gain", [-1])
            dist_gain_hat = out_dict.get("dist_gain_hat", [-1])
            osc_shape = out_dict.get("osc_shape", [-1])
            osc_shape_hat = out_dict.get("osc_shape_hat", [-1])
            osc_gain = out_dict.get("osc_gain", [-1])
            osc_gain_hat = out_dict.get("osc_gain_hat", [-1])
            learned_alpha = out_dict.get("learned_alpha", [-1])
            learned_alpha_hat = out_dict.get("learned_alpha_hat", [-1])

            y_coords = pl_module.spectral_visualizer.center_freqs
            y_ticks = [
                (idx, f"{f:.0f}")
                for idx, f in list(enumerate(y_coords))[:: len(y_coords) // 10]
            ]
            y_indices, y_tick_labels = zip(*y_ticks)
            vmin = None
            vmax = None
            if log_spec_wet is not None and log_spec_wet_hat is not None:
                vmin = min(log_spec_wet[0].min(), log_spec_wet_hat[0].min())
                vmax = max(log_spec_wet[0].max(), log_spec_wet_hat[0].max())

            title = f"{trainer.global_step}_idx_{example_idx}"
            fig, ax = plt.subplots(nrows=3, figsize=(6, 18), squeeze=True)
            fig.suptitle(title, fontsize=14)

            if log_spec_wet is not None:
                ax[0].imshow(
                    log_spec_wet[0].numpy(),
                    extent=[0, log_spec_wet.size(2), 0, log_spec_wet.size(1)],
                    aspect=log_spec_wet.size(2) / log_spec_wet.size(1),
                    origin="lower",
                    cmap="magma",
                    interpolation="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax[0].set_xlabel("n_frames")
                ax[0].set_yticks(y_indices, y_tick_labels)
                ax[0].set_ylabel("Freq (Hz)")
                ax[0].set_title("log_spec_wet")

            if log_spec_wet_hat is not None:
                ax[1].imshow(
                    log_spec_wet_hat[0].numpy(),
                    extent=[0, log_spec_wet_hat.size(2), 0, log_spec_wet_hat.size(1)],
                    aspect=log_spec_wet_hat.size(2) / log_spec_wet_hat.size(1),
                    origin="lower",
                    cmap="magma",
                    interpolation="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax[1].set_xlabel("n_frames")
                ax[1].set_yticks(y_indices, y_tick_labels)
                ax[1].set_ylabel("Freq (Hz)")
                ax[1].set_title("log_spec_wet_hat")

            if pl_module.log_envelope and envelope is not None:
                assert envelope.ndim == 2
                envelope = util.linear_interpolate_dim(
                    envelope, pl_module.ac.n_samples, dim=1, align_corners=True
                )
                ax[2].plot(envelope[0].numpy(), label="env", color="blue")
                ax[2].set(aspect=envelope.size(1))

            if temp_params is not None:
                assert temp_params.ndim == 3
                temp_params = util.linear_interpolate_dim(
                    temp_params, pl_module.ac.n_samples, dim=1, align_corners=True
                )
                temp_params_np = temp_params[0].numpy()
                for idx in range(temp_params_np.shape[1]):
                    ax[2].plot(
                        temp_params_np[:, idx],
                        label=f"{pl_module.temp_params_name}_{idx}",
                        color="black",
                    )
                ax[2].set(aspect=temp_params.size(1))
                # mod_sig_fitted = piecewise_fitting_noncontinuous(
                #     mod_sig_np, degree=degree, n_knots=n_segments - 1
                # )
                # ax[2].plot(
                #     mod_sig_fitted,
                #     label=f"poly{degree}s{n_segments}",
                #     color="red"
                # )

            if temp_params_hat is not None:
                assert temp_params_hat.ndim == 3
                temp_params_hat = util.linear_interpolate_dim(
                    temp_params_hat, pl_module.ac.n_samples, dim=1, align_corners=True
                )
                temp_params_hat_np = temp_params_hat[0].numpy()
                for idx in range(temp_params_hat_np.shape[1]):
                    ax[2].plot(
                        temp_params_hat_np[:, idx],
                        label=f"{pl_module.temp_params_name_hat}_{idx}",
                        color="orange",
                    )
                ax[2].set(aspect=temp_params_hat.size(1))

            ax[2].set_xlabel("n_samples")
            ax[2].set_ylabel("Amplitude")
            ax[2].set_ylim(0, 1)
            ax[2].set_title(
                # f"env (blu), ms (blk), ms_hat (orange), p{degree}s{n_segments} (red)\n"
                f"env (blue), mod_sig (black), mod_sig_hat (orange)\n"
                f"ms_l1: {mod_sig_l1:.3f}, ms_esr: {mod_sig_esr:.3f}\n"
                f"q: {q[0]:.2f}, q': {q_hat[0]:.2f}, "
                f"dg: {dist_gain[0]:.2f}, dg': {dist_gain_hat[0]:.2f}, "
                f"la: {learned_alpha[0]:.2f}, la': {learned_alpha_hat[0]:.2f}\n"
                f"os: {osc_shape[0]:.2f}, os': {osc_shape_hat[0]:.2f}, "
                f"og: {osc_gain[0]:.2f}, og': {osc_gain_hat[0]:.2f}"
            )

            fig.tight_layout()
            img = fig2img(fig)
            images.append(img)

        if images:
            for logger in trainer.loggers:
                # TODO(cm): enable for tensorboard as well
                if isinstance(logger, WandbLogger):
                    logger.log_image(
                        key="spectrograms", images=images, step=trainer.global_step
                    )

        self.out_dicts.clear()


# TODO(cm): make ABC
class LogAudioCallback(Callback):
    def __init__(self, n_examples: int = 5, render_time_sec: float = 1.0) -> None:
        super().__init__()
        self.n_examples = n_examples
        self.render_time_sec = render_time_sec
        self.out_dicts = {}
        columns = [f"idx_{idx}" for idx in range(n_examples)]
        self.table = wandb.Table(columns=columns)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AcidDDSPLightingModule,
        out_dict: Dict[str, T],
        batch: Dict[str, T],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx != 0:  # TODO(cm): tmp
            return
        example_idx = batch_idx // trainer.accumulate_grad_batches
        if example_idx < self.n_examples:
            if example_idx not in self.out_dicts:
                out_dict = {
                    k: v.detach().cpu() for k, v in out_dict.items() if v is not None
                }
                # TODO(cm): tmp
                # self.out_dicts[example_idx] = out_dict
                for idx in range(self.n_examples):
                    idx_out_dict = {
                        k: v[idx : idx + 1] for k, v in out_dict.items() if v.ndim > 0
                    }
                    self.out_dicts[idx] = idx_out_dict

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: AcidDDSPLightingModule
    ) -> None:
        sample_time_sec = pl_module.ac.buffer_size_seconds
        n_repeat = math.ceil(self.render_time_sec / sample_time_sec)

        images = []
        dry_audio_waveforms = []
        wet_waveforms = []
        wet_hat_waveforms = []
        wet_eval_waveforms = []
        for example_idx in range(self.n_examples):
            if example_idx not in self.out_dicts:
                log.warning(f"example_idx={example_idx} not in out_dicts")
                continue

            out_dict = self.out_dicts[example_idx]
            title = f"{trainer.global_step}_idx_{example_idx}"
            dry = out_dict.get("dry")
            wet = out_dict.get("wet")
            wet_hat = out_dict.get("wet_hat")
            wet_eval = out_dict.get("wet_eval")
            waveforms = []
            labels = []

            if dry is not None:
                dry = dry[0:1]
                waveforms.append(dry)
                labels.append("dry")
                dry = dry.repeat(1, n_repeat)
                dry_audio_waveforms.append(dry.swapaxes(0, 1).numpy())
            if wet is not None:
                wet = wet[0:1]
                waveforms.append(wet)
                labels.append("wet")
                wet = wet.repeat(1, n_repeat)
                wet_waveforms.append(wet.swapaxes(0, 1).numpy())
            if wet_hat is not None:
                wet_hat = wet_hat[0:1]
                waveforms.append(wet_hat)
                labels.append("wet_hat")
                wet_hat = wet_hat.repeat(1, n_repeat)
                wet_hat_waveforms.append(wet_hat.swapaxes(0, 1).numpy())
            if wet_eval is not None:
                wet_eval = wet_eval[0:1]
                waveforms.append(wet_eval)
                labels.append("wet_eval")
                wet_eval = wet_eval.repeat(1, n_repeat)
                wet_eval_waveforms.append(wet_eval.swapaxes(0, 1).numpy())

            fig = plot_waveforms_stacked(waveforms, pl_module.ac.sr, title, labels)
            img = fig2img(fig)
            images.append(img)

        # import torchaudio
        # import torch
        # for idx, wet in enumerate(wet_waveforms):
        #     torchaudio.save(f"../out/x_{idx}_dg_{pl_module.ac.min_dist_gain}.wav", torch.tensor(wet).T, pl_module.ac.sr)
        # exit()

        for logger in trainer.loggers:
            # TODO(cm): enable for tensorboard as well
            if isinstance(logger, WandbLogger):
                if images:
                    logger.log_image(
                        key="waveforms", images=images, step=trainer.global_step
                    )

                data = defaultdict(list)
                for idx, curr_dry in enumerate(dry_audio_waveforms):
                    data["dry"].append(
                        wandb.Audio(
                            curr_dry,
                            caption=f"dry_{idx}",
                            sample_rate=int(pl_module.ac.sr),
                        )
                    )
                for idx, curr_wet in enumerate(wet_waveforms):
                    data["wet"].append(
                        wandb.Audio(
                            curr_wet,
                            caption=f"wet_{idx}",
                            sample_rate=int(pl_module.ac.sr),
                        )
                    )
                for idx, curr_wet_hat in enumerate(wet_hat_waveforms):
                    data["wet_hat"].append(
                        wandb.Audio(
                            curr_wet_hat,
                            caption=f"wet_hat_{idx}",
                            sample_rate=int(pl_module.ac.sr),
                        )
                    )
                for idx, curr_wet_eval in enumerate(wet_eval_waveforms):
                    data["wet_eval"].append(
                        wandb.Audio(
                            curr_wet_eval,
                            caption=f"wet_eval_{idx}",
                            sample_rate=int(pl_module.ac.sr),
                        )
                    )
                data = list(data.values())
                for row in data:
                    self.table.add_data(*row)
                logger.log_table(
                    key="audio",
                    columns=self.table.columns,
                    data=self.table.data,
                    step=trainer.global_step,
                )

        self.out_dicts.clear()

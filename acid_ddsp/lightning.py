import logging
import os
from contextlib import suppress
from datetime import datetime
from typing import Dict, Any, Optional, List, Mapping

import auraloss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import torch as tr
from auraloss.time import ESRLoss
from torch import Tensor as T
from torch import nn
from tqdm import tqdm

import util
from audio_config import AudioConfig
from fad import save_and_concat_fad_audio, calc_fad
from feature_extraction import LogMelSpecFeatureExtractor
from losses import MFCCL1
from paths import OUT_DIR
from synths import AcidSynthLPBiquad, AcidSynthBase

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AcidDDSPLightingModule(pl.LightningModule):
    def __init__(
        self,
        ac: AudioConfig,
        model: nn.Module,
        loss_func: nn.Module,
        spectral_visualizer: LogMelSpecFeatureExtractor,
        synth: Optional[AcidSynthBase] = None,
        synth_hat: Optional[AcidSynthBase] = None,
        synth_eval: Optional[AcidSynthBase] = None,
        temp_params_name: Optional[str] = None,
        temp_params_name_hat: str = "mod_sig",
        global_param_names: Optional[List[str]] = None,
        global_param_names_hat: Optional[List[str]] = None,
        use_p_loss: bool = False,
        log_envelope: bool = True,
        fad_model_names: Optional[List[str]] = None,
        run_name: Optional[str] = None,
    ):
        super().__init__()
        if synth is None:
            synth = AcidSynthLPBiquad(ac)
        if synth_hat is None:
            synth_hat = AcidSynthLPBiquad(ac)
        with suppress(Exception):
            assert synth_hat.interp_logits == synth_eval.interp_logits
        with suppress(Exception):
            assert synth_hat.interp_coeff == synth.interp_coeff
        if global_param_names is None:
            global_param_names = []
        if global_param_names_hat is None:
            global_param_names_hat = []
        if fad_model_names is None:
            fad_model_names = []
        if run_name is None:
            self.run_name = f"run__{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
        else:
            self.run_name = run_name
        log.info(f"Run name: {self.run_name}")

        self.ac = ac
        self.model = model
        self.loss_func = loss_func
        self.spectral_visualizer = spectral_visualizer
        self.synth = synth
        self.synth_hat = synth_hat
        self.synth_eval = synth_eval
        self.temp_params_name = temp_params_name
        self.temp_params_name_hat = temp_params_name_hat
        self.global_param_names = global_param_names
        self.global_param_names_hat = global_param_names_hat
        self.use_p_loss = use_p_loss
        self.log_envelope = log_envelope
        self.fad_model_names = fad_model_names

        self.loss_name = self.loss_func.__class__.__name__
        self.esr = ESRLoss()
        self.l1 = nn.L1Loss()

        self.audio_metrics = nn.ModuleDict()
        self.audio_metrics["mss"] = auraloss.freq.MultiResolutionSTFTLoss()
        self.audio_metrics["mel_stft"] = auraloss.freq.MelSTFTLoss(
            sample_rate=self.ac.sr,
            fft_size=spectral_visualizer.n_fft,
            hop_size=spectral_visualizer.hop_len,
            win_length=spectral_visualizer.n_fft,
            n_mels=spectral_visualizer.n_mels,
        )
        self.audio_metrics["mfcc"] = MFCCL1(
            sr=self.ac.sr,
            log_mels=True,
            n_fft=spectral_visualizer.n_fft,
            hop_len=spectral_visualizer.hop_len,
            n_mels=spectral_visualizer.n_mels,
        )

        self.global_n = 0
        self.test_out_dicts = []

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        return super().load_state_dict(state_dict, strict=False)

    def on_train_start(self) -> None:
        self.global_n = 0

    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T]:
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        phase = batch["phase"]
        phase_hat = batch["phase_hat"]

        batch_size = f0_hz.size(0)
        assert f0_hz.shape == (batch_size,)
        assert note_on_duration.shape == (batch_size,)
        assert phase.shape == (batch_size, 1)
        assert phase_hat.shape == (batch_size, 1)

        filter_args = {}
        if self.temp_params_name is not None:
            temp_params = batch[self.temp_params_name]
            assert temp_params.size(0) == batch_size
            assert temp_params.ndim == 3
            filter_args[self.temp_params_name] = temp_params

        global_params_0to1 = {p: batch[f"{p}_0to1"] for p in self.global_param_names}
        global_params = {
            k: self.ac.convert_from_0to1(k, v) for k, v in global_params_0to1.items()
        }
        # TODO(cm): generalize
        if "q" in self.global_param_names:
            q_0to1 = batch["q_0to1"]
            q_mod_sig = q_0to1.unsqueeze(-1)
            filter_args["q_mod_sig"] = q_mod_sig

        # Generate ground truth wet audio
        with tr.no_grad():
            synth_out = self.synth(
                self.ac.n_samples,
                f0_hz,
                note_on_duration,
                phase,
                filter_args,
                global_params,
            )
            dry = synth_out["dry"]
            wet = synth_out["wet"]
            envelope = synth_out["envelope"]
            assert dry.shape == wet.shape == (batch_size, self.ac.n_samples)

        batch.update(global_params)
        batch["dry"] = dry
        batch["wet"] = wet
        batch["envelope"] = envelope
        return batch

    def step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        batch_size = batch["f0_hz"].size(0)
        if stage == "train":
            self.global_n = (
                self.global_step * self.trainer.accumulate_grad_batches * batch_size
            )
        self.log(f"global_n", float(self.global_n), sync_dist=True)

        batch = self.preprocess_batch(batch)

        # Get mandatory params
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        wet = batch["wet"]
        phase_hat = batch["phase_hat"]
        global_params_0to1 = {
            p_name: batch[f"{p_name}_0to1"]
            for p_name in self.global_param_names
            if f"{p_name}_0to1" in batch
        }
        global_params = {
            p_name: batch[p_name]
            for p_name in self.global_param_names
            if p_name in batch
        }

        # Get optional params
        dry = batch.get("dry")
        envelope = batch.get("envelope")
        temp_params = None
        if self.temp_params_name is not None:
            temp_params = batch[self.temp_params_name]

        # Perform model forward pass
        model_in = wet.unsqueeze(1)
        model_out = self.model(model_in)
        filter_args_hat = {}

        # Postprocess temp_params_hat
        temp_params_hat = model_out[self.temp_params_name_hat]
        assert temp_params_hat.ndim == 3
        filter_args_hat[self.temp_params_name_hat] = temp_params_hat

        # Postprocess global_params_hat
        global_params_0to1_hat = {}
        global_params_hat = {}
        for p_name in self.global_param_names_hat:
            p_val_0to1_hat = model_out[f"{p_name}_0to1"]
            p_val_hat = self.ac.convert_from_0to1(p_name, p_val_0to1_hat)
            global_params_0to1_hat[p_name] = p_val_0to1_hat
            global_params_hat[p_name] = p_val_hat
            if not self.ac.is_fixed(p_name) and p_name in global_params_0to1:
                p_val_0to1 = global_params_0to1[p_name]
                with tr.no_grad():
                    p_val_l1 = self.l1(p_val_0to1_hat, p_val_0to1)
                self.log(f"{stage}/{p_name}_l1", p_val_l1, prog_bar=False)

        # Postprocess log_spec_wet
        log_spec_wet = model_out.get("log_spec_wet")
        if log_spec_wet is None:
            log_spec_wet = self.spectral_visualizer(model_in)
        assert log_spec_wet.ndim == 4
        log_spec_wet = log_spec_wet.squeeze(1)

        # Postprocess q_hat TODO(cm): generalize
        if "q" in self.global_param_names_hat:
            q_0to1_hat = global_params_0to1_hat["q"]
            q_mod_sig_hat = q_0to1_hat.unsqueeze(-1)
            filter_args_hat["q_mod_sig"] = q_mod_sig_hat

        # Generate audio x_hat
        synth_out_hat = self.synth_hat(
            self.ac.n_samples,
            f0_hz,
            note_on_duration,
            phase_hat,
            filter_args_hat,
            global_params_hat,
        )
        wet_hat = synth_out_hat["wet"]
        with tr.no_grad():
            log_spec_wet_hat = self.spectral_visualizer(wet_hat.unsqueeze(1)).squeeze(1)

        # TODO(cm): refactor
        if envelope is None:
            envelope = synth_out_hat.get("envelope")

        # Compute loss
        if self.use_p_loss:
            temp_params_hat = util.linear_interpolate_dim(
                temp_params_hat, temp_params.size(1), dim=1, align_corners=True
            )
            assert temp_params.shape == temp_params_hat.shape
            loss = self.loss_func(temp_params, temp_params)

            for p_name in self.global_param_names:
                if not self.ac.is_fixed(p_name):
                    p_val = global_params_0to1[p_name]
                    p_val_hat = global_params_0to1_hat[p_name]
                    p_loss = self.loss_func(p_val_hat, p_val)
                    self.log(
                        f"{stage}/ploss_{self.loss_name}_{p_name}",
                        p_loss,
                        prog_bar=False,
                    )
                    loss += p_loss
            self.log(f"{stage}/ploss_{self.loss_name}", loss, prog_bar=False)
        else:
            loss = self.loss_func(wet_hat.unsqueeze(1), wet.unsqueeze(1))
            self.log(f"{stage}/audio_{self.loss_name}", loss, prog_bar=False)

        self.log(f"{stage}/loss", loss, prog_bar=False)

        # Log audio metrics
        audio_metrics_hat = {}
        for metric_name, metric in self.audio_metrics.items():
            with tr.no_grad():
                audio_metric = metric(wet_hat.unsqueeze(1), wet.unsqueeze(1))
            audio_metrics_hat[metric_name] = audio_metric
            self.log(f"{stage}/audio_{metric_name}", audio_metric, prog_bar=False)

        # Log temp_params metrics
        temp_param_metrics = {}
        if (
            self.temp_params_name is not None
            and self.temp_params_name == self.temp_params_name_hat
        ):
            with tr.no_grad():
                temp_params_hat = util.linear_interpolate_dim(
                    temp_params_hat, temp_params.size(1), dim=1, align_corners=True
                )
                assert temp_params.shape == temp_params_hat.shape
                temp_params_l1 = self.l1(temp_params_hat, temp_params)
                temp_params_esr = self.esr(temp_params_hat, temp_params)
            temp_param_metrics[f"{self.temp_params_name}_l1"] = temp_params_l1
            temp_param_metrics[f"{self.temp_params_name}_esr"] = temp_params_esr
            self.log(
                f"{stage}/{self.temp_params_name}_l1", temp_params_l1, prog_bar=False
            )
            self.log(
                f"{stage}/{self.temp_params_name}_esr", temp_params_esr, prog_bar=False
            )

        # Log eval synth metrics if applicable
        wet_eval = None
        audio_metrics_eval = {}
        if stage != "train" and self.synth_eval is not None:
            # Generate audio x_eval
            try:
                synth_out_eval = self.synth_eval(
                    self.ac.n_samples,
                    f0_hz,
                    note_on_duration,
                    phase_hat,
                    filter_args_hat,
                    global_params_hat,
                )
                wet_eval = synth_out_eval["wet"]
                for metric_name, metric in self.audio_metrics.items():
                    with tr.no_grad():
                        audio_metric = metric(wet_eval.unsqueeze(1), wet.unsqueeze(1))
                    audio_metrics_eval[metric_name] = audio_metric
                    self.log(
                        f"{stage}/audio_{metric_name}_eval",
                        audio_metric,
                        prog_bar=False,
                    )
            except Exception as e:
                log.error(f"Error in eval synth: {e}")

        global_params_hat = {f"{k}_hat": v for k, v in global_params_hat.items()}
        audio_metrics_hat = {f"{k}_hat": v for k, v in audio_metrics_hat.items()}
        audio_metrics_eval = {f"{k}_eval": v for k, v in audio_metrics_eval.items()}
        out_dict = {
            "loss": loss,
            "dry": dry,
            "wet": wet,
            "wet_hat": wet_hat,
            "wet_eval": wet_eval,
            "envelope": envelope,
            "log_spec_wet": log_spec_wet,
            "log_spec_wet_hat": log_spec_wet_hat,
            "temp_params": temp_params,
            "temp_params_hat": temp_params_hat,
        }
        out_dict.update(temp_param_metrics)
        out_dict.update(global_params)
        out_dict.update(global_params_hat)
        out_dict.update(audio_metrics_hat)
        out_dict.update(audio_metrics_eval)
        return out_dict

    def training_step(self, batch: Dict[str, T], batch_idx: int) -> Dict[str, T]:
        return self.step(batch, stage="train")

    def validation_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        return self.step(batch, stage="val")

    def test_step(self, batch: Dict[str, T], stage: str) -> Dict[str, T]:
        out = self.step(batch, stage="test")
        self.test_out_dicts.append(out)
        return out

    def on_test_epoch_end(self) -> None:
        tsv_rows = []

        test_metrics = ["loss"]
        for metric_name in self.audio_metrics:
            test_metrics.append(f"{metric_name}_hat")
            test_metrics.append(f"{metric_name}_eval")

        for metric_name in test_metrics:
            metric_values = [d.get(metric_name) for d in self.test_out_dicts]
            if any(v is None for v in metric_values):
                log.warning(f"Skipping test metric: {metric_name}")
                continue
            metric_values = tr.stack(metric_values, dim=0)
            assert metric_values.ndim == 1
            metric_mean = metric_values.mean()
            metric_std = metric_values.std()
            metric_ci95 = 1.96 * scipy.stats.sem(metric_values.numpy())
            self.log(f"test/{metric_name}", metric_mean, prog_bar=False)
            tsv_rows.append(
                [
                    metric_name,
                    metric_mean.item(),
                    metric_std.item(),
                    metric_ci95,
                    metric_values.size(0),
                    metric_values.numpy(),
                ]
            )

        for fad_model_name in self.fad_model_names:
            fad_hat_values = []
            fad_eval_values = []
            for out in self.test_out_dicts:
                wet = out["wet"]
                wet_hat = out["wet_hat"]
                wet_eval = out.get("wet_eval")

                fad_wet_dir = os.path.join(OUT_DIR, f"{self.run_name}__fad_wet")
                fad_wet_hat_dir = os.path.join(OUT_DIR, f"{self.run_name}__fad_wet_hat")
                save_and_concat_fad_audio(
                    self.ac.sr,
                    wet,
                    fad_wet_dir,
                    fade_n_samples=self.spectral_visualizer.hop_len,
                )
                save_and_concat_fad_audio(
                    self.ac.sr,
                    wet_hat,
                    fad_wet_hat_dir,
                    fade_n_samples=self.spectral_visualizer.hop_len,
                )
                clean_up_baseline = True
                if wet_eval is not None:
                    clean_up_baseline = False
                fad_hat = calc_fad(
                    fad_model_name,
                    baseline_dir=fad_wet_dir,
                    eval_dir=fad_wet_hat_dir,
                    clean_up_baseline=clean_up_baseline,
                    clean_up_eval=True,
                )
                fad_hat_values.append(fad_hat)
                if wet_eval is not None:
                    fad_wet_eval_dir = os.path.join(
                        OUT_DIR, f"{self.run_name}__fad_wet_eval"
                    )
                    save_and_concat_fad_audio(
                        self.ac.sr,
                        wet_eval,
                        fad_wet_eval_dir,
                        fade_n_samples=self.spectral_visualizer.hop_len,
                    )
                    fad_eval = calc_fad(
                        fad_model_name,
                        baseline_dir=fad_wet_dir,
                        eval_dir=fad_wet_eval_dir,
                        clean_up_baseline=True,
                        clean_up_eval=True,
                    )
                    fad_eval_values.append(fad_eval)

            fad_hat_mean = np.mean(fad_hat_values)
            fad_hat_std = np.std(fad_hat_values)
            fad_hat_ci95 = 1.96 * scipy.stats.sem(fad_hat_values)
            self.log(f"test/fad_{fad_model_name}_hat", fad_hat_mean, prog_bar=False)
            tsv_rows.append(
                [
                    f"fad_{fad_model_name}_hat",
                    fad_hat_mean,
                    fad_hat_std,
                    fad_hat_ci95,
                    len(fad_hat_values),
                    fad_hat_values,
                ]
            )
            if fad_eval_values:
                fad_eval_mean = np.mean(fad_eval_values)
                fad_eval_std = np.std(fad_eval_values)
                fad_eval_ci95 = 1.96 * scipy.stats.sem(fad_eval_values)
                self.log(
                    f"test/fad_{fad_model_name}_eval", fad_eval_mean, prog_bar=False
                )
                tsv_rows.append(
                    [
                        f"fad_{fad_model_name}_eval",
                        fad_eval_mean,
                        fad_eval_std,
                        fad_eval_ci95,
                        len(fad_eval_values),
                        fad_eval_values,
                    ]
                )

        tsv_path = os.path.join(OUT_DIR, f"{self.run_name}__test.tsv")
        if os.path.exists(tsv_path):
            log.warning(f"Overwriting existing TSV file: {tsv_path}")
        df = pd.DataFrame(
            tsv_rows, columns=["metric_name", "mean", "std", "ci95", "n", "values"]
        )
        df.to_csv(tsv_path, sep="\t", index=False)

        # H = tr.cat([out["H"] for out in self.test_outs], dim=0)
        # H = H.detach().cpu()
        # H_path = os.path.join(OUT_DIR, f"{self.run_name}__H.pt")
        # tr.save(H, H_path)
        # log.info(f"Saved H to: {H_path}")


class PreprocLightningModule(AcidDDSPLightingModule):
    def preprocess_batch(self, batch: Dict[str, T]) -> Dict[str, T]:
        wet = batch["wet"]
        f0_hz = batch["f0_hz"]
        note_on_duration = batch["note_on_duration"]
        assert wet.ndim == 2
        assert wet.size(1) == self.ac.n_samples
        batch_size = wet.size(0)
        assert f0_hz.shape == (batch_size,)
        assert note_on_duration.shape == (batch_size,)
        return batch

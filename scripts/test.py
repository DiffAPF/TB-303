import logging
import os
from tempfile import NamedTemporaryFile

import yaml
from tqdm import tqdm

from cli import CustomLightningCLI
from paths import MODELS_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    model_names = [
        "cnn_mss_coeff_td__abstract_303_48k__6k__4k_min__epoch_199_step_1200",
        "cnn_mss_coeff_fs_128__abstract_303_48k__6k__4k_min__epoch_143_step_864",
        "cnn_mss_coeff_fs_256__abstract_303_48k__6k__4k_min__epoch_159_step_960",
        "cnn_mss_coeff_fs_512__abstract_303_48k__6k__4k_min__epoch_191_step_1152",
        "cnn_mss_coeff_fs_1024__abstract_303_48k__6k__4k_min__epoch_167_step_1008",
        "cnn_mss_coeff_fs_2048__abstract_303_48k__6k__4k_min__epoch_159_step_960",
        "cnn_mss_coeff_fs_4096__abstract_303_48k__6k__4k_min__epoch_167_step_1008",
        "cnn_mss_lp_td__abstract_303_48k__6k__4k_min__epoch_183_step_1104",
        "cnn_mss_lp_fs_128__abstract_303_48k__6k__4k_min__epoch_183_step_1104",
        "cnn_mss_lp_fs_256__abstract_303_48k__6k__4k_min__epoch_183_step_1104",
        "cnn_mss_lp_fs_512__abstract_303_48k__6k__4k_min__epoch_183_step_1104",
        "cnn_mss_lp_fs_1024__abstract_303_48k__6k__4k_min__epoch_135_step_816",
        "cnn_mss_lp_fs_2048__abstract_303_48k__6k__4k_min__epoch_135_step_816",
        "cnn_mss_lp_fs_4096__abstract_303_48k__6k__4k_min__epoch_135_step_816",
        "cnn_mss_lstm_64__abstract_303_48k__6k__4k_min__epoch_175_step_1056",
    ]
    fad_model_names = [
        "vggish",
    ]

    model_dir = MODELS_DIR
    for model_name in tqdm(model_names):
        config_path = os.path.join(model_dir, model_name, f"config.yaml")
        ckpt_path = os.path.join(
            model_dir, model_name, f"checkpoints/{model_name}.ckpt"
        )
        with open(config_path, "r") as in_f:
            config = yaml.safe_load(in_f)
        if config.get("ckpt_path"):
            assert os.path.abspath(config["ckpt_path"]) == os.path.abspath(ckpt_path)

        test_set_n = 68
        config["custom"]["cpu_batch_size"] = test_set_n
        config["data"]["init_args"]["batch_size"] = test_set_n

        config["model"]["init_args"]["fad_model_names"] = fad_model_names
        config["model"]["init_args"]["run_name"] = model_name
        # Modify this to reduce the number of N for confidence intervals
        config["data"]["init_args"]["n_phases_per_file"] = 20

        tmp_config_file = NamedTemporaryFile()
        with open(tmp_config_file.name, "w") as out_f:
            yaml.dump(config, out_f)
            cli = CustomLightningCLI(
                args=["test", "--config", tmp_config_file.name, "--ckpt_path", ckpt_path],
                trainer_defaults=CustomLightningCLI.trainer_defaults,
            )

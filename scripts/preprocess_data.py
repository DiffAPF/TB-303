import logging
import os

import numpy as np
import pretty_midi
import torch as tr
import torchaudio
from tqdm import tqdm

from paths import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def concat_wav_files(dir_path: str, out_path: str, sr: int = 44100) -> None:
    """Concatenate all wav files in a directory into a single wav file."""
    wav_files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]
    log.info(f"Found {len(wav_files)} wav files in {dir_path}")
    wav_files.sort()
    wav_data = []
    for wav_file in tqdm(wav_files):
        wav_file_path = os.path.join(dir_path, wav_file)
        wav, wav_sr = torchaudio.load(wav_file_path)
        if wav_sr == sr:
            wav_data.append(wav)

    wav_data = tr.cat(wav_data, dim=1)
    torchaudio.save(out_path, wav_data, sr)


def extract_notes_from_midi(
    audio_path: str,
    midi_path: str,
    out_dir: str,
    min_n_samples: int = 4000,
    max_n_samples: int = 6000,
) -> None:
    """Extract notes from a midi file and save them as wav files."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    wav, sr = torchaudio.load(audio_path)
    assert len(midi_data.instruments) == 1
    midi_data = midi_data.instruments[0]

    all_note_n_samples = []
    n_saved_notes = 0
    all_note_min = []
    all_note_max = []
    all_midi_f0 = []
    for idx, note in tqdm(enumerate(midi_data.notes)):
        start_time = note.start
        end_time = note.end
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        note_n_samples = end_sample - start_sample
        all_note_n_samples.append(note_n_samples)
        if note_n_samples < min_n_samples:
            continue
        note_wav = wav[:, start_sample:end_sample]
        if note_n_samples > max_n_samples:
            note_wav = note_wav[:, :max_n_samples]
        elif note_n_samples < max_n_samples:
            pad = tr.zeros((1, max_n_samples - note_n_samples))
            note_wav = tr.cat([note_wav, pad], dim=1)
        note_min = note_wav.min().item()
        note_max = note_wav.max().item()
        all_note_min.append(note_min)
        all_note_max.append(note_max)
        midi_f0 = note.pitch
        all_midi_f0.append(midi_f0)
        note_wav_path = os.path.join(
            out_dir,
            f"note_{idx:03d}__{midi_f0}__{min(note_n_samples, max_n_samples)}__.wav",
        )
        torchaudio.save(note_wav_path, note_wav, sr)
        n_saved_notes += 1

    log.info(f"Saved {len(all_note_n_samples)} notes to {out_dir}")
    log.info(f"Min note length: {min(all_note_n_samples)} samples")
    log.info(f"Max note length: {max(all_note_n_samples)} samples")
    log.info(
        f"Average note length: {sum(all_note_n_samples) / len(all_note_n_samples)} samples"
    )
    log.info(f"STD note length: {np.std(all_note_n_samples)} samples")
    log.info(f"Total note length: {sum(all_note_n_samples)} samples")
    log.info(f"Total note length: {sum(all_note_n_samples) / sr} seconds")

    log.info(f"Saved {n_saved_notes} notes to {out_dir}")
    log.info(f"Total saved note length: {n_saved_notes * max_n_samples} samples")
    log.info(f"Total saved note length: {n_saved_notes * max_n_samples / sr} seconds")

    log.info(f"Min note min: {min(all_note_min)}")
    log.info(f"Max note min: {max(all_note_min)}")
    log.info(f"Average note min: {sum(all_note_min) / len(all_note_min)}")
    log.info(f"STD note min: {np.std(all_note_min)}")
    log.info(f"Min note max: {min(all_note_max)}")
    log.info(f"Max note max: {max(all_note_max)}")
    log.info(f"Average note max: {sum(all_note_max) / len(all_note_max)}")
    log.info(f"STD note max: {np.std(all_note_max)}")

    log.info(f"Min midi f0: {min(all_midi_f0)}")
    log.info(f"Max midi f0: {max(all_midi_f0)}")


if __name__ == "__main__":
    # data_dir = os.path.join(DATA_DIR, "samplescience_abstract_303")
    # out_path = os.path.join(DATA_DIR, "all.wav")
    # concat_wav_files(data_dir, out_path)

    audio_path = os.path.join(DATA_DIR, "abstract_303_all_48k.wav")
    midi_path = os.path.join(DATA_DIR, "abstract_303_all.mid")
    out_dir = os.path.join(DATA_DIR, "abstract_303_48k__6k__4k_min")
    extract_notes_from_midi(audio_path, midi_path, out_dir)

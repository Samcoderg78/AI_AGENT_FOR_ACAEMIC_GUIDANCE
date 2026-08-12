"""Multilingual Speech Recognition - Whisper Small inference pipeline."""

import argparse
from pathlib import Path

import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_NAME = "openai/whisper-small"


def load_audio(file_path: Path):
    """Load audio and resample it to 16 kHz."""
    audio, _ = librosa.load(file_path, sr=16000)
    return audio


class WhisperTranscriber:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: Path) -> str:
        audio = load_audio(audio_path)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)

        return self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].lower().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV containing an `audio` column")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    test_df = pd.read_csv(args.csv)
    audio_dir = Path(args.audio_dir)
    transcriber = WhisperTranscriber()
    predictions = []

    for audio_file in tqdm(test_df["audio"], desc="Transcribing"):
        audio_path = audio_dir / audio_file
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        predictions.append(transcriber.transcribe(audio_path))

    pd.DataFrame({"audio": test_df["audio"], "text": predictions}).to_csv(
        args.output, index=False
    )
    print(f"Saved predictions to {args.output}")
    print(f"Device used: {transcriber.device}")


if __name__ == "__main__":
    main()

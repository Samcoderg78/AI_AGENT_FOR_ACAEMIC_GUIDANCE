# Multilingual Speech Recognition

End-to-end multilingual Automatic Speech Recognition (ASR) pipeline built with **OpenAI Whisper Small, PyTorch, Hugging Face Transformers, and Librosa**.

This project was developed as part of the IIT Madras DLP NPPE-2: Multilingual Speech Recognition competition.

## Project Overview

The objective was to convert multilingual audio recordings into text while working under constrained compute resources using Kaggle notebooks.

The pipeline:

1. Loads competition metadata from CSV files.
2. Loads audio recordings and resamples them to **16 kHz**.
3. Uses the **OpenAI Whisper Small** pretrained speech-to-text model.
4. Processes audio through the Hugging Face `WhisperProcessor`.
5. Runs GPU-accelerated inference when CUDA is available.
6. Decodes model predictions into text.
7. Generates a submission CSV containing audio filenames and predicted transcriptions.

## Dataset

The competition dataset contained approximately **2,100 audio files** and associated metadata.

The audio files are **not included in this repository**. They were accessed through the Kaggle competition environment.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- OpenAI Whisper
- Librosa
- Pandas
- NumPy
- tqdm
- CUDA / GPU acceleration

## Evaluation

The competition used **Word Error Rate (WER)** as the evaluation metric. WER measures substitutions, deletions, and insertions; lower is better.

## Repository Structure

```text
multilingual-speech-recognition/
├── README.md
├── requirements.txt
├── inference.py
├── .gitignore
└── notebooks/
    └── multilingual_speech_recognition.ipynb
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

For local inference:

```bash
python inference.py --csv test.csv --audio-dir competition_data --output submission.csv
```

The original project was executed in the Kaggle environment where the competition dataset was mounted.

## Key Learning

This project provided hands-on experience with Automatic Speech Recognition, multilingual speech processing, audio preprocessing, transformer-based speech models, Hugging Face inference, GPU-accelerated deep learning, and WER-based evaluation.

## Limitations and Future Improvements

The implementation primarily uses pretrained Whisper Small inference. It does not include model fine-tuning, LoRA/QLoRA training, or extensive data augmentation.

Potential improvements include fine-tuning Whisper on the training set, language-specific preprocessing, audio augmentation, decoding optimization, language-wise WER analysis, and comparison with other ASR architectures.

## Author

**Saurabh Yadav** — B.S. Data Science & Applications, IIT Madras

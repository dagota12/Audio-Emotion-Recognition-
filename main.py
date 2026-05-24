import librosa
import torch
from typing import Tuple
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    pipeline,
    Pipeline,
)

# ============================================================================
# LOCAL EMOTION RECOGNITION MODEL - INFERENCE SCRIPT
# ============================================================================
# This module loads a fine‑tuned Wav2Vec 2.0 model and uses it to classify
# emotions from spoken audio files. The model was trained on the RAVDESS
# dataset to recognize 8 emotion classes from raw audio waveforms.
# ============================================================================

# Configuration
MODEL_PATH: str = "./final_emotion_model_85pct"
"""Path to the locally saved / unzipped Hugging Face model folder."""

DEVICE: int = -1
"""Force CPU usage for local testing (-1 = CPU, 0+ = GPU device index)."""

def load_components(model_path: str = MODEL_PATH, device: int = DEVICE) -> Tuple[Pipeline, AutoFeatureExtractor, AutoModelForAudioClassification]:
    """Load the feature extractor, model, and create the classification pipeline.

    Args:
        model_path: Directory containing the saved model.
        device: Device index for the pipeline (‑1 for CPU).
    Returns:
        A tuple of (pipeline, feature_extractor, model).
    """
    print("Loading emotion recognition model from disk...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    classifier = pipeline(
        task="audio-classification",
        model=model,
        feature_extractor=feature_extractor,
        device=device,
    )
    return classifier, feature_extractor, model

def predict_emotion(audio_file: str, classifier: Pipeline) -> None:
    """Predict emotion from an audio file using the provided classifier.

    Args:
        audio_file: Path to the .wav audio file to analyze.
        classifier: An instantiated Hugging Face audio‑classification pipeline.
    """
    # Load audio file and resample to 16kHz (required by Wav2Vec 2.0)
    speech, sample_rate = librosa.load(audio_file, sr=16000)

    # Run emotion classification using the fine‑tuned model
    results = classifier(speech)

    # Display prediction results with formatted output
    print("\n========== Emotion Prediction ========== ")
    print(f"Audio File : {audio_file}")
    print(f"Sample Rate: {sample_rate} Hz")
    print("----------------------------------------")
    for result in results:
        label = result["label"]
        score = result["score"]
        print(f"{label:<15} -> {score:.4f}")
    print("========================================\n")

def main() -> None:
    """Entry point for the CLI.

    Uses argparse to accept an audio file path and optional arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict emotion from a .wav file using a fine‑tuned Wav2Vec 2.0 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the input .wav audio file.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_PATH,
        help="Directory containing the saved Hugging Face model.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=DEVICE,
        help="Device index for inference (‑1 = CPU, 0 = first GPU).",
    )
    args = parser.parse_args()

    classifier, _, _ = load_components(model_path=args.model_dir, device=args.device)
    predict_emotion(args.audio_file, classifier)

if __name__ == "__main__":
    main()

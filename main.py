import librosa
import torch
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    pipeline,
)


# Local Emotion Recognition Model Configuration


# Path to the locally saved / unzipped Hugging Face model
MODEL_PATH = "./final_emotion_model_85pct"

# Force CPU usage for local testing
DEVICE = -1

print("Loading emotion recognition model from disk...")

# Load feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

# Create audio classification pipeline
classifier = pipeline(
    task="audio-classification",
    model=model,
    feature_extractor=feature_extractor,
    device=DEVICE,
)

def predict_emotion(audio_file: str):
    """
    Predict emotion labels from an audio file.

    Steps:
    1. Load audio using librosa
    2. Resample audio to 16kHz
    3. Run inference using Hugging Face pipeline
    4. Print predicted emotions with confidence scores
    """

    # Load audio and convert sample rate to 16kHz
    speech, sample_rate = librosa.load(audio_file, sr=16000)

    # Run emotion classification
    results = classifier(speech)

    # Display prediction results
    print(f"\n========== Emotion Prediction ==========")
    print(f"Audio File : {audio_file}")
    print(f"Sample Rate: {sample_rate} Hz")
    print("----------------------------------------")

    for result in results:
        label = result["label"]
        score = result["score"]

        print(f"{label:<15} -> {score:.4f}")

    print("========================================\n")


if __name__ == "__main__":
    # Replace with any audio file you want to test
    TEST_AUDIO_FILE = "happy-2.wav"

    predict_emotion(TEST_AUDIO_FILE)

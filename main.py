import librosa
import torch
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    pipeline,
)


# ============================================================================
# LOCAL EMOTION RECOGNITION MODEL - INFERENCE SCRIPT
# ============================================================================
# This module loads a fine-tuned Wav2Vec 2.0 model and uses it to classify
# emotions from spoken audio files. The model was trained on the RAVDESS
# dataset to recognize 8 emotion classes from raw audio waveforms.
# ============================================================================


# Configuration
MODEL_PATH = "./final_emotion_model_85pct"
"""Path to the locally saved / unzipped Hugging Face model folder."""

DEVICE = -1
"""Force CPU usage for local testing (-1 = CPU, 0+ = GPU device index)."""

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
    Predict emotion from an audio file using the fine-tuned Wav2Vec 2.0 model.
    
    Args:
        audio_file (str): Path to the .wav audio file to analyze.
        
    Predicts emotion by:
        1. Loading and resampling audio to 16 kHz (model input requirement)
        2. Running the Hugging Face audio classification pipeline
        3. Displaying ranked emotion predictions with confidence scores
    """
    # Load audio file and resample to 16kHz (required by Wav2Vec 2.0)
    speech, sample_rate = librosa.load(audio_file, sr=16000)

    # Run emotion classification using the fine-tuned model
    results = classifier(speech)

    # Display prediction results with formatted output
    print(f"\n========== Emotion Prediction ==========")
    print(f"Audio File : {audio_file}")
    print(f"Sample Rate: {sample_rate} Hz")
    print("----------------------------------------")

    for result in results:
        label = result["label"]
        score = result["score"]
        # Display emotion label and confidence score (0-1 scale)
        print(f"{label:<15} -> {score:.4f}")

    print("========================================\n")


if __name__ == "__main__":
    # ========================================================================
    # ENTRY POINT: Test the emotion prediction on a sample audio file
    # ========================================================================
    # Replace TEST_AUDIO_FILE with any .wav file path to predict its emotion.
    # The model outputs confidence scores for all 8 emotion classes from
    # RAVDESS: neutral, calm, happy, sad, angry, fearful, disgust, surprised
    # ========================================================================
    
    # Path to test audio file (replace with your own)
    TEST_AUDIO_FILE = "happy-2.wav"

    # Run prediction and display results
    predict_emotion(TEST_AUDIO_FILE)

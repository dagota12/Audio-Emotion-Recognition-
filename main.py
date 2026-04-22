import librosa
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, pipeline

# 1. Path to your unzipped model folder
model_path = "./final_emotion_model_85pct"

# 2. Load the local model and feature extractor
print("Loading model from disk...")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = AutoModelForAudioClassification.from_pretrained(model_path)

# 3. Create the classifier pipeline
# We set device=-1 to use CPU (standard for local testing)
classifier = pipeline(
    "audio-classification", 
    model=model, 
    feature_extractor=feature_extractor,
    device=-1 
)

def predict_emotion(audio_file):
    # Load and resample to 16kHz
    speech, _ = librosa.load(audio_file, sr=16000)
    
    # Run classification
    results = classifier(speech)
    
    print(f"\n--- Results for {audio_file} ---")
    for res in results:
        print(f"{res['label']}: {res['score']:.4f}")

if __name__ == "__main__":
    # Change this to whatever file you want to test locally
    test_file = "neutral.wav" 
    predict_emotion(test_file)
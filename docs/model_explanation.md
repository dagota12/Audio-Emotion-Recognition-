# How the Emotion Recognition Model Works

This document explains in detail how the trained model processes raw speech audio and outputs emotion predictions. Understanding this workflow helps in debugging and improving the inference pipeline.

---

## 1. Input Audio

The model starts with a `.wav` file, such as `happy-2.wav` or `angry.wav`. 

The audio file contains the spoken voice signal, which is the raw data we want to classify. No preprocessing happens at this stage—the file is simply a digital representation of speech.

## 2. Loading and Resampling

In `main.py`, the audio is loaded with `librosa` and resampled to **16 kHz**.

**Why this matters:** The pretrained Wav2Vec 2.0 model expects exactly 16,000 samples per second. If the audio has a different sample rate, the input would not match what the model was trained to understand, leading to incorrect predictions.

### Resampling Step
- Input: Raw audio at any sample rate (e.g., 44.1 kHz, 48 kHz)
- Output: Normalized audio at 16 kHz
- Tool: `librosa.load(audio_file, sr=16000)`

## 3. Feature Extraction

The script uses a Hugging Face feature extractor from the saved model folder. 

**Purpose:** The feature extractor prepares the waveform into a format that can be fed into the neural network.

### Feature Extraction Pipeline
- **Input:** Raw audio waveform (numpy array of audio samples)
- **Processing:** Normalizes amplitude, applies padding/truncation for consistent shapes
- **Output:** Tensor ready for model input
- **Tool:** `AutoFeatureExtractor.from_pretrained(MODEL_PATH)`

At this stage, the audio is converted into the numeric form the model can process. In training, the same preprocessing was applied to ensure every clip had a consistent input shape.

## 4. Wav2Vec 2.0 Backbone

The core model is **Wav2Vec 2.0**, a transformer-based speech model that was pretrained on large amounts of unlabeled audio data.

### Why Wav2Vec 2.0?
It works exceptionally well for this project because it can learn directly from the structure of voice itself:

- **Tone**: The color and quality of the voice
- **Pitch**: How high or low the voice sounds  
- **Rhythm**: The timing and pacing of speech
- **Pauses**: Silence and hesitation patterns
- **Emphasis**: Stress on particular words or syllables

**Key Advantage:** Instead of manually engineering audio features (like spectral components or MFCCs), the model learns useful patterns directly from raw waveforms through deep learning.

## 5. Emotion Classification Head

The pretrained speech model is fine-tuned with a specialized classification head for the emotion labels from RAVDESS.

### Architecture Modification
- **Pretrained Backbone:** Wav2Vec 2.0 (learns general speech patterns)
- **Fine-tuning Head:** Added classification layer with 8 output neurons
- **Output Classes:**
  - Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

**Key Point:** The model does not just understand speech in general. Through fine-tuning on the RAVDESS dataset, it learns to map speech patterns to specific emotion classes. The backbone weights are slightly adjusted to specialize in emotion recognition.

## 6. Prediction Pipeline

The local inference script uses the Hugging Face `audio-classification` pipeline.

The pipeline handles three things:

1. preprocessing the audio
2. running the model forward pass
3. converting the output logits into human-readable labels and confidence scores

This is why the script is short but still produces a full ranked prediction list.

## 7. Output Interpretation

The result is a list of emotions with confidence values. The top entry is the model’s best prediction.

For example, the screenshot in [example-run.png](example-run.png) shows the model predicting `happy` as the strongest label, followed by other emotions with lower confidence values.

## 8. Why The Model Was Trained This Way

The model was fine-tuned rather than trained from scratch because transfer learning gives strong results with much less data.

The project also used a longer audio window during training so the model could see more context. Emotion is often clearer when the model can hear a few seconds of speech instead of only a very short clip.

## 9. Summary

In short, the workflow is:

1. load a `.wav` file
2. resample it to 16 kHz
3. convert it into model-ready features
4. pass it through the fine-tuned Wav2Vec 2.0 model
5. output the most likely emotion and confidence scores

That is how the project turns raw voice audio into an emotion prediction.

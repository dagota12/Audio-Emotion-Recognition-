# How the Emotion Recognition Model Works

This document explains how the trained model turns raw speech audio into emotion predictions.

## 1. Input Audio

The model starts with a `.wav` file, such as `happy-2.wav` or `angry.wav`. The audio file contains the spoken voice signal, which is the raw data we want to classify.

## 2. Loading and Resampling

In `main.py`, the audio is loaded with `librosa` and resampled to **16 kHz**.

This step matters because the pretrained Wav2Vec 2.0 model expects 16,000 samples per second. If the audio has a different sample rate, the input would not match what the model was trained to understand.

## 3. Feature Extraction

The script uses a Hugging Face feature extractor from the saved model folder. The feature extractor prepares the waveform so it can be fed into the neural network.

At this stage, the audio is converted into the numeric form the model can process. In training, the same kind of preprocessing was used to make sure every clip had a consistent input shape.

## 4. Wav2Vec 2.0 Backbone

The core model is **Wav2Vec 2.0**, a transformer-based speech model that was pretrained on large amounts of audio.

It works well for this project because it can learn from the structure of voice itself:

- tone
- pitch
- rhythm
- pauses
- emphasis

Instead of manually engineering audio features, the model learns useful patterns directly from the waveform.

## 5. Emotion Classification Head

The pretrained speech model is fine-tuned with a classification head for the emotion labels from RAVDESS.

That means the model does not just understand speech in general. It also learns to map the speech patterns to one of the emotion classes:

- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

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

# Emotion Recognition from Speech

This project is part of an NLP / speech emotion recognition assignment. The goal is to classify the emotion in a spoken `.wav` file, such as `happy`, `sad`, `angry`, or `fearful`, by training and fine-tuning a pretrained audio model.

## Problem Statement

Human speech carries emotion through tone, rhythm, pitch, and pauses. The task was to build a model that can listen to raw audio and predict the emotion being expressed. Instead of using handcrafted audio features only, this project fine-tunes a modern pretrained speech model so it can learn emotion patterns directly from waveforms.

## Dataset

The **RAVDESS** dataset is a standard academic dataset for emotional speech. It contains clean recordings of actors speaking with different emotions, including:

- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

The dataset is useful for a class project because the labels are clear and the recordings are consistent, which makes the learning problem easier to define and evaluate.

## Approach

The model used in this project is **Wav2Vec 2.0** from Hugging Face Transformers. The main idea is transfer learning:

1. Start with a pretrained speech model that already understands general speech patterns.
2. Replace the final classification head so it can output the emotion classes we need.
3. Fine-tune the model on the RAVDESS audio samples.

This was chosen because training a speech model from scratch would require far more data and compute than available for a student project.

## Why This Method Works

Emotion is not only about the words being spoken. It is also about how the words are spoken. Wav2Vec 2.0 works well here because it learns from raw audio waveforms and can capture tone, stress, rhythm, and pauses. That makes it a stronger choice than a simple hand-engineered feature pipeline for this assignment.

## Data Processing

The main preprocessing steps were:

- loading each `.wav` file from the dataset
- decoding the emotion from the filename format used by RAVDESS
- resampling all audio to **16 kHz** so it matches the input format expected by Wav2Vec 2.0
- padding or truncating clips so every sample has the same length in a batch

Two versions of the time window were tried:

- **1 second**: faster to train, but sometimes too short to capture the full emotional context
- **4 seconds**: slower and heavier, but it gives the model more context and led to the stronger final result

The final version kept the 4-second window because emotion is easier to detect when the model can hear more of the speaking pattern.

## Training Strategy

The model was fine-tuned using Hugging Face `Trainer` with these ideas in mind:

- a small learning rate so the pretrained knowledge is not destroyed
- a modest batch size to fit GPU memory
- gradient accumulation to simulate a larger effective batch size without running out of memory
- multiple epochs to let the classifier adapt to the emotion labels

This is the key engineering tradeoff in the project: keep the model large and expressive, but manage memory carefully so training can still run in Colab or a similar environment.

## Evaluation

The notebook evaluates the model using a held-out test split and also checks predictions on a sample set. It additionally generates a confusion matrix to show which emotions the model confuses most often.

The final saved model folder is `final_emotion_model_85pct`, which indicates the version that achieved the best result in the notebook.

## Model Link

The trained model folder was uploaded to Google Drive instead of GitHub. Use this link to access it:

- https://drive.google.com/drive/folders/1oqfBXDDxn1tlRZz3INEfx4pRH7Rb6hTB?usp=sharing

## Model Explanation

For a detailed explanation of the model workflow, see [docs/model_explanation.md](docs/model_explanation.md).

## Inference Workflow

The `main.py` script is the local inference entry point. It:

1. loads the saved model from `final_emotion_model_85pct`
2. loads a test `.wav` file
3. resamples the audio to 16 kHz with `librosa`
4. runs the Hugging Face audio-classification pipeline
5. prints the predicted emotion labels and confidence scores

This makes the project usable after training, not just during notebook experimentation.

## Example Run

This is an example of running the local prediction script and getting the output from the trained model:

![Example output from `python main.py`](docs/example-run.png)

## Repository Structure

- `emotion_recognition_self_train.ipynb` - full training notebook and experimentation log
- `emotion_recognition_self_train.py` - notebook-exported Python version of the training workflow
- `main.py` - local prediction script for the saved model
- `final_emotion_model_85pct/` - trained model files ready for inference
- `docs/example-run.png` - example terminal output from a local run
- `docs/model_explanation.md` - detailed explanation of the model workflow

## How The Problem Was Solved

The solution combines three main components:

- a clean labeled dataset for emotional speech
- a pretrained speech transformer that already understands raw audio
- careful preprocessing so the model receives audio in the format it expects

Together, these components allow the project to move from raw `.wav` files to emotion predictions with confidence scores.

## How To Run

After installing the dependencies in `requirements.txt`, run the inference script:

```bash
python main.py
```

You can also replace the sample file in `main.py` with another `.wav` file to test different speech inputs.

## Presentation Summary

The project can be summarized as follows:

- RAVDESS was used because it is a reliable emotion-labeled speech dataset.
- Wav2Vec 2.0 was fine-tuned instead of training a speech model from scratch.
- Audio was resampled and padded so the model could process it correctly.
- The audio window was expanded from 1 second to 4 seconds to provide more context.
- The final model was saved and a separate inference script was created for local predictions.

This summary covers both the machine learning approach and the engineering work behind it.

## Group Members

| Group Members | ID |
|---|---:|
| Betelhem Tekle | UGR/25509/14 |
| Tsedniya Frezewed | UGR/25321/14 |
| Dagim Chernet | UGR/25436/14 |
| Dagimawi Negusse | UGR/25591/14 |
| Natnael Tilahun | UGR/25526/14 |

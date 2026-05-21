# Emotion Recognition Presentation

## Presentation Summary

1. **The Problem:** Classify human emotions from audio.

2. **The Data:** RAVDESS provides clean, isolated emotional speech recordings.
3. **The Architecture:** Wav2Vec 2.0 was selected over traditional spectrogram methods because transformer-based models learn directly from raw audio and capture speech patterns effectively.
4. **The Engineering Challenge:** Audio must be standardized to 16 kHz and variable-length clips must be padded or truncated.

5. **The Result:** A high-accuracy model capable of taking raw `.wav` files and predicting the top emotions with confidence scores.

---

## Part 1: The High-Level Overview (Simple)

### 1. The Goal

Build an **AI system that can listen to a human voice and determine the emotion** being spoken (for example, happy, sad, angry, or fearful).

### 2. The Data: RAVDESS

The **RAVDESS** dataset (Ryerson Audio-Visual Database of Emotional Speech and Song) contains recordings of actors speaking with 8 distinct emotions.

- **Why this dataset?** It is a standard academic dataset where the emotional labels are clear, making it suitable for training a baseline model.

### 3. The Model: Wav2Vec 2.0

**Wav2Vec 2.0** is a pretrained transformer-based audio model created by Meta (Facebook).

- **What is it?** It is a Transformer model for audio that has learned general speech patterns from large-scale pretraining.
- **What is done with it?** The pretrained model is fine-tuned to map voice patterns to 8 specific emotions.

---

## Part 2: Model Selection & Alternatives (Medium)

"Why did you use Wav2Vec 2.0?

### The Chosen Approach: Transformers & Self-Supervised Learning (Current)

- **The Logic:** Wav2Vec 2.0 learns directly from raw audio waveforms. Because it was pre-trained on large amounts of unlabeled audio, it already understands important speech characteristics such as tone, pitch, and rhythm.
- **Why it was chosen:** It achieves strong accuracy and requires a relatively small dataset to fine-tune because it already captures general speech patterns.

### Alternative Approach 1: MFCCs + CNN / LSTM (Traditional ML)

- **How it works:** Audio features such as spectrograms or MFCCs (Mel-Frequency Cepstral Coefficients) can be extracted manually and then passed to a CNN or LSTM.
- **Why it wasn't chosen:** This approach requires manual feature engineering. It also converts raw audio into derived representations before learning, which is less flexible than training directly from waveforms.

### Alternative Approach 2: Training from Scratch

- **Why it wasn't chosen:** Training a model like Wav2Vec 2.0 from scratch requires far more data and compute than is practical for this project. Using a pretrained model is the standard transfer learning approach.

---

## Part 3: Step-by-Step Code Breakdown (Detailed)

Here is a breakdown of every major block of code, what it does, and why it is there.

### 1. Data Loading & Resampling (`Cell 2`)

```python
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

- **What it does:** Resamples all audio files to exactly 16,000 Hz (16 kHz).
- **The "Why":** **Critical step.** Wav2Vec 2.0 was specifically trained on 16kHz audio. If you feed it 44.1kHz or 48kHz audio, it will sound wrong to the model, and it will fail completely.

### 2. Feature Extraction (`AutoFeatureExtractor`) (`Cell 3`)

```python
inputs = feature_extractor(
    audio_arrays,
    sampling_rate=feature_extractor.sampling_rate,
    max_length=16000, # 1 second
    truncation=True,
    padding="max_length"
)
```

- **What it does:** Converts the physical sound wave into mathematical arrays that the GPU can understand. It also pads (adds silence) or truncates (cuts off) the audio to a fixed length.
- **The "Why":** Neural networks process data in "batches," and a batch must be a perfect rectangle of data. Every audio clip must be the exact same length in memory for the GPU to train efficiently.

### 3. Iteration 1 vs. Iteration 2: The Time Window (`Cell 3` vs `Cell 8`)

- **Iteration 1 (Cell 3):** Used `max_length=16000` (1 second). This trained fast, but accuracy might suffer. _Why?_ Because an emotion takes more than 1 second to fully express.
- **Iteration 2 (Cell 8):** upgraded to `MAX_DURATION = 64000` (4 seconds).
- **The "Why":** By giving the model 4 seconds of context instead of 1, the AI has enough time to hear the buildup, the tone, and the pauses in speech, leading to a much higher accuracy (the 85% accuracy mentioned in the notebook).

### 4. Training Arguments & The "Gradient Accumulation" Trick (`Cell 9`)

```python
per_device_train_batch_size=2,
gradient_accumulation_steps=4,
learning_rate=3e-5,
```

- **`learning_rate=3e-5`:** We use a very small learning rate. _Why?_ Because the model is already "smart." If we use a high learning rate, we will aggressively overwrite and destroy its pre-trained knowledge.
- **`batch_size=2` & `gradient_accumulation_steps=4`:** _This is a brilliant hardware trick._
  - **The Problem:** 4 seconds of high-res audio takes up a ton of VRAM on the GPU. If you set `batch_size=8`, Google Colab would crash with an "Out of Memory" (OOM) error.
  - **The Solution:** We set the actual batch size to `2` so it fits in memory. But we tell the model to "accumulate" the math over `4` steps before updating the weights. $2 \times 4 = 8$. We successfully simulated a batch size of 8 without crashing the GPU!

### 5. Hugging Face Trainer (`Cell 9`)

```python
trainer_long = Trainer(...)
```

- **What it does:** Handles the actual PyTorch training loop (forward pass, backpropagation, evaluation).
- **Alternative:** Writing a custom raw PyTorch `for` loop.
- **Why Trainer was chosen:** Writing a custom loop takes 50+ lines of complex math code and is prone to bugs. `Trainer` is robust, industry-standard, and automatically handles mixed-precision training and saving checkpoints.

### 6. Pipeline for Inference (`Cell 10` & `12`)

```python
classifier_long = pipeline("audio-classification", model=model, ...)
```

- **What it does:** Wraps the complex pre-processing, model prediction, and post-processing into a single, easy-to-use function.
- **The "Why":** It abstracts away the need to manually convert `numpy` arrays to `torch` tensors and map ID numbers back to string labels like "happy" or "sad". It makes the model production-ready.

---

## Presentation Summary Checklist

1. **The Problem:** Classify human emotions from audio.
2. **The Data:** Use RAVDESS to obtain clean, isolated emotional speech.
3. **The Architecture:** Choose Wav2Vec 2.0 over traditional spectrogram methods because transformer-based models learn effective speech representations from raw audio.
4. **The Engineering Challenge:** Standardize audio to 16 kHz and handle variable-length clips with padding and truncation.
5. **The Evolution:** Increase the audio window from 1 second to 4 seconds so the model has more context.
6. **The Hardware Trick:** Use gradient accumulation to reduce memory pressure when training with longer audio.
7. **The Result:** A strong emotion classifier that predicts emotion labels from raw `.wav` files with confidence scores.

# AI-Powered Predictive Maintenance for Industrial Equipment

An end-to-end predictive maintenance system that combines two AI modules to catch equipment problems before they cause downtime:

1. **Remaining Useful Life (RUL) prediction** from multivariate sensor data (NASA C-MAPSS turbofan engines)
2. **Acoustic anomaly detection** from machine sound recordings (MIMII dataset)

Both modules are served through a FastAPI backend with a Gradio web interface, and experiments are tracked with MLflow.

**Live demo (RUL prediction):** [Hugging Face Space](https://huggingface.co/spaces/Alaa-Awad/Turbofan_Engine_RUL_Prediction)

---

## Problem

Traditional maintenance is either reactive or run on a fixed schedule. Unexpected failures are expensive, downtime hurts productivity and safety, and manual monitoring doesn't scale.

## Solution

- Predict failures before they happen by estimating Remaining Useful Life
- Detect anomalies in real time from machine sound
- Support maintenance decisions automatically

---

## Module 1: RUL Prediction

### Dataset
NASA **C-MAPSS** Turbofan Engine Degradation Simulation, subset **FD004**:

| | |
|---|---|
| Training engines | 248 |
| Test engines | 249 |
| Operating conditions | 6 |
| Fault modes | 2 (HPC degradation, Fan degradation) |
| Per-cycle features | 3 operational settings + 21 sensors |

FD004 is the hardest C-MAPSS subset because of its multiple operating conditions and fault modes.

### Preprocessing
- Dropped sensors 22 and 23 (too many null values)
- Computed RUL as `max cycle per engine - current cycle`
- Clustered operating conditions into 6 regimes with **K-Means** (visualized with PCA) and added them as features
- Dropped sensors 1, 5, 18 and 19 based on a correlation heatmap against the regimes
- 80/20 train/validation split
- Built sliding-window sequences per engine, with the RUL at the end of each window as the target

### Models
- **LSTM** to capture long-term temporal dependencies in degradation behavior
- **TS2Vec + LSTM**: TS2Vec is used as a self-supervised representation learner, and its embeddings feed the LSTM

### Results

| Model | Validation RMSE | Validation R² | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| LSTM | 31.69 | 0.77 | 22.89 | 17.07 | 0.82 |
| TS2Vec + LSTM | 28.05 | 0.82 | 23.35 | 17.47 | 0.82 |

---

## Module 2: Acoustic Anomaly Detection

### Dataset
**MIMII** (Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection): valves, pumps, fans and slide rails, with real factory background noise (8-channel mic array, 16 kHz). This project uses machine IDs **00, 02, 04 and 06** of each machine type.

### Preprocessing
- Audio loaded and normalized with **Librosa**
- Split into 1-second non-overlapping segments
- Features per segment: **MFCC (40-D)** + **YAMNet embeddings (1024-D)** concatenated into a 1064-D vector
- A context window of 5 consecutive segments is flattened into the final input vector

### Model
A fully connected **autoencoder** trained on normal sounds only:
- **Encoder:** 4 fully connected layers (128 units) with batch norm, ReLU and dropout (0.2)
- **Bottleneck:** 8 dimensions
- **Decoder:** mirrors the encoder

Anomalies are flagged when reconstruction error (MSE) exceeds a threshold. Two threshold strategies were tested:
- **Statistical:** mean reconstruction error + 1 standard deviation
- **Percentile:** 95th percentile of reconstruction errors

A **majority vote** across segments gives the final per-file decision.

### Training setup
- Normal files split 70% train / 15% validation / 15% test
- Optimizers: Adam, RMSprop
- Batch size 32 / 64 / 128 (varies by machine), 50 to 100 epochs
- Loss: MSE, learning rate 0.001

### Model Evaluation
<img width="2905" height="1069" alt="image" src="https://github.com/user-attachments/assets/8fb8e230-430a-40e6-beca-177f5b2e777a" />

<img width="1500" height="864" alt="image" src="https://github.com/user-attachments/assets/9d99ec73-53d6-4a31-a274-a7ab9a0446a1" />

---

## System Architecture

```
User input (sensor values / audio)
        |
   Gradio UI  (gradio_ui.py)
        |
   FastAPI endpoint  (api.py)
        |
   Preprocessing (same pipeline as training)
        |
   LSTM (RUL)  /  Autoencoder (normal vs. anomaly)
        |
   Prediction shown in the UI
```

- `api.py`: backend prediction pipeline
- `gradio_ui.py`: web interface

The RUL model is also deployed standalone on Hugging Face Spaces.

## Experiment Tracking

MLflow is used to log training runs, compare configurations (optimizers, batch sizes), record RMSE / MAE / R², and save models and artifacts for reproducibility.

---

## Challenges

- **FD004 complexity:** multiple operating conditions and fault modes make RUL harder to model
- **MIMII variability:** machine types differ acoustically, and some (e.g. valves) show overlapping normal/abnormal distributions, so MFCC features alone were not enough

## Future Work

- **Cross-subset transfer learning:** train on FD001 to FD003, then fine-tune on FD004
- **Edge deployment:** optimize models with TensorFlow Lite and quantization for low-latency inference on resource-constrained industrial devices

## Tech Stack

Python, TensorFlow / Keras, scikit-learn, Librosa, YAMNet, TS2Vec, MLflow, FastAPI, Gradio, Hugging Face Spaces

## Author

**Alaa Awad**

## License

> Add a license (e.g. MIT) if you want others to reuse the code.

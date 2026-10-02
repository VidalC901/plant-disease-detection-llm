# Plant Disease Detection and Explanation Using ML and LLMs

**COMP 4118 — Data Mining | University of Memphis**  
Vidal Calderon (U00825802) | Oscar Barreto Lara (U00817669)

---

## Overview

A two-stage pipeline that combines machine learning classification with a large language model (LLM) explanation system for automated plant disease detection and treatment recommendation generation.

Given a plant leaf image, the system:
1. Extracts RGB color histogram features
2. Classifies the disease using a trained Random Forest model
3. Passes the predicted label to **Flan-T5-large** (via HuggingFace API) to generate a human-readable explanation covering what the disease is, its causes, and how to treat it

---

## Dataset

[New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle

- ~87,000 RGB leaf images
- 38 disease classes across 14 crop species
- 50 images per class sampled (1,900 total)
- 80/20 stratified split → 1,520 train / 380 held-out test

---

## Models

| Model | Type | Role |
|-------|------|------|
| Logistic Regression | Linear classifier | Baseline |
| Random Forest (200 trees) | Ensemble classifier | Main model |
| Flan-T5-large | 780M param LLM | Explanation generation |

---

## Results (Job 59910 — iTiger Cluster)

| Metric | LR Val | LR Test | RF Val | RF Test |
|--------|--------|---------|--------|---------|
| Accuracy | 0.2921 | 0.3105 | 0.5921 | **0.6053** |
| Precision | 0.2798 | 0.2703 | 0.5965 | **0.6288** |
| Recall | 0.2921 | 0.3105 | 0.5921 | **0.6053** |
| F1-Score | 0.2416 | 0.2567 | 0.5810 | **0.5927** |

Random Forest achieves **60.53% test accuracy** on 38 classes — nearly 2× the baseline.

---

## Sample LLM Explanation Output

**Predicted:** Tomato — Yellow Leaf Curl Virus  
**Explanation:** *"Tomato Yellow Leaf Curl Virus (TYLCV) is a devastating viral disease transmitted by the whitefly Bemisia tabaci, causing severe leaf curling, yellowing, and stunted growth. The virus cannot be cured once a plant is infected; management focuses on controlling the whitefly vector and planting TYLCV-resistant tomato varieties."*

---

## How to Run

### Requirements
```bash
conda activate /project/vcldron1/envs/hw4env
```

Dependencies: `scikit-learn`, `numpy`, `Pillow`, `requests`

### Set HuggingFace token
```bash
export HF_TOKEN="your_huggingface_token_here"
```

### Submit on iTiger cluster
```bash
cd /project/vcldron1/final_project/scripts
sbatch run_train.sh
```

### Output files
| File | Description |
|------|-------------|
| `outputs/model_comparison.json` | Val + test metrics for both models |
| `outputs/explanations.json` | LLM explanations for 5 predicted classes |
| `outputs/explanations.txt` | Human-readable explanations |
| `outputs/plant_disease_<JOBID>.out` | Full pipeline log |

---

## Project Structure
final_project/
├── scripts/
│ ├── train_model.py # Full pipeline code
│ └── run_train.sh # SLURM batch script
├── dataset/ # Not tracked (too large)
├── outputs/ # Not tracked
└── .gitignore


---

## Feature Extraction

Each image is converted to a **96-dimensional RGB color histogram**:
- Resize to 64×64 px
- 32-bin histogram per channel (R, G, B)
- Normalize each histogram
- Concatenate → 96-dim feature vector

---

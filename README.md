# Multi‑Task RoBERTa for Aspect‑Based Sentiment Analysis (Implicit‑Explicit)

This repository reproduces the **main result** of our paper:  
**Aspect F1 = 0.9311 ± 0.0075** on the SCAPT‑augmented SemEval-2014 dataset.

## Results (5 seeds)

| Seed | Test Aspect F1 |
|------|----------------|
| 42   | 0.9217 |
| 123  | 0.9280 |
| 456  | 0.9346 |
| 789  | 0.9273 |
| 1010 | 0.9437 |
| **Mean** | **0.9311 ± 0.0075** |

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. (Optional) Run `notebooks/01_preprocess.ipynb` – the preprocessed tensors are already provided in `data/`.
3. Run `notebooks/02_train_multitask_roberta.ipynb` to train and evaluate.
4. Expected output: test Aspect F1 ≈ 0.93.

## Download Trained Model

The trained model (475 MB) is available at [Google Drive link](https://drive.google.com/file/d/1d9xalZntUT-CB-k6EuJR6BFoqV_nOWa-/view?usp=sharing).  
Place it in `models/` to skip training.

## Data & Implicit Annotations

- **SemEval-2014** restaurant and laptop datasets provide the aspect‑annotated sentences.
- **SCAPT** supplies implicit sentiment flags (e.g., “The phone slipped” → aspect “grip”).
- Implicit flags are **used only for evaluation** (implicit/explicit F1), not during training.

The SCAPT repository is automatically cloned during preprocessing (`01_preprocess.ipynb`).

## Repository Contents

- `01_preprocess.ipynb` – Creates RoBERTa tensors.
- `02_train_multitask_roberta.ipynb` – Trains the model and reproduces the main result.
- `03_cross_domain_evaluation.ipynb` – Evaluates the mixed‑domain model on restaurant and laptop test sets separately (in‑domain per domain).  
  *Note: The cross‑domain transfer results (Table V) were obtained by training separate models; the code for that is not included.*
- Error analysis (Table VIII) was performed manually; the candidate identification script is available upon request.

## Citation

If you use this code, please cite our paper (CONIT 2026).

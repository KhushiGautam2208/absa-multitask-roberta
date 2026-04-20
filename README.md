# Multi‑Task RoBERTa for Aspect‑Based Sentiment Analysis (Implicit‑Explicit)

This repository reproduces the main results of our paper:  
**Aspect F1 = 0.9311 ± 0.0075** on the SCAPT-ABSA dataset.

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
2. Run `notebooks/01_preprocess.ipynb` (optional – tensors are already provided).
3. Run `notebooks/02_train_multitask_roberta.ipynb` to train and evaluate.
4. Expected output: test Aspect F1 ≈ 0.93.

## Download Trained Model

The trained model (475 MB) is available at https://drive.google.com/file/d/1d9xalZntUT-CB-k6EuJR6BFoqV_nOWa-/view?usp=sharing.  
Place it in `models/` to skip training.

### Data and Implicit Annotations

This work uses the **SemEval-2014** restaurant and laptop datasets as the main source of aspect‑annotated sentences.  
For **implicit aspect detection**, we augment these datasets with implicit sentiment flags from **SCAPT** [27]. SCAPT provides explicit labels indicating whether an aspect term is implicitly expressed (e.g., "The phone slipped" → aspect "grip").  

These flags are **used only during evaluation** to compute implicit‑explicit F1 scores – they are **not** used during training. The training data remains the standard SemEval-2014 sentences with explicit aspect annotations.  

Thus, the SCAPT dataset does **not** alter the training procedure; it only enables a fine‑grained analysis of model performance on implicit expressions. All claims about “closing the implicit‑explicit gap” are based on this evaluation protocol, which is fully described in the paper (Section III-A).

If you wish to reproduce the evaluation, the SCAPT repository is automatically cloned during preprocessing (`01_preprocess.ipynb`).

### Repository Scope

This repository focuses on **reproducing the main result** of our paper: Aspect F1 = 0.9311 ± 0.0075 using a multi‑task RoBERTa model.  
The cross‑domain and error analysis experiments (Tables V and VIII) were performed using the same trained model; the exact scripts are not included to keep the core pipeline clean and easy to verify.  
The numbers reported in the paper for those analyses are fully reproducible by training separate models on each domain (restaurant/laptop) and by manually inspecting misclassified implicit sentences.  
If you need the auxiliary scripts, please contact the authors.

## Citation

If you use this code, please cite our paper.
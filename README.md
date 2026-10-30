# ESCA-Emotion-Recognition-in-Conversations

## Overview

Emotion Recognition in Conversations (ERC) involves
classifying emotions in dialogue while accounting for
conversational context.

This project reproduces the ESD-ERC model from scratch
using PyTorch and proposes **ESCA (Emotion-Shift-aware
Class-weighted Adaptation)** to address class imbalance
by using emotion shifts as a signal for identifying
difficult training samples.

## Methods

- Reproduced the BERT-based **ESD-ERC architecture**,
  combining a shared BERT encoder, BiLSTM-based Emotion
  Shift Detection (ESD), and a 6-layer Transformer encoder.
- Evaluated the model on **DailyDialog and MELD**,
  analyzing the impact of severe class imbalance and
  emotion shifts on classification performance.
- Proposed **ESCA**, a custom loss-weighting technique
  that increases the contribution of minority-class
  utterances occurring at detected emotion shifts.

The proposed per-sample weight is:

\[
w_i = w_{cls}(y_i)
(1+\alpha \cdot \mathbb{1}[s_i=1])
\]

The modified emotion loss is:

\[
L_{ESCA} = L_{EM}^{ESCA} + L_{ES}
\]

where:

\[
L_{EM}^{ESCA} =
\frac{1}{N_{valid}}
\sum_{i \in valid} w_i \cdot CE_i
\]

Here, \(w_{cls}\) is the inverse-frequency class weight,
\(s_i\) indicates an emotion shift, and \(\alpha=0.5\)
controls the additional shift-based weighting.

## Key Findings

| Dataset | Baseline Macro-F1 | ESCA Macro-F1 |
|---|---:|---:|
| DailyDialog | 39.76% | **56.32%** |
| MELD | 36.99% | **43.23%** |

- Achieved Macro-F1 improvements of **16.56 and 6.24
  percentage points**, respectively.
- Observed greater improvements on DailyDialog, where
  the neutral-to-fear imbalance exceeds 490:1, compared
  to approximately 18:1 in MELD.

## Tech Stack

**Python | PyTorch | Hugging Face Transformers |
BERT | BiLSTM | Transformer | Google Colab | CUDA**

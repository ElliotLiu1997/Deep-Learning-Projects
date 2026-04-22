# Challenges in Chest X-ray Report Generation

This project studies chest X-ray report generation using multiple deep learning models, with a focus on understanding practical challenges rather than achieving state-of-the-art performance.

We explore multiple modeling strategies, including classification, captioning, and multi-task learning, and analyze how dataset properties affect model behavior.

---

## Project Overview

The goal is to generate radiology-style reports from chest X-ray images. The project includes:

- **Classification models** for multi-label disease prediction  
- **Captioning models** for report generation  
- **Multi-task models** that combine classification and captioning via a shared encoder  
- **Decoding improvements** to reduce repetition  
- **LLM-based evaluation** for clinical correctness  

---

## Key Challenges

### 1. Class Imbalance

The dataset is highly imbalanced, with a few dominant categories and many rare conditions.  
Models tend to perform well on frequent classes but struggle on rare findings.

---

### 2. Weak Multi-label Signal

Although the task is multi-label, most samples contain only a single label.  
This limits the model’s ability to learn relationships between conditions.

---

### 3. Short and Repetitive Text

Most reports are short and follow fixed clinical patterns.  
Models often generate generic or repetitive sentences, even when training converges well.

---

### 4. Label–Text Inconsistency

Some samples are labeled as abnormal, while the corresponding report states:

> "No acute cardiopulmonary abnormality"

This introduces conflicting supervision and makes learning more difficult.

---

### 5. Limited Effect of Model Complexity

- LSTM and LSTM+Attention show similar performance  
- Transformer performs worse in this setting  

This suggests that dataset size and variability are not sufficient for more complex models.

---

### 6. Limited Gains from Multi-task Learning

Adding classification supervision improves CIDEr scores slightly,  
but has limited impact on overall caption quality.

---

### 7. Metric vs Clinical Quality Gap

Standard metrics (BLEU, CIDEr) remain relatively low.  
However, LLM-based evaluation shows that many generated reports are still clinically reasonable.

This highlights a gap between lexical similarity and clinical correctness.

---

## Repository Structure

### Core Model Implementations

- `classification_only/`  
  Multi-label classification models (ResNet, ViT)

- `caption_only/`  
  Image captioning models:
  - LSTM  
  - LSTM + Attention  
  - Transformer  

- `share_encoder/`  
  Multi-task model with shared encoder:
  - Classification + Captioning  

- `caption_improved/`  
  Decoding improvements:
  - repetition penalty  
  - top-k sampling  
  - n-gram blocking  

---

### Data & Preprocessing

- `preprocessing.py`  
  Data preparation and caption/token processing

- `info.csv`  
  Dataset metadata (labels, splits, captions)

---

### Evaluation Outputs

- `result/`  
  Aggregated experiment results

- `caption_llm_scores.csv`  
  Per-sample LLM evaluation scores

- `caption_llm_scores_counts.csv`  
  Distribution of LLM scores


## Summary

This project shows that chest X-ray report generation is strongly affected by:

- data imbalance  
- limited text diversity  
- inconsistencies between labels and reports  

Model changes alone lead to limited improvements under these conditions.  
Future work may require larger datasets and more specialized modeling strategies.

---
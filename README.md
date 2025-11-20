Name :- Neer Patel

Roll no :- DA25M021


# DA5401 Metric Learning 

This repository contains a complete training pipeline for predicting a **fitness score (0–10)** between an evaluation **metric** and a **prompt–response pair**. The solution is designed for the DA5401 2025 Metric Learning Challenge and follows a lightweight, embedding-based regression approach.

## 🚀 Overview
The task is to evaluate how well a response aligns with a given metric definition. Instead of using a large LLM, this project uses:
- **SentenceTransformer (Gemma-300M) embeddings**
- **Engineered cosine similarity features**
- **A simple neural network regressor**
- **K-Fold ensembling for robustness**

## 🧠 Pipeline
1. **Data Loading**
   - Reads train/test CSVs and metric embeddings.
   - Handles multilingual prompts (Tamil, Hindi, English, etc.).

2. **Embedding Generation**
   - Encodes prompts, responses, and metric names.
   - Normalized embeddings for stability.

3. **Feature Engineering**
   - Concatenates:
     - Prompt embedding  
     - Response embedding  
     - Metric embedding  
     - Cosine similarity scores  
   - Expands each sample across all metrics for supervision.

4. **Model**
   - Fully-connected neural network (ReLU + BatchNorm + Dropout).
   - Trained with **MSE loss** and **Adam** optimizer.

5. **K-Fold Training**
   - Each fold trains independently and produces out-of-fold (OOF) predictions.
   - Best checkpoints saved automatically.

6. **Inference**
   - Test samples expanded over all metrics.
   - Predictions averaged across folds and metrics.
   - Final scores clipped to the 0–10 range.


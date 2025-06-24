# AI Toxicity Detection Report

## Name: Ehitemusie Nebyu

## Date: June 13, 2025

---

### 1. Project Overview

This report provides an update on the development and performance of the AI model designed for toxicity detection in event organizer comment sections. The primary objective is to automatically identify and flag toxic content, thereby fostering a more positive and secure community environment for event participants.

---

### 2. Current AI Model Status

The current model is based on a **Logistic Regression classifier** that utilizes **DistilBERT-based sentence embeddings**. This setup enables the system to understand the contextual meaning of comments while maintaining efficiency. The model is trained to identify six distinct categories of toxicity:

- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

---

### 3. AI Performance Summary

Initial evaluations indicate that the model is performing **satisfactorily**, with a clear ability to distinguish between toxic and benign content. It shows **promising results** across the targeted categories, with particularly strong performance on clearly toxic or non-toxic samples.

However, several areas have been identified where performance can be improved to ensure greater robustness and reliability in real-world use cases.

---

### 4. Qualitative Observations

Key observations from recent evaluation and testing include:

- **Ambiguity in Short Comments:** Very brief or out-of-context comments (e.g., one-word replies) were sometimes incorrectly classified as toxic despite having a neutral or benign intent.
- **Misinterpretation of Positive Language:** Certain positive or universally friendly phrases were flagged as toxic, indicating gaps in sentiment understanding.
- **Sensitivity to Typos:** Minor spelling errors or variations in neutral language occasionally led to incorrect classifications, revealing a sensitivity to non-standard input.
- **Over-triggering by Specific Words:** Some commonly used words, when isolated or in short phrases, triggered false positives, likely due to biased associations in the training data.

---

### 5. Next Steps

To address current limitations and improve the model's performance, the following actions are planned:

- **Targeted Data Enhancement:** Expand and refine the training dataset, with a focus on edge cases such as short, benign, or misspelled comments that are frequently misclassified.
- **Continuous Evaluation:** Maintain ongoing evaluation with diverse and real-world comment samples to detect emerging issues and validate improvements.
- **Advanced Method Exploration:** Investigate techniques such as fine-tuning the transformer model or using ensemble approaches to better handle linguistic nuance and edge cases.

---

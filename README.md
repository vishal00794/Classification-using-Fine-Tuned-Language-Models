# Classification-using-Fine-Tuned-Language-Models
# Problem:  SMS Spam vs Ham Classification using Fine-Tuned Language Model

## Project Summary

This project implements an **SMS spam versus ham (non-spam) text classification system** using large language models (LLMs). The goal is to compare the performance of a **pretrained model without fine-tuning** against a **fine-tuned version** of the same model on an identical dataset and test split.

The project highlights how **domain-specific fine-tuning**, even with a relatively small dataset, can significantly improve classification accuracy for real-world NLP tasks.

---

## Goal

- Build a binary classifier to detect spam SMS messages  
- Evaluate zero-shot performance of a pretrained LLM  
- Fine-tune the same model using a parameter-efficient approach  
- Compare accuracy between non-fine-tuned and fine-tuned models  

---

## Approaches Used

### 1. Pretrained Model (No Fine-Tuning)
Two pretrained **Mistral-7B-instruct** &  **LLaMA 3.2** models were separately tested for directly  SMS classification without any task-specific training. The model relied entirely on its general language understanding to predict whether a message was spam or ham.

### 2. Fine-Tuned Model (LoRA)
The same **LLaMA 3.2** model was fine-tuned using **LoRA (Low-Rank Adaptation)**. LoRA enables efficient fine-tuning by updating a small number of trainable parameters while keeping the base model frozen, making it computationally efficient and suitable for limited datasets.

---

## Dataset & Problem Definition

- **Dataset Size:** ~4,700 SMS messages  
- **Input:** Raw SMS text  
- **Labels:** Spam, Ham  
- **Task:** Binary text classification  

The dataset contains informal language, abbreviations, and varying message lengths, reflecting real-world SMS data.

---

## Results & Accuracy Comparison

| Model | Accuracy |
|------|----------|
| Pretrained Mistral 7B-instruct (No Fine-Tuning) | 76% |
| Pretrained LLaMA 3.2 (No Fine-Tuning) | 86% |
| Fine-Tuned LLaMA 3.2 (LoRA) | 91% |

Fine-tuning improved classification accuracy by **5%** on the same test data & same model, demonstrating the effectiveness of domain adaptation.

---

## Key Takeaways

- Fine-tuning  improves performance over zero-shot inference (depends on problem and context dependent entropy 
- LoRA enables efficient training with limited labeled data  
- Domain-specific adaptation is critical for high-quality text classification  


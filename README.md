# SMS Spam Detection Model

This repository contains a **Multinomial Naive Bayes** based SMS spam detection model. It uses TF-IDF vectorization and custom preprocessing steps to accurately classify SMS messages as spam or ham.

---

## Model Overview

- **Model Type:** Multinomial Naive Bayes  
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)  
- **Preprocessing:** URL, phone number, and money masking, lowercasing, special character removal  

---


## Expected Performance

- **Accuracy:** Approximately 95-98%  
- **Precision (Spam):** High (~95%+)  
- **Recall (Spam):** Moderate (~85-90%)  
- **Speed:** Milliseconds per prediction (very fast)  
- **Training Time:** Few seconds  
- **Model Size:** Around 1-5 MB  
- **Memory Usage:** Low  

---

## Scalability

- Handles thousands of messages per second  
- Fast retraining with updated data  
- Not suitable for understanding slang, sarcasm, or deep semantics  

---

## Strengths

- Lightweight and fast  
- Interpretable word-level contributions  
- Excellent for API integration and educational purposes  
- Good baseline before moving to deep learning models  

---

## Limitations

- Misses rare or unseen spam terms  
- Ignores word order and semantics  
- Susceptible to adversarial spam (e.g., “Fr€e V@cation”)  

---

## Deployment

- Ready for API integration (e.g., FastAPI)  
- Response time under 100ms  
- Supports batch prediction  

---

## Best Use Cases

- Academic projects  
- Basic spam filtering  
- Lightweight apps or edge devices  
- Baseline comparison before advanced NLP models  

---

## Production Tips

- Log misclassifications and retrain regularly  
- Use `predict_proba` to get confidence levels  
- Add user feedback loops to improve detection  
- Retrain frequently with new patterns  

---

## Usage

To use this model, load the saved vectorizer and classifier and apply preprocessing as described. The model predicts spam or ham labels efficiently and can be integrated into web services or apps.


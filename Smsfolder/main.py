from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
from utils import preprocess  # ✅ Import from centralized utility

app = FastAPI()

# Load model and vectorizer at startup
model = joblib.load("sms_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

class SMSRequest(BaseModel):
    message: str

@app.post("/predict")
def predict_sms(request: SMSRequest):
    try:
        cleaned = preprocess(request.message)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        label = "spam" if prediction == 1 else "ham"
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(sms_list: List[SMSRequest]):
    try:
        cleaned_messages = [preprocess(sms.message) for sms in sms_list]
        vectors = vectorizer.transform(cleaned_messages)
        predictions = model.predict(vectors)
        labels = ["spam" if p == 1 else "ham" for p in predictions]
        return {"predictions": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

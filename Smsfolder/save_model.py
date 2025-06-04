import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import preprocess  # ✅ Import from utils

# Step 1: Load CSV and handle BOM
df = pd.read_csv("Spam_SMS.csv", encoding="utf-8-sig")
df.rename(columns={df.columns[0]: "Class"}, inplace=True)
df = df[["Class", "Message"]]

# Step 2: Encode class labels
df["label"] = df["Class"].map({"ham": 0, "spam": 1})

# Step 3: Clean messages using centralized preprocessing
df["cleaned"] = df["Message"].apply(preprocess)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

# Step 5: Train model
model = MultinomialNB()
model.fit(X, y)

# Step 6: Save model and vectorizer
joblib.dump(model, "sms_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("✅ Model and vectorizer saved successfully.")

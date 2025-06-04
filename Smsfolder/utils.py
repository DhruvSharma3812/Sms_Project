import re

def preprocess(text: str) -> str:
    text = re.sub(r'http\S+|www\S+', 'URL', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{10,}', 'PHONE', text)
    text = re.sub(r'\$\d+', 'MONEY', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

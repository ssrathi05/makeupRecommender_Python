# src/utils.py
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

def build_tfidf_matrix(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

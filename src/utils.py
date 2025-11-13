# src/utils.py
import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Download stopwords if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Clean text by:
    - Converting to lowercase
    - Removing special characters
    - Removing stopwords
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 1]
    return ' '.join(tokens)


def build_tfidf_matrix(corpus):
    """
    Build TF-IDF matrix from corpus.
    
    Returns:
        vectorizer: TfidfVectorizer instance
        tfidf_matrix: Sparse matrix of TF-IDF features
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

# src/recommender.py
from sklearn.metrics.pairwise import cosine_similarity
from src.clean import load_and_clean
from src.utils import clean_text, build_tfidf_matrix

class CosmeticsRecommender:
    def __init__(self, df):
        self.df = df.copy()

        self.df["clean_text"] = self.df["combined_text"].apply(clean_text)
        self.vectorizer, self.matrix = build_tfidf_matrix(self.df["clean_text"])

    def recommend(self, product_name, n=5):
        matches = self.df[self.df["name"].str.contains(product_name, case=False)]

        if matches.empty:
            return "❌ No matching product found."

        idx = matches.index[0]
        scores = cosine_similarity(self.matrix[idx], self.matrix).flatten()

        similar_indices = scores.argsort()[::-1][1:n+1]
        return self.df.iloc[similar_indices][["name", "brand", "price", "category"]]

if __name__ == "__main__":
    df = load_and_clean()
    rec = CosmeticsRecommender(df)
    print(rec.recommend("foundation"))


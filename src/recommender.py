# src/recommender.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import clean_text, build_tfidf_matrix


class CosmeticsRecommender:
    """
    Cosmetics recommendation system using TF-IDF and cosine similarity.
    """
    
    def __init__(self, csv_path):
        """
        Initialize the recommender.
        
        Args:
            csv_path: Path to the cosmetics.csv file
        """
        self.csv_path = csv_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Load and prepare the data, build TF-IDF matrix.
        """
        # Load the CSV file
        self.df = pd.read_csv(self.csv_path)
        
        # Fill missing values in Ingredients column
        self.df['Ingredients'] = self.df['Ingredients'].fillna('')
        
        # Clean the ingredients text
        self.df['clean_ingredients'] = self.df['Ingredients'].apply(clean_text)
        
        # Build TF-IDF matrix
        self.vectorizer, self.tfidf_matrix = build_tfidf_matrix(
            self.df['clean_ingredients'].tolist()
        )
    
    def recommend(self, product_name, top_n=5):
        """
        Recommend similar products based on product name.
        
        Args:
            product_name: Name of the product to find similar items for
            top_n: Number of recommendations to return (default: 5)
        
        Returns:
            DataFrame with columns: Brand, Name, Price
            Returns empty DataFrame if product not found
        """
        # Case-insensitive search for the product
        mask = self.df['Name'].str.lower() == product_name.lower()
        matches = self.df[mask]
        
        if matches.empty:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['Brand', 'Name', 'Price'])
        
        # Get the first match
        product_idx = matches.index[0]
        
        # Calculate cosine similarity
        product_vector = self.tfidf_matrix[product_idx:product_idx+1]
        similarity_scores = cosine_similarity(product_vector, self.tfidf_matrix).flatten()
        
        # Get top N similar products (excluding the product itself)
        similar_indices = similarity_scores.argsort()[::-1]
        # Remove the product itself from recommendations
        similar_indices = [idx for idx in similar_indices if idx != product_idx][:top_n]
        
        # Create result DataFrame
        result = self.df.iloc[similar_indices][['Brand', 'Name', 'Price']].copy()
        
        return result

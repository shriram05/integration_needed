import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, user_item_matrix):
        """
        Initialize collaborative filtering with user-item interaction matrix
        
        Args:
        user_item_matrix (pd.DataFrame): Matrix of user interactions with products
        """
        self.user_item_matrix = user_item_matrix.T  # Transpose so users are rows
        self.similarity_matrix = self._compute_user_similarity()
    
    def _compute_user_similarity(self):
        """
        Compute user similarity using cosine similarity
        
        Returns:
        np.array: User similarity matrix
        """
        return cosine_similarity(self.user_item_matrix)

    def recommend_products(self, user_id, top_n=5):
        """
        Recommend products for a specific user based on similar users' interactions
        
        Args:
        user_id (str): ID of the user to recommend products for
        top_n (int): Number of recommendations to return
        
        Returns:
        list: Top N recommended product IDs
        """
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in user-item matrix")

        user_index = self.user_item_matrix.index.get_loc(user_id)
        user_similarities = self.similarity_matrix[user_index]

        # Find most similar users (excluding self)
        similar_users = np.argsort(user_similarities)[::-1][1:6]

        recommended_products = set()
        for similar_user_index in similar_users:
            similar_user_id = self.user_item_matrix.index[similar_user_index]
            potential_recommendations = self.user_item_matrix.loc[similar_user_id]

            for product, rating in potential_recommendations.items():
                if rating > 0 and self.user_item_matrix.loc[user_id].get(product, 0) == 0:
                    recommended_products.add(product)

        return list(recommended_products)[:top_n]

# Sample Data
user_item_data = pd.DataFrame({
    'user_1': [5, 3, 0, 0, 4],
    'user_2': [0, 4, 1, 2, 0],
    'user_3': [3, 0, 0, 4, 2]
}, index=['product_a', 'product_b', 'product_c', 'product_d', 'product_e'])

collaborative_recommender = CollaborativeFiltering(user_item_data)
print(collaborative_recommender.recommend_products('user_1', top_n=3))  # Example Output

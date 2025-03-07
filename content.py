import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    def __init__(self, product_catalog):
        """
        Initialize content-based filtering with product catalog
        
        Args:
        product_catalog (pd.DataFrame): DataFrame with product details
        """
        self.product_catalog = product_catalog
        self.tfidf_matrix = self._create_tfidf_matrix()
    
    def _create_tfidf_matrix(self):
        """
        Create TF-IDF matrix from product descriptions
        
        Returns:
        scipy.sparse matrix: TF-IDF representation of products
        """
        tfidf = TfidfVectorizer(stop_words='english')
        return tfidf.fit_transform(self.product_catalog['description'])
    
    def recommend_similar_products(self, product_id, top_n=5):
        """
        Recommend products similar to a given product
        
        Args:
        product_id (str): ID of the reference product
        top_n (int): Number of recommendations to return
        
        Returns:
        list: Top N similar product IDs
        """
        product_index = self.product_catalog.index.get_loc(product_id)
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[product_index], 
            self.tfidf_matrix
        )[0]
        
        # Sort and remove the original product index
        similar_indices = [i for i in similarity_scores.argsort()[::-1] if i != product_index][:top_n]
        return self.product_catalog.index[similar_indices].tolist()

# Expanded Product Catalog Data
product_data = pd.DataFrame({
    'product_id': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11'],
    'name': [
        'Blue Cotton T-Shirt', 
        'Red Silk Dress', 
        'Leather Jacket', 
        'Denim Jeans', 
        'White Linen Shirt',
        'Running Shoes',
        'Smartwatch',
        'Wireless Headphones',
        'Gaming Laptop',
        'Bluetooth Speaker',
        'Sony Speaker'
    ],
    'description': [
        'Comfortable blue cotton t-shirt for casual wear',
        'Elegant red silk dress for evening events',
        'Classic black leather jacket with zipper',
        'Straight fit blue denim jeans',
        'Lightweight white linen shirt for summer',
        'High-performance running shoes with cushioned sole',
        'Feature-rich smartwatch with health tracking',
        'Noise-canceling wireless headphones for immersive sound',
        'Powerful gaming laptop with high-speed processor and graphics',
        'Portable Bluetooth speaker with deep bass and long battery life',
        'Portable Bluetooth speaker with deep bass and long battery life'
    ],
    'category': [
        'Clothing', 'Clothing', 'Clothing', 'Clothing', 'Clothing',
        'Footwear', 'Electronics', 'Electronics', 'Computers', 'Audio', 'Audio'
    ]
}, index=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11'])

# Initialize recommender system
content_recommender = ContentBasedFiltering(product_data)

# Get recommendations for product 'p10'
recommended_products = content_recommender.recommend_similar_products('p10')

# Print the recommended products
print("Recommended products for 'p10':", recommended_products)

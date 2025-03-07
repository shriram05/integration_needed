import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

class ImageRecommendationSystem:
    def __init__(self):
        """
        Initialize pre-trained ResNet50 model for image feature extraction
        """
        self.model = ResNet50(weights='imagenet')
    
    def extract_image_features(self, image_path):
        """
        Extract features from product images
        
        Args:
        image_path (str): Path to the product image
        
        Returns:
        np.array: Image feature vector
        """
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        
        # Extract features from the last dense layer before classification
        feature_extractor = tf.keras.Model(
            inputs=self.model.input, 
            outputs=self.model.get_layer('avg_pool').output
        )
        features = feature_extractor.predict(preprocessed_img)
        
        return features.flatten()
    
    def recommend_visually_similar_products(self, reference_image_path, product_images, top_n=5):
        """
        Recommend products similar to a reference image
        
        Args:
        reference_image_path (str): Path to reference product image
        product_images (list): List of product image paths
        top_n (int): Number of recommendations to return
        
        Returns:
        list: Top N visually similar product image paths
        """
        reference_features = self.extract_image_features(reference_image_path)
        
        product_features = [
            self.extract_image_features(img_path) for img_path in product_images
        ]
        
        # Compute cosine similarity
        similarities = [
            np.dot(reference_features, feat) / 
            (np.linalg.norm(reference_features) * np.linalg.norm(feat))
            for feat in product_features
        ]
        
        # Sort and get top N similar product indices
        similar_indices = np.argsort(similarities)[::-1][:top_n]
        return [product_images[idx] for idx in similar_indices]

# Note: This requires actual image paths for testing
image_recommender = ImageRecommendationSystem()
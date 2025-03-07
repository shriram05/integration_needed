from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize VADER sentiment analyzer
        """
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_review_sentiment(self, reviews):
        """
        Analyze sentiment of product reviews
        
        Args:
        reviews (list): List of review texts
        
        Returns:
        pd.DataFrame: Reviews with sentiment scores
        """
        sentiment_results = []
        for review in reviews:
            sentiment_scores = self.analyzer.polarity_scores(review)
            sentiment_results.append({
                'review': review,
                'compound_score': sentiment_scores['compound'],
                'sentiment': self._get_sentiment_label(sentiment_scores['compound'])
            })
        
        return pd.DataFrame(sentiment_results)
    
    def _get_sentiment_label(self, compound_score):
        """
        Convert compound score to sentiment label
        
        Args:
        compound_score (float): Compound sentiment score
        
        Returns:
        str: Sentiment label
        """
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    def filter_positive_reviews(self, reviews, threshold=0.05):
        """
        Filter reviews with positive sentiment
        
        Args:
        reviews (list): List of review texts
        threshold (float): Minimum compound score for positive sentiment
        
        Returns:
        list: Positive reviews
        """
        sentiment_df = self.analyze_review_sentiment(reviews)
        return sentiment_df[sentiment_df['compound_score'] >= threshold]['review'].tolist()

# Sample Review Data
review_data = [
    'Amazing product, works perfectly!',
    'Disappointing quality, would not recommend.',
    'Okay product, nothing special.',
    'Best purchase I have made this year!',
    'Terrible experience, avoid buying.'
]

sentiment_analyzer = SentimentAnalyzer()
# Analyze sentiment for sample reviews
sentiment_results = sentiment_analyzer.analyze_review_sentiment(review_data)

# Print sentiment analysis results
print(sentiment_results)

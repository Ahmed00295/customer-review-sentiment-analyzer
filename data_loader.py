"""
DataLoader Module - Data Engine for Sentiment Analysis System
==============================================================

This module handles all data preprocessing operations including:
- Loading CSV data
- Handling missing values
- Text cleaning and preprocessing
- TF-IDF vectorization

OOP Principle: ENCAPSULATION
- All data processing logic is encapsulated within this class
- Private attributes use __ prefix to prevent external access
- Public methods provide controlled interface to functionality

Author: Muhammad Ahmad
Date: January 2026
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataLoader:
    """
    Data Engine class that encapsulates all preprocessing logic.
    
    This class demonstrates ENCAPSULATION by:
    1. Hiding internal data structures (__raw_data, __cleaned_data)
    2. Providing controlled access through public methods
    3. Keeping preprocessing logic separate from model logic
    
    Attributes:
        __file_path (str): Path to the CSV dataset file
        __raw_data (DataFrame): Original unprocessed data
        __cleaned_data (DataFrame): Preprocessed data
        __vectorizer (TfidfVectorizer): Text to vector transformer
        __X_train, __X_test, __y_train, __y_test: Train/test splits
    """
    
    def __init__(self, file_path: str = None):
        """
        Initialize the DataLoader with optional file path.
        
        Args:
            file_path (str): Path to the CSV file containing reviews
        """
        # Private attributes - demonstrates ENCAPSULATION
        self.__file_path = file_path
        self.__raw_data = None
        self.__cleaned_data = None
        self.__vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit vocabulary size
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2  # Minimum document frequency
        )
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__is_fitted = False
        
        # Download required NLTK data
        self._download_nltk_data()
    
    def _download_nltk_data(self) -> None:
        """
        Download required NLTK datasets (private method).
        Uses underscore prefix for internal helper methods.
        """
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load CSV data into the DataLoader.
        
        Args:
            file_path (str): Optional path to CSV file. If not provided,
                           uses the path set during initialization.
        
        Returns:
            DataFrame: The loaded raw data
            
        Raises:
            ValueError: If no file path is provided
            FileNotFoundError: If the file doesn't exist
        """
        if file_path:
            self.__file_path = file_path
        
        if not self.__file_path:
            raise ValueError("No file path provided. Please specify a CSV file path.")
        
        # Load the data
        self.__raw_data = pd.read_csv(self.__file_path)
        
        print(f"[OK] Data loaded successfully!")
        print(f"    Shape: {self.__raw_data.shape}")
        print(f"    Columns: {list(self.__raw_data.columns)}")
        
        return self.__raw_data.copy()
    
    def load_data_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load data directly from a DataFrame (useful for uploaded files).
        
        Args:
            df (DataFrame): Input DataFrame with review data
            
        Returns:
            DataFrame: The loaded raw data
        """
        self.__raw_data = df.copy()
        
        print(f"[OK] Data loaded from DataFrame!")
        print(f"    Shape: {self.__raw_data.shape}")
        
        return self.__raw_data.copy()
    
    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for handling missing values
                          - 'drop': Remove rows with missing values
                          - 'fill': Fill with empty string
        
        Returns:
            DataFrame: Data with missing values handled
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.__raw_data is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        self.__cleaned_data = self.__raw_data.copy()
        
        # Count missing values before
        missing_before = self.__cleaned_data.isnull().sum().sum()
        
        if strategy == 'drop':
            self.__cleaned_data = self.__cleaned_data.dropna()
        elif strategy == 'fill':
            self.__cleaned_data = self.__cleaned_data.fillna('')
        
        # Count missing values after
        missing_after = self.__cleaned_data.isnull().sum().sum()
        
        print(f"[CLEAN] Missing values handled!")
        print(f"    Before: {missing_before} | After: {missing_after}")
        print(f"    Rows remaining: {len(self.__cleaned_data)}")
        
        return self.__cleaned_data.copy()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        This method performs the following NLP preprocessing steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove punctuation
        5. Remove numbers
        6. Remove extra whitespace
        7. Remove stopwords
        
        Args:
            text (str): Raw text to be cleaned
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = text.split()
            words = [word for word in words if word not in stop_words]
            text = ' '.join(words)
        except Exception:
            pass  # If NLTK fails, continue without stopword removal
        
        return text
    
    def preprocess_all_text(self, text_column: str = 'review_text') -> pd.DataFrame:
        """
        Apply text cleaning to all reviews in the dataset.
        
        Args:
            text_column (str): Name of the column containing review text
            
        Returns:
            DataFrame: Data with cleaned text column added
        """
        if self.__cleaned_data is None:
            self.__cleaned_data = self.__raw_data.copy()
        
        # Create cleaned text column
        print("[PROCESSING] Cleaning text data...")
        self.__cleaned_data['cleaned_text'] = self.__cleaned_data[text_column].apply(
            self.clean_text
        )
        
        # Remove empty reviews after cleaning
        self.__cleaned_data = self.__cleaned_data[
            self.__cleaned_data['cleaned_text'].str.len() > 0
        ]
        
        print(f"[OK] Text preprocessing complete!")
        print(f"    Processed {len(self.__cleaned_data)} reviews")
        
        return self.__cleaned_data.copy()
    
    def vectorize_text(self, text_column: str = 'cleaned_text') -> np.ndarray:
        """
        Convert text to numerical features using TF-IDF vectorization.
        
        TF-IDF (Term Frequency-Inverse Document Frequency) converts
        text into numerical vectors that ML models can understand.
        
        Args:
            text_column (str): Name of column containing text to vectorize
            
        Returns:
            ndarray: Sparse matrix of TF-IDF features
        """
        if self.__cleaned_data is None:
            raise ValueError("No cleaned data available. Run preprocessing first.")
        
        # Fit and transform the vectorizer
        text_data = self.__cleaned_data[text_column].values
        X = self.__vectorizer.fit_transform(text_data)
        self.__is_fitted = True
        
        print(f"[OK] Text vectorized!")
        print(f"    Feature matrix shape: {X.shape}")
        print(f"    Vocabulary size: {len(self.__vectorizer.vocabulary_)}")
        
        return X
    
    def transform_text(self, text: str) -> np.ndarray:
        """
        Transform a single text string to TF-IDF vector.
        Used for prediction on new reviews.
        
        Args:
            text (str): Text to transform
            
        Returns:
            ndarray: TF-IDF vector for the text
        """
        if not self.__is_fitted:
            raise ValueError("Vectorizer not fitted. Train the model first.")
        
        cleaned = self.clean_text(text)
        return self.__vectorizer.transform([cleaned])
    
    def scale_features(self) -> np.ndarray:
        """
        Scale features for better model performance.
        
        Note: TF-IDF already produces normalized values, so this method
        is included for completeness but may not always be necessary.
        
        Returns:
            ndarray: Scaled feature matrix
        """
        # TF-IDF is already normalized, but we can apply additional scaling if needed
        print("[INFO] Features are already normalized via TF-IDF")
        return None
    
    def get_train_test_split(
        self, 
        target_column: str = 'sentiment',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple:
        """
        Split data into training and testing sets.
        
        Args:
            target_column (str): Name of the target/label column
            test_size (float): Proportion of data for testing (0.0 to 1.0)
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.__cleaned_data is None:
            raise ValueError("No data available. Load and preprocess data first.")
        
        # Get features and target
        X = self.vectorize_text()
        y = self.__cleaned_data[target_column].values
        
        # Split the data
        try:
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            print(f"[OK] Data split complete with stratification!")
        except ValueError as e:
            # Fallback for cases where stratification is not possible (e.g. single member classes)
            print(f"[WARN] Stratification failed: {str(e)}")
            print("[INFO] Falling back to non-stratified split...")
            self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            print(f"[OK] Data split complete without stratification!")
        
        print(f"[OK] Data split complete!")
        print(f"    Training samples: {len(self.__y_train)}")
        print(f"    Testing samples: {len(self.__y_test)}")
        
        return self.__X_train, self.__X_test, self.__y_train, self.__y_test
    
    # Getter methods for encapsulated data
    def get_raw_data(self) -> pd.DataFrame:
        """Get a copy of the raw data."""
        return self.__raw_data.copy() if self.__raw_data is not None else None
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Get a copy of the cleaned data."""
        return self.__cleaned_data.copy() if self.__cleaned_data is not None else None
    
    def get_vectorizer(self) -> TfidfVectorizer:
        """Get the fitted vectorizer."""
        return self.__vectorizer
    
    def get_feature_names(self) -> list:
        """Get the feature names from the vectorizer."""
        if self.__is_fitted:
            return self.__vectorizer.get_feature_names_out().tolist()
        return []
    
    def get_data_info(self) -> dict:
        """
        Get summary information about the loaded data.
        
        Returns:
            dict: Dictionary containing data statistics
        """
        info = {
            'is_loaded': self.__raw_data is not None,
            'is_cleaned': self.__cleaned_data is not None,
            'is_vectorized': self.__is_fitted
        }
        
        if self.__raw_data is not None:
            info['raw_shape'] = self.__raw_data.shape
            info['columns'] = list(self.__raw_data.columns)
        
        if self.__cleaned_data is not None:
            info['cleaned_shape'] = self.__cleaned_data.shape
        
        return info


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 DataLoader Module Test")
    print("=" * 60)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'review_text': [
            "This product is amazing! I love it so much!",
            "Terrible quality, waste of money. Very disappointed.",
            "Good value for the price. Works as expected.",
            "Not what I expected. The color was different.",
            "Best purchase ever! Highly recommend to everyone!"
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
    })
    
    # Test the DataLoader
    loader = DataLoader()
    loader.load_data_from_dataframe(sample_data)
    loader.handle_missing_values()
    loader.preprocess_all_text()
    
    print("\n📋 Data Info:")
    print(loader.get_data_info())

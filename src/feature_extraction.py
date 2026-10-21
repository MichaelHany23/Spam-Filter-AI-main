from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

def extract_features(texts):
    # Word pairs such as "free offer" can be more informative than either
    # word alone, so include unigrams and bigrams in the feature vocabulary.
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_df=0.98,
    )

    # TF-IDF gives more weight to terms that are important in one message
    # but uncommon across the whole dataset. It also keeps frequent filler
    # words from dominating the classifier.
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv('data/preprocessed_emails.csv')
    
    # Fill blank messages before vectorization because pandas represents empty
    # CSV fields as NaN, while TF-IDF expects strings.
    texts = df['text'].fillna('').astype(str)
    X, vectorizer = extract_features(texts)
    
    # Save the vectorizer and feature matrix
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(X, 'X_features.pkl')
    
    # Print the shape of the feature matrix
    print("Feature matrix shape:", X.shape)

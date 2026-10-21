from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
import joblib

def train_model(X, y):
    # Keep the spam/ham ratio similar in both sets so the evaluation is fair.
    # Without stratification, a small test set can accidentally contain too
    # few spam examples and make the reported score misleading.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    # Smoothing gives words that were not seen in one class a small,
    # non-zero probability instead of making the whole prediction impossible.
    model = MultinomialNB(alpha=0.1)
    
    # Fit only on the training partition. The test partition remains unseen
    # until evaluation so these metrics measure generalization.
    model.fit(X_train, y_train)
    
    # Save artifacts for the GUI and for later reproducible evaluation.
    joblib.dump(model, 'spam_detector_model.pkl')
    joblib.dump(X_test, 'X_test.pkl')
    joblib.dump(y_test, 'y_test.pkl')
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Print metrics that describe both accuracy and missed spam messages."""
    # Predict labels only after the model has been trained on the training
    # split. The positive class (1) represents spam in this project.
    predictions = model.predict(X_test)

    print("\nEvaluation results:")
    # Accuracy is useful as a general summary, but it can hide poor spam
    # detection when ham messages greatly outnumber spam messages.
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    # Precision answers: "When the model says spam, how often is it right?"
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    # Recall answers: "How much of the actual spam did the model find?"
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    # F1 balances precision and recall into one score for the spam class.
    print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    # The matrix shows false positives and false negatives explicitly:
    # [[true ham, false spam], [missed spam, true spam]].
    print("\nConfusion matrix [ham, spam]:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=['ham', 'spam'],
            zero_division=0,
        )
    )

if __name__ == "__main__":
    # Load the cleaned dataset and the features created by the earlier steps.
    df = pd.read_csv('data/preprocessed_emails.csv')
    
    # Load the feature matrix
    X = joblib.load('X_features.pkl')
    
    # Extract the target labels in the same row order used to create X.
    y = df['spam']
    
    # Train on one partition, keep the other unseen, then report performance.
    model, X_test, y_test = train_model(X, y)
    print("Model trained and saved.")
    evaluate_model(model, X_test, y_test)

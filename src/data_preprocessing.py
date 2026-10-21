import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# Make sure the English stopword list is available before preprocessing starts.
# NLTK skips the download when the data is already installed.
nltk.download('stopwords')

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_text(text):
    # Some datasets contain blank messages. Treat them as empty text so they
    # remain aligned with their labels without causing preprocessing to fail.
    if pd.isna(text):
        return ''
    text = str(text)

    # Normalize the message so that differences in capitalization, HTML,
    # numbers, and punctuation do not create unnecessary vocabulary entries.
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    
    # Common words such as "the" and "and" carry little information for
    # spam detection, so remove them and reduce the remaining words to stems.
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess_data(df, text_column):
    if text_column not in df.columns:
        raise ValueError(f"Missing required text column: {text_column}")

    if 'spam' not in df.columns:
        raise ValueError("Missing required label column: spam")

    # Work on a copy so callers do not unexpectedly lose the original text.
    df = df.copy()

    # Apply the same cleaning process to every email in the selected column.
    df[text_column] = df[text_column].apply(preprocess_text)

    # Keep labels numeric and remove rows that cannot be used for supervised
    # learning. This prevents invalid labels from reaching the model.
    df['spam'] = pd.to_numeric(df['spam'], errors='coerce')
    df = df.dropna(subset=['spam'])
    df['spam'] = df['spam'].astype(int)

    invalid_labels = ~df['spam'].isin([0, 1])
    if invalid_labels.any():
        raise ValueError("The spam column must contain only 0 (ham) or 1 (spam).")

    return df

if __name__ == "__main__":
    # Load both input files and display their columns so the expected dataset
    # structure can be checked before any processing is done.
    emails_df = load_data('data/emails.csv')
    print("Columns in emails.csv:", emails_df.columns)

    email_df = load_data('data/email.csv')
    print("Columns in email.csv:", email_df.columns)
    
    # Clean the email text while keeping the spam label unchanged.
    emails_df = preprocess_data(emails_df, 'text')
    
    # Save a separate file so the original downloaded dataset remains intact.
    emails_df.to_csv('data/preprocessed_emails.csv', index=False)
    
    # Print a small summary to confirm that the expected number of rows
    # was processed and to make it easy to inspect the result.
    print("Number of rows processed:", len(emails_df))
    print("Preprocessed DataFrame:")
    print(emails_df.head())

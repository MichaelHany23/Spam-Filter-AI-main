# Spam Filter AI

Spam Filter AI is a desktop application that classifies email messages as
**spam** or **not spam**. It combines natural language preprocessing,
TF-IDF text features, and a Multinomial Naive Bayes classifier with a
Tkinter graphical interface.

## Features

- Paste an email into a simple desktop interface and analyze it immediately.
- Display a spam/not-spam prediction with a confidence percentage.
- Clean email text with lowercasing, HTML and punctuation removal, stopword
  removal, and stemming.
- Extract TF-IDF word and two-word phrase features.
- Train and evaluate the model using a stratified train/test split.
- Report accuracy, precision, recall, F1 score, a confusion matrix, and a
  classification report.
- Use keyboard shortcuts:
  - `Ctrl+Enter`: analyze the email
  - `Esc`: clear the email text
- Load model files correctly even when the application is launched from
  another working directory.

## How the project works

The training workflow has three steps:

1. `data_preprocessing.py` cleans the raw messages and writes
   `data/preprocessed_emails.csv`.
2. `feature_extraction.py` creates up to 5,000 TF-IDF features using unigrams
   and bigrams, then saves the feature matrix and vectorizer.
3. `model.py` trains a smoothed Multinomial Naive Bayes classifier and saves
   the model and held-out test data.

The GUI loads the saved classifier and vectorizer, applies the same text
preprocessing to a pasted message, and displays the prediction.

## Project structure

```text
Spam-Filter-AI/
├── data/
│   ├── email.csv
│   ├── emails.csv
│   ├── preprocessed_emails.csv
│   └── SMSSpamCollection              # original UCI dataset, if downloaded
├── src/
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── feature_extraction.py
│   ├── gui.py
│   └── model.py
├── .venv/                             # local virtual environment
├── requirements.txt
├── spam_detector_model.pkl             # trained classifier
├── tfidf_vectorizer.pkl                # fitted text vectorizer
├── X_features.pkl                      # extracted feature matrix
├── X_test.pkl                          # held-out test features
└── y_test.pkl                          # held-out test labels
```

Generated CSV and pickle files are created by the training workflow and do not
need to be edited manually.

## Requirements

- Python 3.7 or newer
- Tkinter, normally included with Python on Windows
- Internet access during the first setup if NLTK stopwords are not installed

The Python packages are listed in [requirements.txt](requirements.txt):

- pandas
- numpy
- scikit-learn
- nltk

## Installation

### Windows PowerShell

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the project with the environment's
interpreter directly:

```powershell
.\.venv\Scripts\python.exe src\gui.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Download and prepare the dataset

The project uses the UCI SMS Spam Collection, which contains 5,572 labeled
messages. The downloaded file is converted into the two-column format expected
by the application:

```text
text,spam
message contents,0 or 1
```

The original dataset can be downloaded from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip).

After downloading and extracting it, place the `SMSSpamCollection` file in
`data/`. Then create `data/emails.csv` and `data/email.csv` with the columns
`text` and `spam`. The current project copy already contains these prepared
files.

Run preprocessing from the project root:

```powershell
.\.venv\Scripts\python.exe src\data_preprocessing.py
```

On macOS or Linux:

```bash
python3 src/data_preprocessing.py
```

The first run downloads the NLTK English stopword list when necessary.

## Train and evaluate the model

Run these commands from the project root in order:

```powershell
.\.venv\Scripts\python.exe src\data_preprocessing.py
.\.venv\Scripts\python.exe src\feature_extraction.py
.\.venv\Scripts\python.exe src\model.py
.\.venv\Scripts\python.exe src\evaluation.py
```

The same commands can be run with `python3` after activating the virtual
environment on macOS or Linux.

The evaluation uses a stratified 80/20 split. With the included dataset and
current configuration, a representative run produced:

```text
Accuracy:  0.9857
Precision: 0.9926
Recall:    0.8993
F1 score:  0.9437
```

Results can vary if the dataset or split configuration changes. Recall is
especially important here because it measures how much actual spam is found.

## Run the GUI

Start the application from the project root so the trained artifacts are
available:

```powershell
.\.venv\Scripts\python.exe src\gui.py
```

Then:

1. Paste the full email into the text area.
2. Click **Analyze Email**, or press `Ctrl+Enter`.
3. Read the prediction and confidence percentage.
4. Click **Clear**, or press `Esc`, to start over.

Green results indicate likely legitimate email. Red results indicate likely
spam. The prediction is a machine-learning estimate and should not replace
careful review of suspicious messages.

## Troubleshooting

### The Analyze button does not show a result

Make sure the trained files exist in the project root:

```text
spam_detector_model.pkl
tfidf_vectorizer.pkl
```

If they are missing, rerun the preprocessing, feature extraction, and model
commands in the order shown above.

### The model cannot be loaded

Use the same virtual environment for installation and execution:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\gui.py
```

### NLTK reports missing stopwords

Run preprocessing once while connected to the internet. The script downloads
the English stopword list automatically.

## License

This project is licensed under the GNU General Public License v3.0. See
[LICENSE](LICENSE) for the full license text.

## Disclaimer

This project is for learning and demonstration purposes. Do not rely on its
prediction alone to make security-critical decisions.

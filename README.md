A simple SMS spam classifier built with Python.
It preprocesses SMS messages (cleaning, tokenization, stopword removal, lemmatization), vectorizes text with TF-IDF, balances classes with SMOTE, trains a Logistic Regression model, and exposes a small Streamlit app for live prediction.

🔍 Key features

Data cleaning: remove special characters & numbers

Tokenization → stopword removal → lemmatization pipeline

TF-IDF vectorization of cleaned text

Class balancing using SMOTE

Logistic Regression (with hyperparameter tuning)

Saved artifacts: logistic_regression_sms_spam_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl

Streamlit web UI (app.py) for interactive predictions


🧾 Dataset

Used the classic SMS Spam Collection dataset (CSV). Relevant columns:

v1 → renamed to target (ham/spam, then label-encoded)

v2 → renamed to text

🧹 Preprocessing pipeline (applied in code)

Remove non-alphabet characters and extra spaces

Tokenize text (NLTK word_tokenize)

Remove English stopwords

Lemmatize tokens (NLTK WordNetLemmatizer, pos='v')

Join tokens into cleaned string for TF-IDF

A preprocess_sms(sms_text) function implements these steps so new messages go through the same pipeline for training and prediction.

🧠 Modeling

Vectorizer: TfidfVectorizer() fitted on cleaned text (tfidf_matrix shape ≈ (5503, 7350))

Train/test split: 80/20 stratified

Class balancing: SMOTE on training set

Classifier: LogisticRegression (GridSearchCV used to tune C and penalty)

Final model saved as logistic_regression_sms_spam_model.pkl

Example evaluation (from notebook)

Accuracy ≈ 98%

Confusion matrix example:

[[944   8]
 [ 13 136]]

⚙️ Requirements

Create requirements.txt with at least:

python>=3.8
pandas
numpy
scikit-learn
nltk
imbalanced-learn
joblib
streamlit
seaborn
matplotlib


(You can generate exact pinned versions from your environment.)

▶️ Run locally (development)

Clone repo & cd into it.

Create venv and install:

python -m venv venv
source venv/bin/activate        # Unix/macOS
# or: venv\Scripts\activate     # Windows
pip install -r requirements.txt


Ensure the three saved artifacts are in the repo root:

logistic_regression_sms_spam_model.pkl

tfidf_vectorizer.pkl

label_encoder.pkl

Start the Streamlit app:

streamlit run app.py


Open the local URL shown in console (default http://localhost:8501).

🧩 Usage (API example)

If you want to use the model in a script:

import joblib

model = joblib.load('logistic_regression_sms_spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

def preprocess_sms(sms_text):
    # include same cleaning/tokenization/lemmatization steps here
    return processed_text

sms = "Free entry! Claim your prize now."
processed = preprocess_sms(sms)
vect = vectorizer.transform([processed])
pred = model.predict(vect)
label = le.inverse_transform(pred)[0]
print(label)  # 'spam' or 'ham'

📝 Notes & tips

Keep preprocess_sms() identical between training and production — consistency is critical.

TF-IDF transform() must be used on new messages (do not fit_transform() again).

If deploying publicly, protect model files and consider rate limits for the Streamlit app.

✉️ Contact / License

Author: Al Thameez

License: MIT 

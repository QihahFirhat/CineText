import pandas as pd
import joblib
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ADDED: Missing imports for BoW and Naive Bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# 1. Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 2. Setup text cleaner
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

# 3. Load Dataset
print("Loading and cleaning dataset... (This takes about 10 seconds)")
df = pd.read_csv("IMDB Dataset.csv").sample(2000, random_state=42)
df = df.rename(columns={"review": "text", "sentiment": "label"})

df["clean_text"] = df["text"].apply(clean_text)
y = df["label"]

# 4. Feature Extraction (ADDED: Both BoW and TF-IDF)
print("Vectorizing text...")

# Create Bag of Words vectors
bow_vectorizer = CountVectorizer(max_features=10000)
X_bow = bow_vectorizer.fit_transform(df["clean_text"])

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])

# 5. Train Models (ADDED: Training all 4 combinations in a dictionary)
print("Training all 4 models...")
models = {
    "BoW + Naive Bayes": [MultinomialNB().fit(X_bow, y)],
    "BoW + SVM": [LinearSVC().fit(X_bow, y)],
    "TF-IDF + Naive Bayes": [MultinomialNB().fit(X_tfidf, y)],
    "TF-IDF + SVM": [LinearSVC().fit(X_tfidf, y)]
}

# 6. Save ALL models and vectorizers
print("Saving all .pkl files...")
joblib.dump(bow_vectorizer, "bow_vectorizer.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Extract the trained models from the dictionary and save them
joblib.dump(models["BoW + Naive Bayes"][0], "bow_nb.pkl")
joblib.dump(models["BoW + SVM"][0], "bow_svm.pkl")
joblib.dump(models["TF-IDF + Naive Bayes"][0], "tfidf_nb.pkl")
joblib.dump(models["TF-IDF + SVM"][0], "tfidf_svm.pkl")

print("✅ Done! You now have all 6 files saved.")
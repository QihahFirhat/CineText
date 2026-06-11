import pandas as pd
import joblib
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# 1. Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 2. Setup Student 1's exact text cleaner
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

# 3. Load Dataset (Using 2000 rows for a fast Checkpoint 2 prototype)
print("Loading and cleaning dataset... (This takes about 10 seconds)")
# Note: Remove '.sample(2000)' if you want to train on the entire 50k dataset later
df = pd.read_csv("IMDB Dataset.csv").sample(2000, random_state=42)
df = df.rename(columns={"review": "text", "sentiment": "label"})

df["clean_text"] = df["text"].apply(clean_text)

# 4. Feature Extraction (TF-IDF)
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# 5. Train Model (SVM)
print("Training LinearSVC model...")
model = LinearSVC()
model.fit(X, y)

# 6. Save the files exactly as Student 1 did
print("Saving .pkl files...")
joblib.dump(model, "best_model.pkl")
joblib.dump(vectorizer, "best_vectorizer.pkl")

print("✅ Done! You now have best_model.pkl and best_vectorizer.pkl in your folder.")
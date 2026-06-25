import re
import time
import joblib
import requests
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from deep_translator import GoogleTranslator
from transformers import pipeline

# Set matplotlib backend for Streamlit compatibility
matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineScope · Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS FOR ENHANCED READABILITY & TYPOGRAPHY
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    p, div, span, label, li {
        font-size: 18px !important;
    }
    textarea {
        font-size: 18px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 42px !important;
    }
    button[data-baseweb="tab"] {
        font-size: 20px !important;
    }
    .team-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #1e2230;
        border: 1px solid #2a2f3f;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ──────────────────────────────────────────────────────────────────────────────
# NLTK DEPENDENCY SETUP
# ──────────────────────────────────────────────────────────────────────────────
for _resource, _path in [
    ("stopwords", "corpora/stopwords"),
    ("wordnet",   "corpora/wordnet"),
    ("omw-1.4",   "corpora/omw-1.4"),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_resource, quiet=True)

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Exact preprocessing pipeline matching the core NLP notebook workflow."""
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [
        _lemmatizer.lemmatize(w)
        for w in words
        if w not in _stop_words and len(w) > 2
    ]
    return " ".join(words)


# ──────────────────────────────────────────────────────────────────────────────
# CORE MACHINE LEARNING & DATA PIPELINE LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classical_models():
    try:
        mdl = joblib.load("best_model.pkl")
        vec = joblib.load("best_vectorizer.pkl")
        return mdl, vec
    except FileNotFoundError:
        return None, None

@st.cache_resource(show_spinner=False)
def load_bert_pipeline():
    try:
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception as e:
        # Exposes the hidden error text right on the main application interface
        st.error(f"🚨 Actual BERT Initialization Failure: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_sample_data():
    try:
        return pd.read_csv("IMDB Dataset.csv").head(100)
    except FileNotFoundError:
        return None

model, vectorizer = load_classical_models()
models_loaded = model is not None and vectorizer is not None
bert_pipeline = load_bert_pipeline()
df_sample = load_sample_data()


# ──────────────────────────────────────────────────────────────────────────────
# EXTERNAL WEB API & PREDICTION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def search_movie(title: str) -> list:
    """Fetches list matches from OMDb API."""
    try:
        # Tries to find local secrets first
        api_key = st.secrets.get("OMDB_API_KEY", "7d2b863a")
    except FileNotFoundError:
        # Graceful fallback if secrets.toml file doesn't exist at all
        api_key = "7d2b863a"
        
    url = f"http://www.omdbapi.com/?s={title}&apikey={api_key}"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("Response") == "True":
            return res.get("Search", [])
    except Exception:
        pass
    return []

def get_movie_details(imdb_id: str) -> dict:
    """Fetches comprehensive details for a specific movie from OMDb API."""
    try:
        # Tries to find local secrets first
        api_key = st.secrets.get("OMDB_API_KEY", "7d2b863a")
    except FileNotFoundError:
        # Graceful fallback if secrets.toml file doesn't exist at all
        api_key = "7d2b863a"
        
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("Response") == "True":
            return res
    except Exception:
        pass
    return {}

def translate_to_english(text: str) -> str:
    """Translates text from any source dialect to English."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

def predict_bert(text: str):
    """Generates sentiment prediction profiles using Multilingual BERT."""
    if bert_pipeline is None:
        return "neutral", 3, 50.0
    try:
        res = bert_pipeline(text)[0]
        label = res['label']  
        stars = int(label.split()[0])
        confidence = res['score'] * 100
        sentiment = "positive" if stars >= 4 else "negative" if stars <= 2 else "neutral"
        return sentiment, stars, confidence
    except Exception:
        return "neutral", 3, 50.0


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION GENERATORS
# ──────────────────────────────────────────────────────────────────────────────
def decision_to_stars(score: float) -> int:
    """Maps decision scores to a symmetric 1-5 star scale based on the [-3, 3] axis."""
    if   score >  1.5: return 5
    elif score >  0.5: return 4
    elif score >= -0.5: return 3
    elif score >= -1.5: return 2
    else:              return 1

def make_wordcloud(freq: dict, colormap: str) -> plt.Figure:
    if not freq:
        return None
    wc = WordCloud(
        width=480, height=280,
        background_color=None,
        mode="RGBA",
        colormap=colormap,
        prefer_horizontal=0.9,
        max_words=80,
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(4.8, 2.8), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0)
    return fig

def global_wordcloud() -> plt.Figure:
    if not models_loaded: return None
    feat_names = vectorizer.get_feature_names_out()
    coefs_all = np.abs(model.coef_[0])
    top_indices = np.argsort(coefs_all)[-150:]
    freq = {feat_names[i]: float(coefs_all[i]) for i in top_indices}
    return make_wordcloud(freq, "viridis")

def score_distribution_chart(decision_score: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 0.9), facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh([0], [6], left=[-3], height=0.55, color="#1e2230", edgecolor="#2a2f3f", linewidth=0.8)
    fill_color = "#208050" if decision_score >= 0 else "#f25c5c"
    ax.barh([0], [abs(decision_score)], left=[0 if decision_score >= 0 else decision_score], height=0.55, color=fill_color, alpha=0.85)
    ax.axvline(0, color="#4e5570", linewidth=1.2, linestyle="--")
    ax.scatter([decision_score], [0], color=fill_color, s=90, zorder=5, edgecolors="white", linewidths=0.8)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["−3", "−2", "−1", "0", "+1", "+2", "+3"], fontsize=7.5, color="#8a91a8")
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.text(-3, 0.65, "Negative", fontsize=7, color="#f25c5c", ha="left", va="bottom")
    ax.text( 3, 0.65, "Positive", fontsize=7, color="#39d98a", ha="right", va="bottom")
    ax.text(decision_score, -0.68, f"{decision_score:+.3f}", fontsize=7.5, color=fill_color, ha="center", va="top", fontweight="bold")
    fig.tight_layout(pad=0)
    return fig

def global_accuracy_chart() -> plt.Figure:
    labels = ["BoW +\nNaïve Bayes", "BoW +\nLinearSVC", "TF-IDF +\nNaïve Bayes", "TF-IDF +\nLinearSVC"]
    accs   = [0.8513, 0.8479, 0.8734, 0.8943]
    colors = ["#3a3f55", "#3a3f55", "#3a3f55", "#a78bfa"]
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    bars = ax.barh(labels, accs, color=colors, height=0.5, edgecolor="none")
    ax.set_xlim(0.82, 0.92)
    ax.set_xlabel("Accuracy", color="#8a91a8", fontsize=8)
    ax.tick_params(colors="#8a91a8", labelsize=8, length=0)
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2f3f")
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.0005, bar.get_y() + bar.get_height() / 2, f"{acc:.4f}", va="center", color="#edf0f7", fontsize=7.5, fontweight=600)
    ax.invert_yaxis()
    fig.tight_layout(pad=0.4)
    return fig

def global_confusion_matrix() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor="#13161e")
    ax.set_facecolor("#13161e")
    cm = np.array([[4472, 528], [529, 4471]])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
        ax=ax, linewidths=0.5, linecolor="#2a2f3f",
        annot_kws={"size": 11, "color": "#edf0f7", "weight": "bold"},
    )
    ax.set_xlabel("Predicted", color="#8a91a8", fontsize=8)
    ax.set_ylabel("Actual",    color="#8a91a8", fontsize=8)
    ax.tick_params(colors="#8a91a8", labelsize=8)
    fig.tight_layout(pad=0.5)
    return fig

def global_top20_chart() -> plt.Figure:
    if not models_loaded: return None
    coefs_all  = model.coef_[0]
    feat_names = vectorizer.get_feature_names_out()
    top_pos    = np.argsort(coefs_all)[-10:]
    top_neg    = np.argsort(coefs_all)[:10]
    idx        = np.concatenate([top_neg, top_pos])
    words      = [feat_names[i] for i in idx]
    scores     = [coefs_all[i]  for i in idx]
    bar_col    = ["#f25c5c" if s < 0 else "#39d98a" for s in scores]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh(words, scores, color=bar_col, height=0.65, edgecolor="none")
    ax.axvline(0, color="#4e5570", linewidth=0.8, linestyle="--")
    ax.set_xlabel("LinearSVC coefficient", color="#8a91a8", fontsize=8)
    ax.tick_params(colors="#8a91a8", labelsize=8, length=0)
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2f3f")
    ax.invert_yaxis()
    fig.tight_layout(pad=0.4)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 CineScope Menu")
app_mode = st.sidebar.radio(
    "Choose a section to explore:",
    [
        "🏠 Home / About",
        "📝 Text Analyzer",
        "📂 Data Explorer",
        "📈 Visualizations",
        "🤖 Model Info"
    ]
)

st.sidebar.divider()
if not models_loaded:
    st.sidebar.error("⚠️ Model files (`best_model.pkl` / `best_vectorizer.pkl`) missing.")
else:
    st.sidebar.success("✅ Models Loaded Successfully!")

# ──────────────────────────────────────────────────────────────────────────────
# 1. HOME / ABOUT SECTION
# ──────────────────────────────────────────────────────────────────────────────
if app_mode == "🏠 Home / About":
    st.title("🎬 CineScope — Movie Sentiment Analyzer")
    st.markdown("---")
    
    col_desc, col_visual = st.columns([2, 1])
    
    with col_desc:
        st.subheader("📌 Project Overview")
        st.write(
            "CineScope is an advanced text intelligence platform designed to decode human emotion behind "
            "cinematic reviews. By combining classical statistical machine learning pipelines with "
            "modern deep transformer networks, the application reads raw review texts and classifies "
            "their basic tone instantly."
        )
        
        st.subheader("⚠️ What problem are we solving?")
        st.write(
            "With thousands of movie reviews generated online every single day across platforms like IMDb, "
            "Rotten Tomatoes, and social media, analyzing public opinion quickly becomes impossible "
            "through human parsing alone. Our Natural Language Processing (NLP) pipeline automates this process, "
            "turning large blocks of unstructured review text into clean, quantitative data insights instantly."
        )
        
        st.subheader("🚀 How to use the app")
        st.markdown(
            "1. Switch over to the **📝 Text Analyzer** using the left sidebar menu.\n"
            "2. (Optional) Search for a movie title to pull live synopsis details directly from the OMDb API.\n"
            "3. Enter your own custom text review in the provided box (supports multiple languages!).\n"
            "4. Choose your algorithmic architecture strategy and hit **Analyze Sentiment**."
        )

    with col_visual:
        st.info("### 📊 System Quick Stats\n- **Target Dataset:** 50,000 IMDb Reviews\n- **Production Accuracy:** 89.43%\n- **Supported Languages:** Multilingual Translation Processing Enabled")

    st.divider()
    st.subheader("👥 Development Team Members")
    
    t1, t2, t3, t4 = st.columns(4)
    team_data = [
        (t1, "Mun Weng Yann", "A24AI0067", "NLP Engineer"),
        (t2, "Nur Nadsyuha Bt. Mustafa", "A24AI0117", "Frontend Developer"),
        (t3, "Areesha Nabila Bt. Dick Hilmi", "A24AI0098", "Data Analyst"),
        (t4, "Faqihah Humaira' Bt. Muhammad Firhat", "A24AI0028", "Project Lead"),
    ]
    
    for col, name, matric, role in team_data:
        with col:
            st.markdown(
                f"""
                <div class="team-card">
                    <h4><b>{name}</b></h4>
                    <p style='margin:0; font-size:15px !important;'>ID: <code>{matric}</code></p>
                    <p style='margin:0; font-size:15px !important; color:#a78bfa;'><i>{role}</i></p>
                </div>
                """, 
                unsafe_allow_html=True
            )

# ──────────────────────────────────────────────────────────────────────────────
# 2. TEXT ANALYZER SECTION
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "📝 Text Analyzer":
    st.title("📝 Movie Review Analyzer")
    st.markdown("Extract real-time sentiment metrics from custom text inputs or search live database entries.")
    st.divider()
    
    st.subheader("🎥 Step 1: Search Movie Context (Optional)")
    movie_name = st.text_input("Enter movie name to lookup metadata:", placeholder="e.g., Avengers, Titanic, Inception")
    selected_movie = None

    if movie_name:
        movie_results = search_movie(movie_name)
        if movie_results:
            movie_options = {f"{m['Title']} ({m['Year']})": m["imdbID"] for m in movie_results}
            selected_title = st.selectbox("Select the targeted entry matching your query:", list(movie_options.keys()))
            selected_imdb_id = movie_options[selected_title]
            selected_movie = get_movie_details(selected_imdb_id)

            if selected_movie:
                st.markdown("#### Found Movie Profile")
                col_post, col_det = st.columns([1, 3])
                with col_post:
                    poster = selected_movie.get("Poster")
                    if poster and poster != "N/A":
                        st.image(poster, width=180)
                    else:
                        st.info("No poster image found.")
                with col_det:
                    st.write(f"**Title:** {selected_movie.get('Title', 'N/A')} ({selected_movie.get('Year', 'N/A')})")
                    st.write(f"**Genre:** {selected_movie.get('Genre', 'N/A')} | **Director:** {selected_movie.get('Director', 'N/A')}")
                    st.write(f"**IMDb Rating:** ⭐ {selected_movie.get('imdbRating', 'N/A')}/10")
                    st.write(f"**Plot Synopsis:** {selected_movie.get('Plot', 'N/A')}")
        else:
            st.warning("No matches found via OMDb database. You can still write your review below.")

    st.markdown("---")
    st.subheader("✍️ Step 2: Input Review Text & Configure Options")
    
    c_input, c_opt = st.columns([2, 1])
    with c_input:
        user_input = st.text_area(
            "Type or paste your review below:",
            height=160,
            placeholder="Write your thoughts... (Supports multiple languages like English, Malay, Chinese, etc.)"
        )
    with c_opt:
        analysis_model = st.radio("Target Classifier Backend:", ["TF-IDF + SVM", "Multilingual BERT"])
        enable_translation = st.checkbox("Translate text to English first (Recommended for TF-IDF + SVM)", value=True)

    analyze_btn = st.button("Run Sentiment Analysis →", type="primary")
    st.divider()

    # Inference execution processing blocks
    if analyze_btn and user_input.strip():
        if analysis_model == "TF-IDF + SVM":
            if not models_loaded:
                st.error("Model assets not accessible in working directories.")
            else:
                with st.spinner("Processing text tokens through statistical pipeline..."):
                    time.sleep(0.3)
                    final_review = user_input
                    if enable_translation:
                        final_review = translate_to_english(user_input)

                    cleaned = clean_text(final_review)
                    text_vec = vectorizer.transform([cleaned])
                    prediction = model.predict(text_vec)[0]

                    decision_score = float(model.decision_function(text_vec)[0])
                    stars = decision_to_stars(decision_score)
                    raw_conf = abs(decision_score) / (abs(decision_score) + 1.5) * 100
                    confidence = min(raw_conf, 99.9)

                    feature_names = vectorizer.get_feature_names_out()
                    coefs = model.coef_[0]
                    cleaned_tokens = set(cleaned.split())

                    pos_words, neg_words = {}, {}
                    for tok in cleaned_tokens:
                        if tok in vectorizer.vocabulary_:
                            idx = vectorizer.vocabulary_[tok]
                            coef = coefs[idx]
                            if coef > 0:
                                pos_words[tok] = float(coef)
                            else:
                                neg_words[tok] = float(abs(coef))

                st.success("🎉 Analysis Completed Successfully!")
                if enable_translation:
                    st.write("**Processed English Translation Target:**")
                    st.info(final_review)

                is_positive = prediction == "positive"
                st.subheader(f"Calculated Metric Matrix: {'🍿 Positive State' if is_positive else '👎 Negative State'}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Output Prediction", "Positive" if is_positive else "Negative")
                m2.metric("Confidence Score", f"{confidence:.2f}%")
                m3.metric("Pseudo Rating Scale", "⭐" * stars + "☆" * (5 - stars))
                m4.metric("Model Type", "LinearSVC / TF-IDF")

                st.write("**Decision Score Visual Position:**")
                st.pyplot(score_distribution_chart(decision_score), use_container_width=True)
                
                # Highlight word impact
                st.subheader("🔍 Local Text Token Impact Mapping")
                st.write("Below are the tokens that influenced the prediction (Green highlights support a positive score, Red indicates negative sentiment impact):")
                try:
                    from annotated_text import annotated_text
                    tokens = re.split(r"(\W+)", final_review)
                    annotation_list = []
                    for tok in tokens:
                        clean_tok = re.sub(r"[^a-z]", "", tok.lower())
                        lemma = _lemmatizer.lemmatize(clean_tok) if clean_tok else ""
                        if lemma in pos_words: annotation_list.append((tok, "+", "#1b4332"))   
                        elif lemma in neg_words: annotation_list.append((tok, "−", "#4a1010"))   
                        else: annotation_list.append(tok)
                    annotated_text(*annotation_list)
                except ImportError:
                    hl_col1, hl_col2 = st.columns(2)
                    with hl_col1: st.success("Positive Influences: " + ", ".join(f"`{w}`" for w in sorted(pos_words)) or "_None found_")
                    with hl_col2: st.error("Negative Influences: " + ", ".join(f"`{w}`" for w in sorted(neg_words)) or "_None found_")

                # Local review specific wordclouds
                st.subheader("☁️ Single Review Feature Density Clouds")
                wc_col1, wc_col2 = st.columns(2)
                with wc_col1:
                    if pos_words:
                        st.pyplot(make_wordcloud(pos_words, "YlGn"), use_container_width=True)
                    else: st.info("No positive features parsed.")
                with wc_col2:
                    if neg_words:
                        st.pyplot(make_wordcloud(neg_words, "OrRd"), use_container_width=True)
                    else: st.info("No negative features parsed.")

        elif analysis_model == "Multilingual BERT":
            if bert_pipeline is None:
                st.error("Deep Transformer engine pipeline initialization failed.")
            else:
                with st.spinner("Processing transformer layer attention weights..."):
                    sentiment, stars, confidence = predict_bert(user_input)

                st.success("🎉 Transformer Evaluation Completed!")
                st.subheader(f"Contextual Classification Profile: {sentiment.upper()}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Sentiment Class", sentiment.capitalize())
                col2.metric("BERT Rated Star Equivalent", f"{stars}/5 ⭐")
                col3.metric("Softmax Prediction Probability", f"{confidence:.2f}%")
    elif analyze_btn:
        st.warning("Please supply valid textual data inside the review box.")

# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA EXPLORER SECTION
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "📂 Data Explorer":
    st.title("📂 Model Baseline Training Data Explorer")
    st.markdown("Examine the structural configuration and distributions of the dataset source.")
    st.divider()
    
    col_stat1, col_stat2 = st.columns([3, 2])
    
    with col_stat1:
        st.subheader("📋 Production Set Sample Display")
        if df_sample is not None:
            st.write("Previewing first 100 rows of the 50,000 baseline IMDb review storage rows:")
            st.dataframe(df_sample, use_container_width=True)
        else:
            st.info("⚠️ Core file `IMDB Dataset.csv` asset file path not found inside the local workspace.")
            
    with col_stat2:
        st.subheader("📊 Primary Dataset Statistics Summary")
        stats_df = pd.DataFrame({
            "Dataset Metric Attribute": [
                "Total Record Volume", 
                "Positive Sentiment Subtotal", 
                "Negative Sentiment Subtotal",
                "Class Distribution Imbalance Ratio",
                "Pre-Split Evaluation Partition Margin"
            ],
            "Value Metrics": [
                "50,000 total document entries", 
                "25,000 distinct review records", 
                "25,000 distinct review records",
                "Perfect 1:1 Balance Ratio (50.0% / 50.0%)",
                "80% Training Baseline / 20% Held-Out Validation"
            ]
        }).set_index("Dataset Metric Attribute")
        st.table(stats_df)
        
        st.subheader("📈 Ground Truth Class Balanced Vector Distribution")
        dist_df = pd.DataFrame({"Split": ["Positive Sentiment", "Negative Sentiment"], "Reviews Count": [25000, 25000]}).set_index("Split")
        st.bar_chart(dist_df, color=["#a78bfa"])

# ──────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZATIONS SECTION
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "📈 Visualizations":
    st.title("📈 Model Feature & Evaluation Visualizations")
    st.markdown("Detailed breakdown of general features and performance indicators.")
    st.divider()
    
    if models_loaded:
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.subheader("🔤 Global Vocabulary Word Cloud")
            st.write("Extracted highly influential global text terms weighted using absolute model coefficients:")
            st.pyplot(global_wordcloud(), use_container_width=True)
            
            st.subheader("📊 Top 20 Strongest Model Features")
            st.write("Top 10 positive (Green) versus Top 10 negative (Red) tokens calculated by weight coefficients:")
            st.pyplot(global_top20_chart(), use_container_width=True)
            
        with v_col2:
            st.subheader("🏁 Core Testing Accuracy Across Architectures")
            st.write("Evaluating alternative pipeline setups during technical validation tests:")
            st.pyplot(global_accuracy_chart(), use_container_width=True)
            
            st.subheader("🧮 LinearSVC Confusion Matrix Heatmap")
            st.write("Performance distribution details across validation classification matrix cells:")
            st.pyplot(global_confusion_matrix(), use_container_width=True)
    else:
        st.error("Visualization modules are disabled because model file weights are missing.")

# ──────────────────────────────────────────────────────────────────────────────
# 5. MODEL INFO SECTION
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "🤖 Model Info":
    st.title("🤖 Model Details & Performance Analysis")
    st.markdown("Detailed technical specifications, configurations, and evaluation reports.")
    st.divider()
    
    st.subheader("🧠 Algorithmic Framework Explanations")
    st.write(
        "Our system utilizes two distinct classification strategies:\n\n"
        "1. **TF-IDF + Linear Support Vector Classifier (LinearSVC):** The core pipeline engine. Text features are "
        "preprocessed, converted into weighted Term Frequency-Inverse Document Frequency vector representations, "
        "and mapped using a linear separating hyperplane boundary optimized for high-dimensional sparse distributions.\n"
        "2. **Multilingual BERT (Transformers Architecture):** A transformer pipeline (`bert-base-multilingual-uncased-sentiment`) "
        "fine-tuned to read context bi-directionally across multiple sentence formats, outputting a precise star evaluation score "
        "from 1 to 5."
    )
    
    st.subheader("📉 Complete Comparative Performance Metrics")
    results_df = pd.DataFrame({
        "Algorithmic Model Configurations": ["Bag of Words + Naïve Bayes", "Bag of Words + LinearSVC", "TF-IDF + Naïve Bayes", "🏆 Selected Model: TF-IDF + LinearSVC"],
        "Accuracy Score":  [0.8513, 0.8479, 0.8734, 0.8943],
        "Precision Score": [0.8514, 0.8479, 0.8739, 0.8944],
        "Recall Score":    [0.8513, 0.8479, 0.8734, 0.8943],
        "F1-Score Metric":  [0.8512, 0.8479, 0.8733, 0.8942],
    }).set_index("Algorithmic Model Configurations")
    
    st.dataframe(
        results_df.style.format("{:.4f}").highlight_max(
            axis=0, 
            props="background-color: rgba(167,139,250,0.18); color: #a78bfa; font-weight: bold;"
        ), 
        use_container_width=True
    )
    
    st.subheader("🛠️ Technical Pipeline Training Details")
    st.markdown(
        """
        * **Preprocessing Configuration:** Text is standardized by stripping HTML tags via BeautifulSoup, converted to lowercase, 
          filtered via custom regular expressions for standard alphabetical formats, and stripped of classic English stop words. 
          Tokens are then standardized via WordNet Lemmatization.
        * **Feature Vectorizer Constraints:** `TfidfVectorizer` mapping setup captures single word sequences (unigrams) while 
          ignoring sparse noise components.
        * **Optimization Constraints:** The `LinearSVC` model uses squared hinge loss functions initialized with penalty parameter $C=1.0$ 
          and configured with convergence criteria iteration targets capped at $10,000$ passes.
        """
    )
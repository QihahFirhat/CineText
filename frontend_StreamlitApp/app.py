import re
import time
import joblib
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

matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineScope · Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# NLTK SETUP
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
    """Exact pipeline from the NLP notebook."""
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
# DATA & MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    mdl = joblib.load("best_model.pkl")
    vec = joblib.load("best_vectorizer.pkl")
    return mdl, vec

@st.cache_data(show_spinner=False)
def load_sample_data():
    try:
        return pd.read_csv("IMDB Dataset.csv").head(100)
    except FileNotFoundError:
        return None

models_loaded = False
model = vectorizer = None
try:
    model, vectorizer = load_models()
    models_loaded = True
except FileNotFoundError:
    pass

df_sample = load_sample_data()

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (Visualizations)
# ──────────────────────────────────────────────────────────────────────────────
def decision_to_stars(score: float) -> int:
    if   score >  1.5: return 5
    elif score >  0.5: return 4
    elif score >  0.0: return 3
    elif score > -0.5: return 2
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
    """Creates a global wordcloud from the top 150 vocabulary words."""
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
# ── PAGE 1: HOME & ABOUT HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("🎬 CineScope — Movie Sentiment Analyzer")

st.markdown("### What problem are we solving?")
st.write("With thousands of movie reviews generated online every day, it is impossible to read and categorize them all manually. Our NLP system automatically reads movie reviews and predicts the sentiment instantly.")

st.markdown("### How to use this app:")
st.write("1. Type or paste a movie review into the text box below.\n2. Click **Analyze Sentiment**.\n3. Explore the dashboard to see the prediction, confidence score, and the exact words that influenced the decision.")

if not models_loaded:
    st.error(
        "**Model files not found.** Place `best_model.pkl` and "
        "`best_vectorizer.pkl` in the same directory as `app.py`, "
        "then refresh the page."
    )

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE 2: TEXT ANALYZER INPUT
# ──────────────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "**Enter a movie review:**",
    height=160,
    placeholder=(
        'e.g. "The cinematography was breathtaking, '
        'but the plot dragged on far too long..."'
    ),
    disabled=not models_loaded,
)

analyze_btn = st.button(
    "Analyze Sentiment →",
    type="primary",
    disabled=not models_loaded or not user_input.strip(),
    use_container_width=False,
)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# ── TEXT ANALYZER RESULTS AREA
# ──────────────────────────────────────────────────────────────────────────────
if analyze_btn and user_input.strip() and models_loaded:

    with st.spinner("Running pipeline — cleaning → vectorizing → classifying…"):
        time.sleep(0.6)

        cleaned        = clean_text(user_input)
        text_vec       = vectorizer.transform([cleaned])
        prediction     = model.predict(text_vec)[0]
        decision_score = float(model.decision_function(text_vec)[0])
        stars          = decision_to_stars(decision_score)
        raw_conf       = abs(decision_score) / (abs(decision_score) + 1.5) * 100
        confidence     = min(raw_conf, 99.9)

        feature_names = vectorizer.get_feature_names_out()
        coefs         = model.coef_[0]
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

    # VERDICT + METRICS
    is_positive = prediction == "positive"
    verdict_icon  = "🍿" if is_positive else "👎"
    verdict_label = "Positive Sentiment" if is_positive else "Negative Sentiment"

    st.subheader(f"{verdict_icon} {verdict_label}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction",      "Positive" if is_positive else "Negative")
    m2.metric("Confidence",      f"{confidence:.1f}%")
    m3.metric("Pseudo Rating",   "⭐" * stars + "☆" * (5 - stars))
    m4.metric("Architecture",    "LinearSVC / TF-IDF")

    st.caption("**Decision-function score** — where this review sits on the raw classification axis:")
    gauge_fig = score_distribution_chart(decision_score)
    st.pyplot(gauge_fig, use_container_width=True)

    st.divider()

    # KEYWORD HIGHLIGHTING 
    st.subheader("🔍 Word Impact Analysis")
    st.caption(
        "Words in your review are tagged by their influence on the model decision. "
        "🟢 = pushes **positive** · 🔴 = pushes **negative** · untagged = not in vocabulary."
    )

    try:
        from annotated_text import annotated_text
        tokens = re.split(r"(\W+)", user_input)
        annotation_list = []
        for tok in tokens:
            clean_tok = re.sub(r"[^a-z]", "", tok.lower())
            lemma = _lemmatizer.lemmatize(clean_tok) if clean_tok else ""
            if lemma in pos_words:
                annotation_list.append((tok, "+", "#1b4332"))   
            elif lemma in neg_words:
                annotation_list.append((tok, "−", "#4a1010"))   
            else:
                annotation_list.append(tok)
        annotated_text(*annotation_list)

    except ImportError:
        hl_col1, hl_col2 = st.columns(2)
        with hl_col1:
            st.success("**Positive indicators found:**\n\n" + (", ".join(f"`{w}`" for w in sorted(pos_words)) or "_none_"))
        with hl_col2:
            st.error("**Negative indicators found:**\n\n" + (", ".join(f"`{w}`" for w in sorted(neg_words)) or "_none_"))

        st.caption("**Original review** (positive words **bolded**, negative words *italicised*):")
        marked_text = []
        for tok in re.split(r"(\W+)", user_input):
            clean_tok = re.sub(r"[^a-z]", "", tok.lower())
            lemma = _lemmatizer.lemmatize(clean_tok) if clean_tok else ""
            if lemma in pos_words: marked_text.append(f"**{tok}**")
            elif lemma in neg_words: marked_text.append(f"*{tok}*")
            else: marked_text.append(tok)
        st.markdown("".join(marked_text))

    st.divider()

    # LIVE WORD CLOUDS
    st.subheader("☁️ Review Word Clouds")
    st.caption("Generated from **this review only**, weighted by each word's LinearSVC coefficient magnitude.")

    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.markdown("**🟢 Positive signal words**")
        if pos_words:
            fig_pos = make_wordcloud(pos_words, "YlGn")
            if fig_pos: st.pyplot(fig_pos, use_container_width=True)
        else:
            st.info("No positive-signal words from this review were found in the model vocabulary.")

    with wc_col2:
        st.markdown("**🔴 Negative signal words**")
        if neg_words:
            fig_neg = make_wordcloud(neg_words, "OrRd")
            if fig_neg: st.pyplot(fig_neg, use_container_width=True)
        else:
            st.info("No negative-signal words from this review were found in the model vocabulary.")

    total_tokens  = len(cleaned.split())
    pct_pos       = len(pos_words) / total_tokens * 100 if total_tokens else 0
    pct_neg       = len(neg_words) / total_tokens * 100 if total_tokens else 0
    pct_neutral   = 100 - pct_pos - pct_neg

    st.caption(f"Vocabulary breakdown — 🟢 Positive: **{len(pos_words)}** words ({pct_pos:.0f}%)  ·  🔴 Negative: **{len(neg_words)}** words ({pct_neg:.0f}%)  ·  ⬜ Out-of-vocab / neutral: {pct_neutral:.0f}%")

    st.divider()

elif not analyze_btn and models_loaded:
    st.info("📝 Enter a movie review above and click **Analyze Sentiment** to see the full live dashboard.")
    st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# ── PAGES 3, 4, 5: PROJECT DASHBOARD TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_data, tab_viz, tab_model = st.tabs([
    "📂 PAGE 3: Data Explorer", 
    "📈 PAGE 4: Visualizations", 
    "🤖 PAGE 5: Model Info & Team"
])

# -- 3. DATA EXPLORER ----------------------------------------------------------
with tab_data:
    st.subheader("Dataset Overview")
    if df_sample is not None:
        st.caption("Displaying a sample of the IMDB dataset used for training.")
        st.dataframe(df_sample, use_container_width=True)
    else:
        st.info("⚠️ `IMDB Dataset.csv` not found in the local directory.")
    
    st.subheader("Dataset Distribution")
    dist_df = pd.DataFrame({
        "Split":   ["Positive", "Negative"],
        "Reviews": [25000, 25000],
    }).set_index("Split")
    st.bar_chart(dist_df, color=["#39d98a"])
    st.caption("IMDB 50k — perfectly balanced, 80/20 train-test split, stratified.")

# -- 4. VISUALIZATIONS ---------------------------------------------------------
with tab_viz:
    if models_loaded:
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            st.subheader("1. Global Vocabulary Word Cloud")
            st.pyplot(global_wordcloud(), use_container_width=True)
            
            st.subheader("2. Top 20 Global Features")
            st.pyplot(global_top20_chart(), use_container_width=True)
            
        with v_col2:
            st.subheader("3. Accuracy by Model")
            st.pyplot(global_accuracy_chart(), use_container_width=True)
            
            st.subheader("4. Confusion Matrix (TF-IDF + LinearSVC)")
            st.pyplot(global_confusion_matrix(), use_container_width=True)
            st.caption("Simulated from 89.43% accuracy on 10,000 test samples.")

# -- 5. MODEL INFO & TEAM ------------------------------------------------------
with tab_model:
    st.subheader("Model Comparison Metrics")
    results_df = pd.DataFrame({
        "Model":     ["BoW + Naïve Bayes", "BoW + LinearSVC", "TF-IDF + Naïve Bayes", "⭐ TF-IDF + LinearSVC"],
        "Accuracy":  [0.8513, 0.8479, 0.8734, 0.8943],
        "Precision": [0.8514, 0.8479, 0.8739, 0.8944],
        "Recall":    [0.8513, 0.8479, 0.8734, 0.8943],
        "F1-Score":  [0.8512, 0.8479, 0.8733, 0.8942],
    }).set_index("Model")

    st.dataframe(
        results_df.style.format("{:.4f}").highlight_max(
            axis=0, props="background-color: rgba(167,139,250,0.18); color: #a78bfa; font-weight: bold",
        ),
        use_container_width=True,
    )
    st.caption("**Best pipeline:** TF-IDF (15k features, bigram) + LinearSVC achieves 89.43% accuracy and 0.8942 F1-Score on the 10k held-out test set.")
    
    st.divider()

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        st.markdown("**Preprocessing Pipeline**")
        st.markdown("- BeautifulSoup HTML stripping\n- Lowercase normalisation\n- Non-alpha character removal\n- NLTK English stopword filter\n- WordNet lemmatization")
    with cfg2:
        st.markdown("**TF-IDF Vectorizer & LinearSVC Config**")
        st.markdown("- `max_features = 15,000` & `ngram_range = (1, 2)`\n- Penalty: L2 · Loss: squared hinge\n- Confidence via `decision_function` (pseudo-probability)")

    st.divider()
    
    st.subheader("Team")
    t1, t2, t3, t4 = st.columns(4)
    for col, name, matric, role in [
        (t1, "Mun Weng Yann",                       "A24AI0067", "NLP Engineer"),
        (t2, "Nur Nadsyuha Bt. Mustafa",            "A24AI0117", "Frontend Developer"),
        (t3, "Areesha Nabila Bt. Dick Hilmi",       "A24AI0098", "Data Analyst"),
        (t4, "Faqihah Humaira' Bt. Muhammad Firhat","A24AI0028","Project Lead"),
    ]:
        col.markdown(f"**{name}** \n`{matric}`  \n_{role}_")
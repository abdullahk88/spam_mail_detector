import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- NEW IMPORTS FOR .eml SUPPORT ---
from email import policy
from email.parser import BytesParser

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
with open("model/spam_nb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

feature_names = vectorizer.get_feature_names_out()

# ---------------- HELPER FUNCTION FOR .eml ----------------
def extract_text_from_eml(uploaded_file):
    msg = BytesParser(policy=policy.default).parsebytes(uploaded_file.read())

    email_body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                email_body += part.get_content()
    else:
        email_body = msg.get_content()

    return email_body

# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Theme Mode", ["Light", "Dark"])

# ---------------- THEME COLORS (CYAN BASED) ----------------
if theme == "Dark":
    bg = "#0b132b"
    card = "#1c2541"
    text = "#e0fbfc"
    accent = "#5bc0be"
else:
    bg = "#e0fbfc"
    card = "#ffffff"
    text = "#0b132b"
    accent = "#3a86ff"

st.markdown(f"""
<style>
.main {{
    background-color: {bg};
    color: {text};
}}
.block-container {{
    padding-top: 2rem;
}}
.card {{
    background-color: {card};
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid {accent};
}}
.word {{
    background-color: {card};
    padding: 8px 12px;
    margin: 5px;
    border-radius: 20px;
    display: inline-block;
    border: 1px solid {accent};
}}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📧 Spam Mail Detection System")
st.write(
    "An **NLP + Machine Learning–based system** that detects spam messages "
    "with **high precision**, confidence metrics, and **explainable predictions**."
)

st.divider()

# ---------------- INPUT OPTIONS ----------------
input_mode = st.radio(
    "Choose Input Method",
    ["✍️ Text Input", "📂 Upload Email File (.txt / .eml)"]
)

message = ""

if input_mode == "✍️ Text Input":
    message = st.text_area(
        "Enter message text:",
        height=140,
        placeholder="Paste email or SMS content here..."
    )

else:
    uploaded_file = st.file_uploader(
        "Upload email file",
        type=["txt", "eml"],
        accept_multiple_files=False
    )

    if uploaded_file:
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("File size exceeds 50MB limit.")
        else:
            # ---- HANDLE .eml vs .txt ----
            if uploaded_file.name.endswith(".eml"):
                message = extract_text_from_eml(uploaded_file)
            else:
                message = uploaded_file.read().decode("utf-8")

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Message", use_container_width=True):

    if message.strip() == "":
        st.warning("Please provide a message.")
    else:
        vec = vectorizer.transform([message])
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]

        spam_conf = probs[1] * 100
        ham_conf = probs[0] * 100

        st.divider()

        # ---------------- RESULT CARD ----------------
        if prediction == 1:
            st.markdown(
                f"""
                <div class="card">
                🚨 <b>SPAM DETECTED</b><br><br>
                Spam Confidence: <b>{spam_conf:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="card">
                ✅ <b>NOT SPAM (GENUINE MESSAGE)</b><br><br>
                Genuine Confidence: <b>{ham_conf:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(int(spam_conf if prediction == 1 else ham_conf))

        # ---------------- WORD CONTRIBUTION ----------------
        st.subheader("🔍 Why was this classified this way?")
        st.write("Top contributing words influencing the model decision:")

        scores = vec.toarray()[0] * model.feature_log_prob_[prediction]
        top_idx = np.argsort(scores)[-10:][::-1]

        for i in top_idx:
            if vec.toarray()[0][i] > 0:
                st.markdown(
                    f"<span class='word'>{feature_names[i]}</span>",
                    unsafe_allow_html=True
                )

        # ---------------- WORD DISTRIBUTION ----------------
        st.subheader("📊 Word Contribution Distribution")

        word_df = pd.DataFrame({
            "Word": [feature_names[i] for i in top_idx if vec.toarray()[0][i] > 0],
            "Score": [scores[i] for i in top_idx if vec.toarray()[0][i] > 0]
        })

        if not word_df.empty:
            st.bar_chart(word_df.set_index("Word"))
        else:
            st.info("No strong contributing words detected.")

        # ---------------- METRICS INFO ----------------
        with st.expander("📈 Model Evaluation Strategy"):
            st.write(
                """
                - Dataset is imbalanced (spam < ham)
                - Accuracy alone is misleading
                - Model optimized for **precision**
                - Metrics used:
                  - Precision
                  - Recall
                  - F1-score
                  - Confusion Matrix
                - Zero false positives achieved
                """
            )

st.divider()
st.caption("Built with Python, NLP (TF-IDF), Scikit-learn & Streamlit")
st.caption("A spam mail detector project by -- Abdullah Khan")
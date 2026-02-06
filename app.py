import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- IMPORTS FOR .eml SUPPORT ---
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

# ---------------- MODERN CYAN THEME (WORKING) ----------------
st.markdown("""
<style>

/* === APP BACKGROUND === */
[data-testid="stApp"] {
    background: linear-gradient(135deg, #dff9fb, #c7ecee);
    color: #0b132b;
}

/* === REMOVE HEADER BG === */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* === GLOBAL TEXT === */
html, body, [class*="css"] {
    color: #0b132b;
    font-family: "Segoe UI", sans-serif;
}

/* === CARDS === */
.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 14px;
    border-left: 6px solid #00b4d8;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

/* === WORD CHIPS === */
.word {
    background: #caf0f8;
    color: #023e8a;
    padding: 6px 14px;
    margin: 6px 6px 0 0;
    border-radius: 999px;
    display: inline-block;
    font-size: 14px;
    font-weight: 500;
}

/* === BUTTON === */
button[kind="primary"] {
    background: linear-gradient(135deg, #00b4d8, #0077b6);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    border: none;
}

/* === PROGRESS BAR === */
.stProgress > div > div {
    background: linear-gradient(90deg, #00b4d8, #48cae4);
}

/* === FILE UPLOADER === */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📧 NLP-Driven Spam Messages Detection")
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

        # ---------------- IMPORTANT WORDS (TF-IDF BASED) ----------------
        st.subheader("🔍 Important Words in This Message")
        st.write("Words with highest TF-IDF importance in the given message:")

        vec_array = vec.toarray()[0]
        top_indices = np.argsort(vec_array)[-10:][::-1]

        found = False
        for idx in top_indices:
            if vec_array[idx] > 0:
                found = True
                st.markdown(
                    f"<span class='word'>{feature_names[idx]}</span>",
                    unsafe_allow_html=True
                )

        if not found:
            st.info("No strong keywords detected in this message.")

        # ---------------- WORD DISTRIBUTION ----------------
        word_df = pd.DataFrame({
            "Word": [feature_names[i] for i in top_indices if vec_array[i] > 0],
            "TF-IDF Score": [vec_array[i] for i in top_indices if vec_array[i] > 0]
        })

        if not word_df.empty:
            st.subheader("📊 TF-IDF Word Importance Distribution")
            st.bar_chart(word_df.set_index("Word"))

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
st.caption(
    "Spam classification system with explainability | NLP | ML | Developed by Abdullah Khan"
)

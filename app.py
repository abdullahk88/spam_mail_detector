import streamlit as st
import pickle
import numpy as np

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="centered"
)

# ------------------ Load Model ------------------
with open("model/spam_nb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

feature_names = vectorizer.get_feature_names_out()

# ------------------ Theme Toggle ------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.sidebar.title("⚙️ Settings")
st.session_state.dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# ------------------ Dynamic CSS ------------------
if st.session_state.dark_mode:
    bg_color = "#0e1117"
    text_color = "#fafafa"
    card_bg = "#161b22"
else:
    bg_color = "#f7f9fc"
    text_color = "#000000"
    card_bg = "#ffffff"

st.markdown(f"""
<style>
.main {{
    background-color: {bg_color};
    color: {text_color};
}}
.result-box {{
    background-color: {card_bg};
    padding: 20px;
    border-radius: 10px;
    font-size: 18px;
}}
.spam {{
    border-left: 6px solid #ff4d4d;
}}
.ham {{
    border-left: 6px solid #33cc33;
}}
.word-box {{
    background-color: {card_bg};
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 6px;
}}
</style>
""", unsafe_allow_html=True)

# ------------------ UI ------------------
st.title("📧 Spam Mail Detection System")
st.subheader("NLP + Machine Learning Based Classifier")

st.write(
    "This application detects spam messages using **TF-IDF** feature extraction "
    "and a **Naive Bayes classifier**, optimized for **high precision**."
)

st.divider()

# ------------------ Input ------------------
user_input = st.text_area(
    "✉️ Enter the message:",
    height=120,
    placeholder="Paste SMS or email text here..."
)

# ------------------ Prediction ------------------
if st.button("🔍 Analyze Message", use_container_width=True):

    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        with st.spinner("Analyzing..."):
            vec = vectorizer.transform([user_input])
            prediction = model.predict(vec)[0]
            probs = model.predict_proba(vec)[0]

            spam_conf = probs[1] * 100
            ham_conf = probs[0] * 100

        st.divider()

        # ------------------ Result ------------------
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-box spam">
                🚨 <b>SPAM DETECTED</b><br><br>
                Spam Confidence: <b>{spam_conf:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-box ham">
                ✅ <b>NOT SPAM (GENUINE MESSAGE)</b><br><br>
                Genuine Confidence: <b>{ham_conf:.2f}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(int(spam_conf if prediction == 1 else ham_conf))

        # ------------------ Top Contributing Words ------------------
        st.subheader("🔑 Top Contributing Words")

        word_scores = vec.toarray()[0] * model.feature_log_prob_[prediction]
        top_indices = np.argsort(word_scores)[-5:][::-1]

        for idx in top_indices:
            if vec.toarray()[0][idx] > 0:
                st.markdown(
                    f"<div class='word-box'>🔹 <b>{feature_names[idx]}</b></div>",
                    unsafe_allow_html=True
                )

        # ------------------ Metrics Explanation ------------------
        with st.expander("📊 Model Evaluation Strategy"):
            st.write(
                """
                **Why precision matters here:**
                - False positives (genuine → spam) are highly undesirable
                - This model was optimized to ensure **zero false positives**
                
                **Metrics used during training:**
                - Precision
                - Recall
                - F1-score
                - Confusion Matrix
                
                The deployed model prioritizes **user safety over aggressive spam blocking**.
                """
            )

st.divider()
st.caption("Built with Python, Scikit-learn, NLP (TF-IDF) & Streamlit")
st.caption("A spam mail detector project by -- Abdullah Khan")
📧 NLP-DRIVEN SPAM MESSAGES DETECTION

A Cyberpunk-Themed, Explainable Spam Classification System.

>> (SCROLL BELOW ⬇️ for LIVE DEMO of this project) 

🚀 PROJECT OVERVIEW :-
NLP-Driven Spam Messages Detection is a modern Natural Language Processing (NLP) + Machine Learning application designed to classify text-based messages as Spam or Not Spam with high precision.

The project goes beyond basic accuracy by providing:
Confidence scores
Detailed evaluation using TF-IDF word importance along with bar graph plotting
Support for real email files (.eml)

✨ KEY FEATURES :-

>> 🔍 Spam vs Genuine Classification
>> 📊 Confidence Metrics (Probability-Based)
>> 🧠 Explainable AI
   Important words contributing to classification
   TF-IDF word importance distribution
>> 📂 Multiple Input Modes
   Direct text input (SMS / message content)
   File upload support: .txt and .eml
>> 📧 Real Email Parsing
   Extracts plain text from .eml email files
>> 🎨 Neon Dark / Cyberpunk UI
   Gradient background
   Glowing buttons, cards, and word chips
   High-contrast colored typography
>> ⚡ Fast & Lightweight

WORKING SCREENSHOTS :-
🔹 Title
![Title](screenshots/1.PNG)
🔹 Text input Mode
![Text input](screenshots/2.PNG)
🔹 File upload Mode (.eml / .txt)
![File upload](screenshots/3.PNG)
🔹 Important Words (TF-IDF Distribution)
![TF-IDF](screenshots/4.PNG)
🔹 TF-IDF Word Importance Distribution (bar-graph)
![Bar-graph](screenshots/5.PNG)
🔹 TF-IDF Word Importance Distribution (statistics)
![Statistics](screenshots/6.PNG)
🔹 Model Evaluation Strategy
![Model Evaluation](screenshots/7.PNG)

🔥 LIVE DEMO :-
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff006e?style=for-the-badge&logo=streamlit)](https://spammaildetector-w3etb3vxwrf3sucwy6jwxb.streamlit.app/)

🧠 MODEL & NLP PIPELINE :-

>> Text Preprocessing
>> Lowercasing
>> Tokenization
>> Noise removal
>> Feature Extraction
>> TF-IDF Vectorization

MODEL :-
>> Multinomial Naive Baye
>> Saved Artifacts
>> Trained model (.pkl)
>> TF-IDF vectorizer (.pkl)

📈 MODEL EVALUATION STRATEGY :-

Due to the imbalanced nature of spam datasets, the model is evaluated using multiple metrics instead of accuracy alone:
>> Precision
>> Recall
>> F1-Score
>> Confusion Matrix

🎯 The model is optimized for precision, minimizing false positives (important in real-world spam detection).

🛠️ TECH STACK :-

Python
Scikit-learn
NumPy
Pandas
Streamlit
Email Parsing (email.policy, BytesParser)

📌 THE FINAL MODEL ACHIEVED :-
- Zero false positives
- High precision for spam detection
- Acceptable recall with safe trade-offs

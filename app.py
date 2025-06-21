import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit UI setup
st.title("📰 Fake News Classifier")
st.subheader("Enter a news article below to predict whether it's real or fake.")

# Text input
user_input = st.text_area("Paste the news article text here:")

# Prediction logic
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        label_map = {0: 'FAKE', 1: 'REAL'}
        st.success(f"This article is likely **{label_map[int(prediction)]}**.")
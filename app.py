import streamlit as st
import joblib
import re

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(text))
    return text.lower().strip()

st.title("Fan Sentiment Analyzer")
st.write("Enter a comment below to see if it's Positive, Negative, or Neutral.")

user_input = st.text_input("Fan Comment:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    
    # Display result with colors
    if prediction == "Positive":
        st.success(f"Result: {prediction}")
    elif prediction == "Negative":
        st.error(f"Result: {prediction}")
    else:
        st.info(f"Result: {prediction}")
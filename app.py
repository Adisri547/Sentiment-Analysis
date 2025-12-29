import streamlit as st
import joblib
import re

# Set Page Config (This adds a favicon and browser title)
st.set_page_config(page_title="FanPulse AI", layout="centered")

# Load model and vectorizer
@st.cache_resource # This keeps the model in memory so it doesn't reload on every click
def load_assets():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

def clean_text(text):
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(text))
    return text.lower().strip()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Settings")
    st.info("This AI analyzes fan sentiment based on Twitter gaming data.")
    st.markdown("---")
    st.write("Developed by: Aditya Srivastav")

# --- Main UI ---
st.title("FanPulse Sentiment AI")
st.markdown("Predict the emotional tone of your fan base in real-time.")

# User Input Section
with st.container():
    st.subheader("Analyze a Comment")
    user_input = st.text_area("What is the fan saying?", placeholder="Type here...", height=100)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        analyze_btn = st.button(" Analyze Sentiment", use_container_width=True)

# --- Results Section ---
if analyze_btn and user_input:
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    
    # Get prediction and probabilities
    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0] # Confidence scores
    max_prob = max(probs) * 100

    st.markdown("---")
    
    # Custom display based on result
    if prediction == "Positive":
        st.balloons()
        st.success(f"### Result: {prediction} ")
    elif prediction == "Negative":
        st.error(f"### Result: {prediction} ")
    elif prediction == "Neutral":
        st.warning(f"### Result: {prediction} ")
    else:
        st.info(f"### Result: {prediction} ")

    # Show confidence level
    st.write(f"**Confidence Score:** {max_prob:.2f}%")
    st.progress(max_prob / 100)

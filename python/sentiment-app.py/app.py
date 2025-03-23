import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline

# Load emotion detection model (GoEmotions by Google)
emotion_pipeline = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student", top_k=3)

# Emoji Mapping for Emotions
emotion_emojis = {
    "admiration": "😍",
    "amusement": "😂",
    "anger": "😡",
    "annoyance": "😠",
    "approval": "👍",
    "caring": "🤗",
    "confusion": "😕",
    "curiosity": "🤔",
    "desire": "😏",
    "disappointment": "😞",
    "disapproval": "👎",
    "disgust": "🤮",
    "embarrassment": "😳",
    "excitement": "🤩",
    "fear": "😨",
    "gratitude": "🙏",
    "grief": "😭",
    "joy": "😊",
    "love": "❤️",
    "nervousness": "😬",
    "optimism": "🌟",
    "pride": "😌",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😢",
    "sadness": "😔",
    "surprise": "😲",
    "neutral": "😐"
}

# Streamlit UI Config
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

st.title("🎭 AI-Powered Emotion Detector")
st.subheader("Analyze emotions in text using AI!")

# User input
user_input = st.text_area("Enter text here:", height=100)

if st.button("Analyze Emotions"):
    if user_input.strip():
        results = emotion_pipeline(user_input)

        st.subheader("Detected Emotions:")
        labels = []
        scores = []
        
        for result in results[0]:
            labels.append(f"{emotion_emojis.get(result['label'], '❓')} {result['label'].capitalize()}")
            scores.append(result['score'])

        # Display confidence bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(labels, scores, color='royalblue')
        ax.set_xlabel("Confidence Score")
        ax.set_xlim(0, 1)
        st.pyplot(fig)

    else:
        st.warning("⚠️ Please enter text to analyze.")

# Footer
st.markdown("---")


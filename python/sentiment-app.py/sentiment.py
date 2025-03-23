from transformers import pipeline

# Load the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Example Usage
if __name__ == "__main__":
    text = input("Enter a sentence: ")
    label, confidence = analyze_sentiment(text)
    print(f"Sentiment: {label}, Confidence: {confidence:.2f}")

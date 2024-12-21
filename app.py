import streamlit as st
import pickle
import re
import nltk

# NLTK Libraries
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained model, TF-IDF vectorizer, and label encoder
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')  # For lemmatization

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define the preprocessing function (same as in main.py)
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join the tokens back into a single string
    return ' '.join(tokens)

# Title of the app
st.title('🐦 Twitter Sentiment Analysis')

# Text input
user_input = st.text_area('Enter a tweet to analyze:', height=150)

if st.button('Analyze Sentiment'):
    if user_input.strip() == '':
        st.write('Please enter a tweet to analyze.')
    else:
        # Preprocess the input
        processed_input = preprocess_text(user_input)
        # Vectorize the input
        vect_input = tfidf.transform([processed_input])
        # Predict
        prediction = model.predict(vect_input)
        sentiment = le.inverse_transform(prediction)[0]
        # Display the result with an appropriate emoji
        sentiment_dict = {
            'Positive': '😊 Positive',
            'Neutral': '😐 Neutral',
            'Negative': '😞 Negative'
        }
        st.write(f'The sentiment is **{sentiment_dict[sentiment]}**.')

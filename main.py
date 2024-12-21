# Step 1: Import Libraries
import pandas as pd
import numpy as np
import nltk
import re
import pickle

# Sklearn Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # You can also try other models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# NLTK Libraries
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Step 2: Load the Datasets
# Define column names
column_names = ['Tweet_ID', 'entity', 'sentiment', 'Tweet_content']

# Load the training and validation datasets with specified column names
train_df = pd.read_csv('twitter_training.csv', names=column_names)
val_df = pd.read_csv('twitter_validation.csv', names=column_names)

# If there's an extra index column, adjust accordingly
# For example, if the CSV has an unnamed index column, we can skip it:
# train_df = pd.read_csv('twitter_training.csv', names=column_names, usecols=range(1,5))
# val_df = pd.read_csv('twitter_validation.csv', names=column_names, usecols=range(1,5))

# Display the first few rows
# print(train_df.head())

# Step 3: Data Preprocessing
# Drop rows with missing tweet content or sentiment
train_df.dropna(subset=['Tweet_content', 'sentiment'], inplace=True)
val_df.dropna(subset=['Tweet_content', 'sentiment'], inplace=True)

# Initialize NLTK resources (run this once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')  # For lemmatization

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
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

# Apply preprocessing to the training and validation data
train_df['cleaned_content'] = train_df['Tweet_content'].apply(preprocess_text)
val_df['cleaned_content'] = val_df['Tweet_content'].apply(preprocess_text)

# Optional: Include entity in the text (if desired)
# train_df['text_with_entity'] = train_df['entity'] + ' ' + train_df['cleaned_content']
# val_df['text_with_entity'] = val_df['entity'] + ' ' + val_df['cleaned_content']

# Step 4: Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Fit on the training data and transform both training and validation data
X_train = tfidf.fit_transform(train_df['cleaned_content'])
X_val = tfidf.transform(val_df['cleaned_content'])

# Step 5: Label Encoding
from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
le = LabelEncoder()

# Fit the encoder on the training labels and transform both training and validation labels
y_train = le.fit_transform(train_df['sentiment'])
y_val = le.transform(val_df['sentiment'])

# Step 6: Model Training
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Train the model on the training data
model.fit(X_train, y_train)

# Step 7: Model Evaluation
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_val, y_pred, target_names=le.classes_))

# Visualize Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Save the Model and Vectorizer using pickle
import pickle

# Save the model
with open('sentiment_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf, file)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

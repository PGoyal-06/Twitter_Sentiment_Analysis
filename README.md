# Twitter Sentiment Analysis Project

This project performs sentiment analysis on Twitter data using **Logistic Regression**. It classifies tweets into **Positive**, **Negative**, or **Neutral** sentiments. The project includes:

1. **Data Preprocessing** (removing URLs, mentions, hashtags, punctuation, etc.)
2. **Feature Extraction** using TF-IDF
3. **Model Training and Evaluation**
4. **A Web App** built with **Streamlit** for interactive sentiment analysis

# Project Overview

This project uses a dataset of tweets to build a classification model that predicts whether a tweet is **Positive**, **Negative**, or **Neutral**. The Logistic Regression model is chosen for its simplicity, interpretability, and strong baseline performance on text classification tasks. We use the **TF-IDF** technique to transform text into numerical features suitable for modelling.

The project also includes a **Streamlit** application with a user-friendly web interface for testing the model with custom tweets.

# Dataset 

You will have two CSV files:

**twitter_training.csv** (Training Data)

**twitter_validation.csv** (Validation Data)

## Training Data
Contains tweets labeled with their sentiment (Positive, Negative, or Neutral). Used to train the Logistic Regression model.

## Validation Data
Contains similarly labeled tweets. Used to evaluate the performance of the trained model.
The CSV files are assumed to have the following structure (with no header row by default):
```
Tweet_ID,entity,sentiment,Tweet_content
...
```
For example:
```
2401,Borderlands,Positive,"im getting on borderlands and i will murder you all ,"
2401,Borderlands,Positive,"I am coming to the borders and I will kill you all,"
...
```

# Project Structure

```
.
├── app.py                # Streamlit application for sentiment analysis
├── main.py               # Main script for training and evaluating the model
├── twitter_training.csv  # Training data (example)
├── twitter_validation.csv # Validation data (example)
├── sentiment_model.pkl   # Saved Logistic Regression model (generated after training)
├── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer
└── label_encoder.pkl     # Saved label encoder
```

# Setup Instructions

1. **Clone Repository**
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. **Create Virtual Environment (Recommended)**
   It is recommended to use a **virtual environment** to avoid package conflicts. You can use **venv** or **conda**.
   ```
   # Using venv
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```
3. **Install Dependencies**
   ```
   pip install pandas numpy scikit-learn nltk streamlit seaborn matplotlib
   ```
4. **Project Files**
   **main.py**: Script to train and evaluate the model. Also saves the trained model, TF-IDF vectorizer, and label encoder.

   **app.py**: Streamlit app for interactive sentiment analysis.

# Usage
1. **Train the model**
   1. Make sure you have **twitter_training.csv** and **twitter_validation.csv** in the project directory.
   2. Run:
      ```
      python main.py
      ```
   This will:
   * Load and preprocess the data.
   * Train a Logistic Regression model on the training data.
   * Evaluate the model on the validation data.
   * Display accuracy, classification report, and a confusion matrix.
   * Save the trained model and other artifacts (**sentiment_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl**).

2. Run the app
   ```
   streamlit run app.py
   ```
   * A new tab will open in your default browser.
   * Enter a tweet in the text area and click Analyze Sentiment.
   * The model’s prediction (Positive, Negative, or Neutral) will be displayed.

# How It Works

## Data Preprocessing
* **Lowercasing**: Converts all text to lowercase.
* **URL Removal**: Strips out any URLs (e.g., http://...).
* **Mentions and Hashtags**: Removes Twitter-specific tokens (e.g., @user, #topic).
* **Punctuation and Numbers**: Removes punctuation and numeric characters.
* **Tokenization**: Splits the text into individual words (tokens).
* **Stopwords Removal**: Eliminates common words (e.g., "the", "and", "to") using NLTK's English stopwords list.
* **Lemmatization**: Reduces words to their base form (e.g., "running" → "run").

## Feature Extraction with TF-IDF
TF-IDF (Term Frequency–Inverse Document Frequency):
Converts text into numerical features by measuring how frequently a term appears in a document (TF) and how important that term is across all documents (IDF).

## Logistic Regression Model
* Training: The model learns weights for each word feature to predict sentiment.
* Class Weights: class_weight='balanced' handles class imbalance by adjusting the importance of each class proportionally.
* Optimization: Uses an iterative algorithm to find the best weights that minimize error on the training set.

## Evaluation Metrics
* Accuracy: Percentage of correctly classified tweets.
* Classification Report: Provides precision, recall, and F1-score for each class.
* Confusion Matrix: Visual representation of correct and incorrect predictions.

# Future Improvements

1. **Use Different Models**: Experiment with **Naive Bayes**, **Support Vector Machines**, or **Transformer-based models** (e.g., BERT).
2. **Hyperparameter Tuning**: Use **GridSearchCV** or **RandomizedSearchCV** to find the best parameters for your model.
3. **Entity Context**: Incorporate the **entity** column (e.g., Borderlands, Amazon) into the analysis for entity-specific sentiment.
4. **Data Visualization**: Include more detailed visualizations, such as a sentiment distribution or a word cloud.

**Thank you for checking out the Twitter Sentiment Analysis project!** If you have any questions or suggestions, feel free to open an issue or submit a pull request.

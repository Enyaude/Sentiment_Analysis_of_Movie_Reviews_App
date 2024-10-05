import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

# Set up the app title
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review and the model will predict whether it is 'fresh' or 'rotten'.")

# Load and preprocess the dataset
@st.cache_data
def load_data():
    # Load the Rotten Tomatoes data
    rotten = pd.read_csv('rt_critics.csv')
    rotten = rotten[rotten['fresh'].isin(['fresh', 'rotten'])]
    rotten['fresh'] = rotten['fresh'].map(lambda x: 1 if x == 'fresh' else 0)
    return rotten

data = load_data()

# Create and fit the model
@st.cache_data
def train_model(data):
    ngram_range = (1, 2)
    max_features = 2000
    cv = CountVectorizer(ngram_range=ngram_range, max_features=max_features, binary=True, stop_words='english')
    
    # Fit the vectorizer
    words = cv.fit_transform(data.quote)
    
    X_train, X_test, y_train, y_test = train_test_split(words, data.fresh.values, test_size=0.25)
    
    model = BernoulliNB()
    model.fit(X_train, y_train)
    
    return model, cv

model, cv = train_model(data)

# Function to predict sentiment
def predict_sentiment(review):
    review_vector = cv.transform([review])
    prediction = model.predict(review_vector)
    probability = model.predict_proba(review_vector)
    
    sentiment = 'Fresh' if prediction == 1 else 'Rotten'
    return sentiment, probability

# User input
user_review = st.text_area("Enter your movie review here:")

# Prediction button
if st.button("Predict Sentiment"):
    if user_review:
        sentiment, probability = predict_sentiment(user_review)
        st.write(f"Sentiment: **{sentiment}**")
        st.write(f"Fresh Probability: {probability[0][1]:.2f}")
        st.write(f"Rotten Probability: {probability[0][0]:.2f}")
    else:
        st.write("Please enter a review to analyze.")

# Display dataset information
if st.checkbox("Show Dataset"):
    st.write(data.head())

# Footer
st.write("This app uses a Naive Bayes classifier to predict whether a movie review is 'fresh' or 'rotten' based on the words in the review.")


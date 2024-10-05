import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import re

# Load the trained model and necessary artifacts
cv = pickle.load(open('countVectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model_rf = pickle.load(open('model_rf.pkl', 'rb'))  # Ensure you save your model to this file

# Function to preprocess user input
def preprocess_input(user_input):
    stemmer = PorterStemmer()
    # Clean and preprocess the input text
    review = re.sub('[^a-zA-Z]', ' ', user_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    return ' '.join(review)
    

# Streamlit application layout
st.title("Amazon Alexa Review Sentiment Analysis")
st.write("Enter a review below and click 'Predict' to see if it's positive or negative!")

# Text input for user reviews
user_review = st.text_area("Review", "")

if st.button("Predict"):
    if user_review:
        # Preprocess the input
        preprocessed_review = preprocess_input(user_review)

        # Vectorize the input
        input_vector = cv.transform([preprocessed_review]).toarray()  # Convert sparse to dense

        # Reshape the input vector to match the expected input shape
        #if input_vector.shape[1] != 2500:  # Assuming 2500 is the expected number of features
         #   st.write("The input features do not match the expected number of features.")
         #       return
        
        # Scale the input using the previously fitted scaler
        input_scaled = scaler.transform(input_vector)  # Transform the input

        # Make prediction
        prediction = model_rf.predict(input_scaled)  # Use the scaled input

        # Display result
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.write("Please enter a review.")


# Optional: Display some visualizations
st.subheader("Rating Distribution Visualization")
data = pd.read_csv(r"C:\Users\sam\Desktop\amazon_alexa.tsv", delimiter='\t', quoting=3)

fig, ax = plt.subplots()
sns.countplot(x='rating', data=data, ax=ax)
ax.set_title('User Rating Distribution')
st.pyplot(fig)

fig2 = plt.figure(figsize=(10, 5))
data['feedback'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['green', 'red'], ax=fig2.gca())
plt.title('Feedback Distribution')
st.pyplot(fig2)

# Run the Streamlit app with the command
# streamlit run app.py

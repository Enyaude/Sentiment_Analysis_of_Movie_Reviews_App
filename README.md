## Sentiment Analysis with Movie Reviews


This GitHub repository explores sentiment analysis using movie reviews from Rotten Tomatoes. The project aims to identify words that appear more frequently in positive ("fresh") and negative ("rotten") reviews.

Project Overview:

Dataset: Rotten Tomatoes movie review dataset (link provided below)
Goal: Analyze word usage patterns in positive vs. negative reviews
Machine learning model: Bernoulli Naive Bayes
Evaluation: Identify words with highest positive/negative association
Data Acquisition and Preprocessing:

Import libraries: pandas, numpy, sklearn
Load data: Reads the CSV file containing movie reviews and their classifications ("fresh" or "rotten")

Preprocessing:
Handles missing values (if any)
Converts "fresh"/"rotten" labels to numerical values (e.g., 1 for "fresh", 0 for "rotten")
Applies text cleaning techniques (e.g., removing stop words, stemming/lemmatization)

Feature Engineering:

CountVectorizer: Converts textual reviews into numerical features based on word frequency (n-grams)
Example: "The movie is excellent!" becomes a vector where "excellent" has a higher frequency than "the" or "is".
Feature Selection (Optional):
Techniques like chi-squared test can be used to identify the most informative words for sentiment classification.

Model Training and Evaluation:

Model Selection: Bernoulli Naive Bayes is used due to its simplicity and efficiency for text classification tasks.
Train/Test Split: Splits the data into training and testing sets to evaluate model performance.
Model Training: Trains the Naive Bayes model on the training data.
Evaluation Metrics:
Accuracy: Measures the overall percentage of correct predictions
Precision/Recall: Measures the balance between true positives and false positives/negatives
F1-score: Harnesses precision and recall into a single metric
Confusion matrix: Visualizes model performance

Word Association Analysis:

Model Predictions: Uses the trained model to predict sentiment for unseen reviews.
Word Probabilities: Analyzes the model to identify words with the highest probability of appearing in "fresh" and "rotten" reviews.
Word Ranking: Ranks words based on their positive or negative association strength.

Results and Conclusion:

The project explores word usage patterns and how they correlate with positive and negative sentiment in movie reviews.
By analyzing words with high positive/negative association, we gain insights into how reviewers express their opinions.

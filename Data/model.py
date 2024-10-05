# Step 1 Importing packages and libraries for calculation and storing data in .ipnyb and .py files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re

# %pip install xgboost
# %pip install wordcloud

# Step 2 Importing Amazone Alexa Dataset and display First five Records
data = pd.read_csv(r"C:\Users\sam\Desktop\amazon_alexa.tsv", delimiter = '\t', quoting = 3)

# Step 3 Display top records
data.head()

# Step 4 Display no of rows and column
print(f"Dataset shape : {data.shape}")

# Step 5 Display Structure of the file
data.info()

# 6 Describing Descriptive Business Analytics
data.describe()

print(f"Feature names : {data.columns.values}")

data.tail()

data.isnull().any()

data.isnull().sum()

# Replace missing values in column 'verified_reviews' with the string 'missing'
data['verified_reviews'] = data['verified_reviews'].fillna('Missing a lot of this kinda os scene or cinema. Would love to do it right someday')

data[data['verified_reviews'].isna() == True]


if data['verified_reviews'].dtype != 'object':
    print("Warning: Some values in 'verified_reviews' might not be strings!")
else:
    print("Might be all strings")


data['length'] = data['verified_reviews'].apply(len)


data.head()


data.groupby('rating').describe()


data.groupby('feedback').describe()


print(f"'verified_reviews' column value: {data.iloc[10]['verified_reviews']}") #Original value
print(f"Length of review : {len(data.iloc[10]['verified_reviews'])}") #Length of review using len()
print(f"'length' column value : {data.iloc[10]['length']}") #Value of the column 'length'

data.head(11)

data.dtypes

len(data)

print(f"Rating value count: \n{data['rating'].value_counts()}")

#   Bar Plotting 

data['rating'].value_counts().plot.bar(color = 'green')
plt.title('Visualizing the Rating distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()

data['rating'].value_counts()

labels = '5', '4', '3', '2', '1'
sizes = [2286, 455, 161, 152, 96]
colors = ['green', 'magenta', 'pink', 'yellow', 'red']
explode = [0.001, 0.001, 0.001, 0.001, 0.001]

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)
plt.title(' pie chart representing ratings occuposition')
plt.show()

print(f"Rating value count - percentage distribution: \n{round(data['rating'].value_counts()/data.shape[0]*100,2)}")

fig = plt.figure(figsize=(7,7))

colors = ('red', 'green', 'blue','orange','yellow')

wp = {'linewidth':1, "edgecolor":'black'}

tags = data['rating'].value_counts()/data.shape[0]

explode=(0.1,0.1,0.1,0.1,0.1)

tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='Percentage wise distrubution of rating')

from io import  BytesIO

graph = BytesIO()

fig.savefig(graph, format="png")

# Date column

# 7 Converting date from object to date datatype and displacing year
data['date']=pd.to_datetime(data['date'])

# 8 Display year
data['date'].dt.year.value_counts()

# 9 Displaying mimimum date
data['date'].min()

# 10  Displaying Maximum date
data['date'].max()

# 11  Displaying data graphically
plt.figure(figsize=(15,16))
sns.countplot(x='date',data=data)
plt.xticks(rotation=90)
plt.show();

# 13  Displaying three months sales in numbers
data['date'].dt.month.value_counts()

# Assuming 'df_reviews' is your DataFrame with a 'date' column

# Calculate the offset for the last 3 months
offset = pd.DateOffset(months=3)

# Filter for dates within the last 3 months (excluding today)
last_three_months = data['date'] >= pd.Timestamp.today() - offset

# Filter the DataFrame
df_filtered = data[last_three_months]

# Create the countplot
sns.countplot(x=data['date'].dt.month, data=df_filtered)

# Add labels and title
plt.xlabel('Month')
plt.ylabel('Sales Count')
plt.title('Sales Distribution for the Last Three Months')

# Show the plot
plt.show()

# 15 Display numerical values of ratings
data.rating.value_counts()

# 14 Graphical Display of User Rating
sns.countplot(x='rating',data=data)

# Feedback column
# 1 is Positive and 0 is Negative

print(f"Feedback value count: \n{data['feedback'].value_counts()}")

review_0 = data[data['feedback'] == 0].iloc[1]['verified_reviews']
print(review_0)

review_1 = data[data['feedback'] == 1].iloc[1]['verified_reviews']
print(review_1)

 # Step 16 Display graph and Numerical values of User feedback for Amazon Alexa products
sns.countplot(x='feedback', data=data)
data.feedback.value_counts()

data['feedback'].value_counts().plot.bar(color = 'blue')
plt.title('Feedback distribution count')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.show()


print(f"Feedback value count - percentage distribution: \n{round(data['feedback'].value_counts()/data.shape[0]*100,2)}")

fig = plt.figure(figsize=(7,7))

colors = ('red', 'green')

wp = {'linewidth':1, "edgecolor":'black'}

tags = data['feedback'].value_counts()/data.shape[0]

explode=(0.1,0.1)

tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='Percentage wise distrubution of feedback')

#Feedback = 0
data[data['feedback'] == 0]['rating'].value_counts()

#Feedback = 1
data[data['feedback'] == 1]['rating'].value_counts()

data['length'].describe()

sns.histplot(data['length'],color='blue').set(title='Distribution of length of review ')

sns.histplot(data[data['feedback']==0]['length'],color='cyan').set(title='Distribution of length of review if feedback = 0')
sns.histplot(data[data['feedback']==1]['length'],color='green').set(title='Distribution of length of review if feedback = 1')

data.groupby('length')['rating'].mean().plot.hist(color = 'blue', figsize=(7, 6), bins = 20)
plt.title(" Review length wise mean ratings")
plt.xlabel('ratings')
plt.ylabel('length')
plt.show()

#Step 18 Generating Wordcloud
neg=data[data['feedback']==0]
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#Step 19 Generating Wordcloud
neg=data[data['feedback']==0]

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

text =neg['verified_reviews'].values
wordcloud=WordCloud(
    width=3000,
    height=2000,
    background_color='black',
    stopwords = STOPWORDS).generate(str(text))

fig=plt.figure(
    figsize=(40,30),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.tick_params(axis='both',labelsize=14)
plt.show()

# Step 21 Sentiment variation Plot
data['variation'].value_counts().plot.bar()

print(f"Variation value count: \n{data['variation'].value_counts()}")

data['variation'].value_counts().plot.bar(color = 'orange')
plt.title('Variation distribution count')
plt.xlabel('Variation')
plt.ylabel('Count')
plt.show()

print(f"Variation value count - percentage distribution: \n{round(data['variation'].value_counts()/data.shape[0]*100,2)}")

# Step 22 Sentiment Variation Distribut
sns.countplot(x='variation',data=data)
plt.title('Variation Distribution in Alexa')
plt.xlabel('Variations')
plt.ylabel('count')
plt.xticks(rotation='vertical')
plt.show()

data.groupby('variation')['rating'].mean()

data.groupby('variation')['rating'].mean().sort_values().plot.bar(color = 'brown', figsize=(11, 6))
plt.title("Mean rating according to variation")
plt.xlabel('Variation')
plt.ylabel('Mean rating')
plt.show()

# Step 17 Length of Review in number and graphs
data['length']=data['verified_reviews'].apply(lambda x:len(x.split()))
data.head()
plt.hist(x='length',data=data,bins=30)
data.length.describe()

cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(data.verified_reviews)

reviews = " ".join([review for review in data['verified_reviews']])
                        
# Initialize wordcloud object
wc = WordCloud(background_color='black', max_words=80)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(reviews))
plt.title('Wordcloud for all reviews', fontsize=10)
plt.axis('off')
plt.show()

neg_reviews = " ".join([review for review in data[data['feedback'] == 0]['verified_reviews']])
neg_reviews = neg_reviews.lower().split()

pos_reviews = " ".join([review for review in data[data['feedback'] == 1]['verified_reviews']])
pos_reviews = pos_reviews.lower().split()

#Finding words from reviews which are present in that feedback category only
unique_negative = [x for x in neg_reviews if x not in pos_reviews]
unique_negative = " ".join(unique_negative)

unique_positive = [x for x in pos_reviews if x not in neg_reviews]
unique_positive = " ".join(unique_positive)

wc = WordCloud(background_color='black', max_words=90)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_negative))
plt.title('Wordcloud for negative reviews', fontsize=10)
plt.axis('off')
plt.show()

wc = WordCloud(background_color='red', max_words=50)

# Generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_positive))
plt.title('Wordcloud for positive reviews', fontsize=10)
plt.axis('off')
plt.show()

corpus = []
stemmer = PorterStemmer()
for i in range(0, data.shape[0]):
  review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
  review = review.lower().split()
  review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
  review = ' '.join(review)
  corpus.append(review)

  # Using countvectorizer to create bag of words

  cv = CountVectorizer(max_features = 2500)

#Storing independent and dependent variables in X and y
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

pickle.dump(cv, open('countVectorizer.pkl', 'wb'))

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")

print(f"X train max value: {X_train.max()}")
print(f"X test max value: {X_test.max()}")

scaler = MinMaxScaler()

X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

pickle.dump(scaler, open('scaler.pkl', 'wb'))

#Fitting scaled X_train and y_train on Random Forest Classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)

print("Training Accuracy :", model_rf.score(X_train_scl, y_train))
print("Testing Accuracy :", model_rf.score(X_test_scl, y_test))

y_preds = model_rf.predict(X_test_scl)

cm = confusion_matrix(y_test, y_preds)
cm

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_rf.classes_)
cm_display.plot()
plt.show()

# K fold cross-validation

accuracies = cross_val_score(estimator = model_rf, X = X_train_scl, y = y_train, cv = 10)

print("Accuracy :", accuracies.mean())
print("Standard Variance :", accuracies.std())

params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}

cv_object = StratifiedKFold(n_splits = 2)

grid_search = GridSearchCV(estimator = model_rf, param_grid = params, cv = cv_object, verbose = 0, return_train_score = True)
grid_search.fit(X_train_scl, y_train.ravel())


print("Best Parameter Combination : {}".format(grid_search.best_params_))

print("Cross validation mean accuracy on train set : {}".format(grid_search.cv_results_['mean_train_score'].mean()*100))
print("Cross validation mean accuracy on test set : {}".format(grid_search.cv_results_['mean_test_score'].mean()*100))
print("Accuracy score for test set :", accuracy_score(y_test, y_preds))

model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)


print("Training Accuracy :", model_xgb.score(X_train_scl, y_train))
print("Testing Accuracy :", model_xgb.score(X_test_scl, y_test))

y_preds = model_xgb.predict(X_test)

cm = confusion_matrix(y_test, y_preds)
print(cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_xgb.classes_)
cm_display.plot()
plt.show()

 # Decision Tree Classifier

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_scl, y_train)

print("Training Accuracy :", model_dt.score(X_train_scl, y_train))
print("Testing Accuracy :", model_dt.score(X_test_scl, y_test))

y_preds = model_dt.predict(X_test)

cm = confusion_matrix(y_test, y_preds)
print(cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_dt.classes_)
cm_display.plot()
plt.show()



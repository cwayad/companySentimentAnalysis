#Import python libraries
import re, json, os, nltk
import pandas as pd
import numpy as np
from sklearn import linear_model
from nltk.corpus import stopwords
import Word2VecUtility
from sklearn.feature_extraction.text import CountVectorizer

#loading training data...
print "loading the training set ...\n"
with open('data/Headline_Trainingdata.json') as f:
   data = json.load(f)

#Convert the file into a dataframe
train = pd.DataFrame(data)

#Preprocessing 
# Initialize an empty list to hold the clean headlines
clean_train_headlines = []
print "pre-processing train data..."
# Loop over each headline;
print "Cleaning and parsing the training set ...\n"
for i in xrange( 0, len(train["title"])):
    clean_train_headlines.append(" ".join(Word2VecUtility.headline_to_wordlist(train["title"][i], True)))

# ****** Create a bag of words from the training set
#
print "Creating the bag of words...\n"


# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                        tokenizer = None,    \
                        preprocessor = None, \
                        stop_words = None,   \
                        max_features = 5000)


#fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_headlines)
# Numpy arrays are easy to work with, so convert the result to an
# array
np.asarray(train_data_features)

# ******* Train a linear regressor using the bag of words
#
print "Training the linear regression model..."
# Fit the Linear regression to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
regr = linear_model.LinearRegression()
regr.fit(train_data_features, train["sentiment"])


#loading test data...
print "loading the test set ...\n"
with open('data/Headlines_Testdata.json') as f:
   data = json.load(f)

#Convert the file into a dataframe
test = pd.DataFrame(data)
# Create an empty list and append the clean headlines one by one
clean_test_headlines = []

print "Cleaning and parsing the test set ...\n"
for i in xrange(0,len(test["title"])):
	clean_test_headlines.append(" ".join(Word2VecUtility.headline_to_wordlist(test["title"][i], True)))

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_headlines)
np.asarray(test_data_features)

print "Predicting test labels...\n"
sentiments_pred = regr.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"Id":test["id"],"Company":test["company"], "Sentiment":sentiments_pred} )

# Use pandas to write the json output file
output.to_json(os.path.join(os.path.dirname(__file__), 'results', 'Bag_of_Words_model.json'), orient='index')
print "Wrote results to Bag_of_Words_model.json" 


#Next steps : 
#1: applying SVM, and Keras/LSTM and other methods if possible
#2: Evaluate the results (accuracy, recall and mean MSE ...)
#3: conclude the best accuracy (and the why!)

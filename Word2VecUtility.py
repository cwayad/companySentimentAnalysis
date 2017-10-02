import re, nltk, string
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.sentiment.util import mark_negation

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class Word2VecUtility(object):
    """Word2VecUtility is a utility class for processing raw text into segments for further learning"""

def headline_to_wordlist( title, remove_stopwords=False ):
	# Function to convert a document to a sequence of words,
	# preprocessing the title.  Returns a list of words.
	#
	# . Convert words to lower case 
	title= title.lower()
	#
	# . Remove non-letters
	title = re.sub("[^a-zA-Z]"," ",title)
	#
	# . Split/tokenize words
	words = tokenize (title)
	#
	# . Handle negation before removing stop words!
	words = handle_negation(words)
	#
	# . Remove punctuation
	#words = remove_punctuation(words)
	#
	# . Stemming
	words =  apply_stemming(words)
	#or Lemmatization
	#words= apply_lemmatization(words)
	#
	# . Remove stop words (false by default)
	if remove_stopwords:
	    stops = set(stopwords.words("english"))
	    words = [w for w in words if not w in stops]
	#
	# . Return a list of words
	return(words)

def apply_stemming(aWords):
	# uses Porter stemming
	stemmer = PorterStemmer()
	return [str(stemmer.stem(word)) for word in aWords] # output of stemmer.stem(term) is u'string

def apply_lemmatization(awords):
	# uses WordNetLemmatizer
	lmtzr = WordNetLemmatizer()
	return [lmtzr.lemmatize(word) for word in awords]

def remove_punctuation(aWords):
	return [x for x in aWords if not re.fullmatch('[' + string.punctuation + ']+', x)]

def handle_negation(aWords):
	# http://www.nltk.org/_modules/nltk/sentiment/util.html#mark_negation
	return mark_negation(aWords)
		
def tokenize(sentence):	
	# returns a list of word tokens and also remove multiple spaces and \t, \n etc.
	return word_tokenize(sentence)

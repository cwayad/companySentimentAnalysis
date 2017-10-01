import re, nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
nltk.download('stopwords')
from nltk.corpus import stopwords

class Word2VecUtility(object):
    """Word2VecUtility is a utility class for processing raw text into segments for further learning"""

def headline_to_wordlist( title, remove_stopwords=False ):
	# Function to convert a document to a sequence of words,
	# optionally removing stop words.  Returns a list of words.
	#
	# 1. Remove non-letters
	headline_text = re.sub("[^a-zA-Z]"," ",title)
	#
	# 2. Convert words to lower case and split them
	words = headline_text.lower().split()
	#
	# 3. Remove stop words (false by default)
	if remove_stopwords:
	    stops = set(stopwords.words("english"))
	    words = [w for w in words if not w in stops]
	#
	#4. Return a list of words
	return(words)

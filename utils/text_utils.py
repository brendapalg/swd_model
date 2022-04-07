# -*- coding: utf-8 -*-
"""
Text preprocessing and analysis

@author: Brenda Palma, Eric Coxon

This module includes functions for cleaning text and  performing sentiment analysis
It is imported by the main_stocktrac module.
"""

# =============================================================================
# Imports
# =============================================================================      
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords_english = stopwords.words('english')
stopwords_english += ['youll', 'ive']


# =============================================================================
# Functions
# =============================================================================  

def remove_tweetElements(tweet):
    '''

    Parameters
    ----------
    tweet : str
        Text of a tweet.

    Returns
    -------
    tw : str
        Tweet without mentions, hyperlinks and RTs.

    '''
    tw = re.sub(r'http\S+', ' ', tweet) #Remove urls
    tw = re.sub(r'RT @[\w_-]+',' ',tw) #Remove RT text
    tw = re.sub('@[\w_-]+',' ',tw) #Remove mentions 
    return tw


def clean_text(text, stopwords = stopwords_english, lemmatize=False):
    '''
    Parameters
    ----------
    text : str
        The text to be cleaned.
    stopwords : list, optional
        List of common words that are to be removed during the transformation. The default is [].

    Returns
    -------
    txt : str
        The text after applying the mentioned transformations.

    '''
    txt = text.lower() # to lower case
    txt = re.sub(r'\d', ' ', txt) #Remove numbers
    txt = re.sub(r'[^\w]', ' ', txt) #Remove punctuation
    txt = word_tokenize(txt) 
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        txt = ' '.join([lemmatizer.lemmatize(word) for word in txt if (not word in stopwords_english and len(word)>2)])
    else:
        txt = ' '.join([word for word in txt if (not word in stopwords and len(word)>2)]) #Remove stopwords
    txt = txt.strip()
    return txt


def get_sentiment(text, sentiment_dict):
    '''
    Parameters
    ----------
    text : str
        Text that will be analyzed. 
    sentiment_dict : pandas.DataFrame
        Dataframe containing the pairs of words and polarities that are used for the sentiment analysis.
        Column names should be 'word' and 'polarity'

    Returns
    -------
    float
        Returns the sentiment for the given text.

    '''
    aux = text.split()
    if len(aux)<1:
        out = [0]
    else:
        out = [sentiment_dict.loc[sentiment_dict['word'] == w, 'polarity'].values[0] if w in sentiment_dict['word'].values else 0 for w in aux]
    return round(np.mean(out),3)



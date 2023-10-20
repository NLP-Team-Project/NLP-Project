import pandas as pd
import unicodedata
import re
import nltk
import os

from requests import get
from bs4 import BeautifulSoup

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


def basic_clean(text):
    
    
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    
    text = re.sub(r"[^a-z\w\s]", '', text)
    
    return text


def tokenize(text):
    
    tokenizer = nltk.tokenize.ToktokTokenizer()

    text = tokenizer.tokenize(text, return_str=True)
    
    text = re.sub(r"[^a-z\w\s]", '', text)
    
    text = re.sub(r'\d', '', text, count = 1)
    
    return text


def stem(text):
    
    ps = nltk.porter.PorterStemmer()
    
    stems = [ps.stem(word) for word in text.split()]
    
    text_stemmed = ' '.join(stems)
    
    return text_stemmed


def lemmatize(text):
    
    wnl = nltk.stem.WordNetLemmatizer()
    
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    
    text_lemmatized = ' '.join(lemmas)

    return text_lemmatized


def remove_stopwords(text, extra_words = None, exclude_words = None):
    
    stopword_list = stopwords.words('english')
    
    if exclude_words is not None:
        
        for w in exclude_words:
        
            stopword_list.remove(w)
    
    if extra_words is not None:
        
        for w in extra_words:
        
            stopword_list.append(w)
    
    words = text.split()
    
    filtered_words = [w for w in words if w not in stopword_list]

    print('Removed {} stopwords'.format(len(words) - len(filtered_words)))
    print('---')

    text_without_stopwords = ' '.join(filtered_words)

    return text_without_stopwords


def advanced_clean(text, l = False, s = False, extra_words = None, exclude_words = None):
    
    if extra_words is None:
        extra_words = []
        
    if exclude_words is None:
        exclude_words = [] 
    
    text = basic_clean(text)
    text = tokenize(text)
    text = remove_stopwords(text, extra_words = extra_words, exclude_words = exclude_words)
    
    if l is not False:
        
        text = lemmatize(text)
    
    if s is not False:
    
        text = stem(text)
    
    return text
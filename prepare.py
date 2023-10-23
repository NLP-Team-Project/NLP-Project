import pandas as pd
import unicodedata
import re
import nltk
import os
import markdown

from requests import get
from bs4 import BeautifulSoup

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


def basic_clean(text):
    
    text = text.lower()
    
    text = re.sub(r"http\S+|www\S+", "", text)    
    
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             )    
    
    text = markdown.markdown(text)
    
    soup = BeautifulSoup(text, 'html.parser')
    
    text = soup.get_text()
    
    text = re.sub(r"[^a-z\w\s]", '', text)
    
    pattern = r"[^a-zA-Z0-9\s'\+]|(?<!\w)C\+\+(?!\w)"
    
    text = re.sub(pattern, "", text)
    
    return text


def tokenize(text):
    
    tokenizer = nltk.tokenize.ToktokTokenizer()

    text = tokenizer.tokenize(text, return_str=True)
    
    text = re.sub(r"[^a-z\w\s]", '', text)
    
    text = re.sub(r"\s\d{1}\s", "", text)
    
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


def actual_languages(df):
    
    valid_languages = ["Python", 'HTML', "JavaScript", "R", "Java", "TypeScript", "PHP", "Ruby", "C#", "C++", "Dart", "Kotlin",
                       "Objective-C", "Swift", "Go", "Rust", "C", "Elixir", "CoffeeScript", "MATLAB", "Visual Basic .NET",
                       "Scala", "Haskell", "Stata", "Haxe", "Lua", "Perl", "Clojure"]

    # Use the isin method to filter the DataFrame
    df = df[df['language'].isin(valid_languages)]
    
    # List of languages to rename to 'other'
    languages_to_rename = ['HTML', 'Java', 'PHP', 'Ruby', 'C#', 'C++', 'Dart', 'Kotlin', 'Objective-C', 'Swift', 'Go', 'Rust',
                           'C', 'Elixir', 'Visual Basic .NET', 'MATLAB', 'CoffeeScript', 'Scala', 'Perl', 'Lua', 'Haskell',
                           'Haxe', 'Stata', 'Clojure']

    # Replace the specified languages with 'other'
    df['language'] = df['language'].replace(languages_to_rename, 'other')
    
    return df


def labels_count(df):
    
    labels = pd.concat([df.language.value_counts(),
                    df.language.value_counts(normalize=True)], axis=1)

    labels.columns = ['n', 'percent']

    return labels
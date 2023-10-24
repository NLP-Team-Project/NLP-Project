import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


import unicodedata
import markdown
import re
import nltk
import os
import prepare as p


from requests import get
from bs4 import BeautifulSoup

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from scipy.stats import chi2_contingency


def get_words(df):

    # Here I call my advanced clean function on my data filtered by each type of language and its 
    # corresponding readme contents then split in to a list of words
    python_words = p.advanced_clean(' '.join(df[df.language == 'Python']['readme_contents']), l = True, extra_words = ['python','eww','sport','data', 'team', 'game', 'season', 'league', 'player', 'ball', 'r', 'javascript', 'typescript','html', "javascript", "r", "java", "php", "ruby", "c#", "c++", "dart", "kotlin", "objective-c", "swift", "go", "rust", "c", "elixir", "coffeescript", "matlab", "visual basic .net", "scala", "haskell", "stata", "haxe", "lua", "perl", "clojure" ]).split()
    javascript_words = p.advanced_clean(' '.join(df[df.language == 'JavaScript']['readme_contents']), l = True, extra_words = ['python','eww','sport','data', 'team', 'game', 'season', 'league', 'player', 'ball', 'r', 'javascript', 'typescript','html', "javascript", "r", "java", "php", "ruby", "c#", "c++", "dart", "kotlin", "objective-c", "swift", "go", "rust", "c", "elixir", "coffeescript", "matlab", "visual basic .net", "scala", "haskell", "stata", "haxe", "lua", "perl", "clojure" ]).split()
    r_words = p.advanced_clean(' '.join(df[df.language == 'R']['readme_contents']), l = True, extra_words = ['python','eww','sport','data', 'team', 'game', 'season', 'league', 'player', 'ball', 'r', 'javascript', 'typescript','html', "javascript", "r", "java", "php", "ruby", "c#", "c++", "dart", "kotlin", "objective-c", "swift", "go", "rust", "c", "elixir", "coffeescript", "matlab", "visual basic .net", "scala", "haskell", "stata", "haxe", "lua", "perl", "clojure" ]).split()
    typescript_words = p.advanced_clean(' '.join(df[df.language == 'TypeScript']['readme_contents']), l = True, extra_words = ['python','eww','sport','data', 'team', 'game', 'season', 'league', 'player', 'ball', 'r', 'javascript', 'typescript','html', "javascript", "r", "java", "php", "ruby", "c#", "c++", "dart", "kotlin", "objective-c", "swift", "go", "rust", "c", "elixir", "coffeescript", "matlab", "visual basic .net", "scala", "haskell", "stata", "haxe", "lua", "perl", "clojure" ]).split()
    other_words = p.advanced_clean(' '.join(df[df.language == 'other']['readme_contents']), l = True, extra_words = ['python','eww','sport','data', 'team', 'game', 'season', 'league', 'player', 'ball', 'r', 'javascript', 'typescript','html', "javascript", "r", "java", "php", "ruby", "c#", "c++", "dart", "kotlin", "objective-c", "swift", "go", "rust", "c", "elixir", "coffeescript", "matlab", "visual basic .net", "scala", "haskell", "stata", "haxe", "lua", "perl", "clojure" ]).split()
    all_words = p.advanced_clean(' '.join(df.readme_contents), l = True, extra_words = ['python','eww','sport','data', 'team', 'game', 'season', 'league', 'player', 'ball', 'r', 'javascript', 'typescript','html', "javascript", "r", "java", "php", "ruby", "c#", "c++", "dart", "kotlin", "objective-c", "swift", "go", "rust", "c", "elixir", "coffeescript", "matlab", "visual basic .net", "scala", "haskell", "stata", "haxe", "lua", "perl", "clojure" ]).split()

    # Here I create a list of extra words I want to remove from each list of words
    extra_words = ['python','python3', 'eww','sport','data', 'team', 'game', 'season', 'league', 'player', 'ball', 'r', 'javascript', 'typescript','html', "javascript", "r", "java", "php", "ruby", "c#", "c++", "dart", "kotlin", "objective-c", "swift", "go", "rust", "c", "elixir", "coffeescript", "matlab", "visual basic .net", "scala", "haskell", "stata", "haxe", "lua", "perl", "clojure" ]

    extra_words1 = (pd.Series(python_words).value_counts()[pd.Series(python_words).value_counts() < 51]).index.values
    extra_words2 = (pd.Series(javascript_words).value_counts()[pd.Series(javascript_words).value_counts() < 51]).index
    extra_words3 = (pd.Series(r_words).value_counts()[pd.Series(r_words).value_counts() < 51]).index
    extra_words4 = (pd.Series(typescript_words).value_counts()[pd.Series(typescript_words).value_counts() < 51]).index
    extra_words5 = (pd.Series(other_words).value_counts()[pd.Series(other_words).value_counts() < 51]).index


    # Here I rejoin my words so that I can add extra words to my stopword list
    python_words = ' '.join(python_words)
    javascript_words = ' '.join(javascript_words)
    r_words = ' '.join(r_words)
    typescript_words = ' '.join(typescript_words)
    other_words = ' '.join(other_words)


    # Here I remove the words from the list i created above
    python_words = p.remove_stopwords(python_words, extra_words = extra_words1)
    javascript_words = p.remove_stopwords(javascript_words, extra_words = extra_words2)
    r_words = p.remove_stopwords(r_words, extra_words = extra_words3)
    typescript_words = p.remove_stopwords(typescript_words, extra_words = extra_words4)
    other_words = p.remove_stopwords(other_words, extra_words = extra_words5)


    # Here I remove the words from the list i created above
    python_words = p.remove_stopwords(python_words, extra_words = extra_words).split()
    javascript_words = p.remove_stopwords(javascript_words, extra_words = extra_words).split()
    r_words = p.remove_stopwords(r_words, extra_words = extra_words).split()
    typescript_words = p.remove_stopwords(typescript_words, extra_words = extra_words).split()
    other_words = p.remove_stopwords(other_words, extra_words = extra_words).split()

    return python_words, javascript_words, r_words, typescript_words, other_words, all_words


def freq(df):

    python_words, javascript_words, r_words, typescript_words, other_words, all_words = get_words(df)
    
    # Here I create a series out of my list of words and get the total count for each unique word
    python_freq = pd.Series(python_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    r_freq = pd.Series(r_words).value_counts()
    typescript_freq = pd.Series(typescript_words).value_counts()
    other_freq = pd.Series(other_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()


    # Here I filter out all the words fir each list that occur more than 50 times
    python_freq = python_freq[python_freq > 50]
    javascript_freq = javascript_freq[javascript_freq > 50]
    r_freq = r_freq[r_freq > 10]
    typescript_freq = typescript_freq[typescript_freq > 10]
    other_freq = other_freq[other_freq > 50]
    all_freq = all_freq[all_freq > 50]

    return python_freq, javascript_freq, r_freq, typescript_freq, other_freq, all_freq


def count_prop(df):
    
    python_words, javascript_words, r_words, typescript_words, other_words, all_words = get_words(df)
    
    python_freq, javascript_freq, r_freq, typescript_freq, other_freq, all_freq = freq(df)
    
    p = all_freq / len(all_words) * 100
    n = all_freq
    
    # Here I concat two series together which outputs the count for each language appearance and the porportion it make up
    labels = pd.concat([n,p], axis = 1)
    labels.columns = ['n', 'percent']
    return labels


def top_10_words(df):
    
    python_words, javascript_words, r_words, typescript_words, other_words, all_words = get_words(df)
    python_freq, javascript_freq, r_freq, typescript_freq, other_freq, all_freq = freq(df)
    
    # Sort the dictionary by values in descending order and take the top 10
    top_10 = dict(sorted(all_freq.items(), key=lambda item: item[1], reverse=True)[:10])

    # Extract words and their frequencies
    words = list(top_10.keys())
    frequencies = list(top_10.values())

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(words, frequencies, color='lightseagreen')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title('Top 10 Most Common Words')
    plt.gca().invert_yaxis()  # To display the most common word at the top
    sns.despine(bottom = True, left = True)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.show()
    
    
    
def plt_average_length(df):
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='language', y='readme_length', errorbar = None)
    ax = plt.gca()

    for p in ax.patches:
        value = float(round(p.get_height(), 2)) * 100  # Get the height (value) of each bar

        label = f"{int(round(value))}%" 

    plt.title('Average README Length by Programming Language')
    sns.despine(bottom = True, left = True)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.show()
    
    
def test1(df):
    
    t_stat, p_value = stats.ttest_ind(df[df.language == 'Python']['readme_length'], df[df.language == 'R']['readme_length'])

    print(f't = {t_stat}')
    print(f'p = {p_value}')

    if p_value < 0.05:
        
        print(f"Reject the null hypothesis. There is a significant difference between README lengths for Python and R.")

    else:
        
        print(f"Fail to reject the null hypothesis. There is no significant difference between README lengths for Python and R.")
        
        
def plt_unique_words(df):
    
    plt.figure(figsize=(12, 6))
    plt.bar(df['language'], df['unique_word'], color = 'lightseagreen')
    plt.xlabel('language')
    plt.ylabel('Unique Words Count')
    plt.title('Number of Unique Words by Programming Language')
    sns.despine(bottom = True, left = True)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.show()
    
    
def test2(df):
    
    contingency_table = pd.crosstab(df['language'], df['unique_word'])

    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f'Chi2 = {chi2}')
    print(f'P = {p}')

    if p < 0.05:  # Choose your significance level
        print("Reject the null hypothesis. There is an association between programming languages and unique word counts.")
    else:
        print("Fail to reject the null hypothesis. There is no significant association.")

import unicodedata
import re

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

import pandas as pd

# nltk.download('wordnet')
# nltk.download('omw-1.4')


def basic_clean(filthy_data):
    filthy_data = filthy_data.lower()
    filthy_data = unicodedata.normalize('NFKD', filthy_data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    clean_data = re.sub(r"[^a-z0-9'\s]", "", filthy_data)
    return clean_data


def tokenize(data):
    tokenizer = ToktokTokenizer()
    data = tokenizer.tokenize(data, return_str=True)
    return data


def remove_stopwords(string, extra_words=[], exclude_words=[]):
    """ This function takes in a string, optional extra_words and exclued_words parameters with default empty lists
     and returns a string """
    stopword_list = stopwords.words('english')
    # use set casting to remove any excluded stopwords
    stopword_set = set(stopword_list) - set(exclude_words)
    # add in extra words to stopwords set using a union
    stopword_set = stopword_set.union(set(extra_words))
    # split the document by spaces
    words = string.split()
    # every word in our document that is not a stopword
    filtered_words = [word for word in words if word not in stopword_set]
    # join it back together with spaces
    string_without_stopwords = ' '.join(filtered_words)
    return string_without_stopwords


def lemmatize(data):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in data.split()]
    lemmatized_data = ' '.join(lemmas)
    return lemmatized_data


def stem(data):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in data.split()]
    stemmed_data = ' '.join(stems)
    return stemmed_data


def cleanse(dataframe, col='', stemm=True, lem=True, extra_words=[], exclude_words=[]):
    df = dataframe.copy()
    df['clean'] = df[col].apply(basic_clean)
    df['clean'] = df['clean'].apply(remove_stopwords, extra_words=extra_words, exclude_words=exclude_words)
    if stemm:
        df['stemmed'] = df.clean.apply(stem)
    if lem:
        df['lemmatized'] = df.clean.apply(lemmatize)
    return df


def train_val_test(df, strat='None', seed=100, stratify=False, print_shape=True):  # Splits dataframe
    """ This function will split my data into train, validate and test. It has the option to stratify."""
    if stratify:  # Will split with stratify if stratify is True
        train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
        val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
        if print_shape:
            print(train.shape, val.shape, test.shape)
        return train, val, test
    if not stratify:  # Will split without stratify if stratify is False
        train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
        val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
        if print_shape:
            print(f' train: {train.shape},  val: {val.shape},  test: {test.shape}')
        return train, val, test


def split_xy(df, target=''):
    """This function will split x and y according to the target variable."""
    x_df = df.drop(columns=target)
    y_df = df[target]
    return x_df, y_df  # Returns dataframe


def replace_values_not_in_list(series, mylist, false_value):
    new_values = []
    for value in series:
        if value in mylist:
            new_values.append(value)
        else:
            new_values.append(false_value)
    return new_values


def top_languages(repo, n):
    top_langs = list(repo.language.value_counts().index[:n])
    filtered_languages = replace_values_not_in_list(repo.language, top_langs, 'Other')
    return filtered_languages


def word_counts(dataframe, col, data):
    df = dataframe.copy()
    uniques = df[col].unique()
    col_names = ['all'] + list(df[col].unique())

    word_count = []

    all_words = df[data]
    all_words = pd.Series(' '.join(all_words).split()).value_counts()
    word_count.append(all_words)

    for name in uniques:
        words = df[data][df[col] == name]
        words = pd.Series(' '.join(words).split()).value_counts()
        word_count.append(words)

    total_counts = pd.concat(word_count, axis=1) \
        .set_axis(col_names, axis=1).fillna(0) \
        .apply(lambda s: s.astype(int))

    return total_counts


def word_appearances(word_count, og_repo):
    unique_words = list(word_count.index)
    total_appearances = []

    for word in unique_words:
        n = 0
        for row in og_repo.clean:
            words = set(row.split())
            if word in words:
                n += 1
        total_appearances.append(n)


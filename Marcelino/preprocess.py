from sklearn.model_selection import train_test_split


def xy_split(df):
    '''
    This function splits your data for modeling. 
    
    Parameter: 
    df = data
    
    Output:
    This function returns subsets of your data. One with all columns except your target variable, and the other with only the target variable.
    '''
    
    return df.drop(columns = 'language'), df.language



def train_val_test(df, strat='None', seed= 42, stratify=False, print_shape=True):  # Splits dataframe
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








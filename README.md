# NLP-Project
Scraping githubs and discovering most frequent program languages.

## Description


## Goal

* Scrape repo names from Github, and use api requests to create a database of repos
* Explore data to see what words are most common in each respective language's readmes
* Build a classification model that predicts program languages used in each repo on Github


## Data dictionary

| Column         | Column_type | Data_type| Description              |
|----------------|-------------|----------|--------------------------|
|repo            |Feature      |string    |Name of the repositiory.  |
|language        |Target       |string    |Programming language used.|
|readme_contents |Feature      |string    |Contents for every readme.|


## Planning:
- Create a project plan
- Create a web scraper to acquire repo names and use api request to acquire repo data
- Clean up the data by tokenizing it, lammatizing, stemming, remove non ASCII characters, etc...
- Explore data to find what words are most common in what language's repo
- Create models to predict primary programming language of a repo


## Acquisition:
  
- Create a web scraper to acquire repo names
- Use api request to acquire repo data
- Cache data
- Data acquired from Github


## Preparation
- Remove all non alphanumeric characters
- Lowercase everything
- Tokenize all words
- Drop all nulls, and invalid data types
- Lemmatize and stem data


## Exploration & pre-processing:
- Made visuals and used stats to understand which features had a significance
- Binned data for better visuals
- Created word-count dataframes to see in how many documents a word appears in and how many times a word appears


## Modeling:
- Modeling will be focused around improving accuracy score of the model
- Baseline had an accuracy of 43%
- Models improved significantly when class_weight was set to 'balanced'
- Models consistenly scored in the 60s
- Removing words with very high/low usage helped slightly


## Key findings, recommendations, and takeaways
We can create models to predict the primary language of a repository based off the readme contents in that repository. By web-scraping repository names from Github and looping through api requests we created a data frame of repository information. Through exploration we found many words that appeared in many many repositories and words that appeared in very few. Removing these words seemed to help model accuracy just slightly. Our best models were able to accurately predict the language of a repository around 2/3 of the time. Binning languages into a single ‘other’ category helped incredibly, however it also means we are not guessing the actual language of a repository but rather a ‘group’ of languages.


## Enviroment setup: 
- Install Conda, Python, MySql, VS Code or Jupyter Notebook
- Github api token

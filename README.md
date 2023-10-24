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
- Create models to 

## Acquisition:
  
  - Create a web scraper to acquire repo names and use api request to acquire repo data

- Data acquired from Coursera into a csv file

## Preparation
- Renamed columns& lowercased column names
- No missing values
- Dropped LoanID column
- Split data 70%,15%,15%

## Exploration & pre-processing:
- Made visuals and used stats to understand which features had a significance
- Binned data for better visuals

## Modeling:
- Decision tree and random forest models with balanced weight parameters perform worse than the baseline
- Distribution of default binary values heavily concentrated on one value
- Knearest tree is weighing one outcome significantly more than the other

## Delivery:
- Deployed my model and a reproducable report
- Made recommendations

## Key findings, recommendations, and takeaways
- Distribution of defaults significantly concentrated on non defaults (0)
- Interest rates, loan amount, and age seem to drive borrrowers to default on loans
- Target loan amounts lowers than 150k
- Require higher qualifications for younger population 
- Target borrowers with low interest rates

## Instructions or an explanation of how someone else can reproduce project and findings

Enviroment setup: 
- Install Conda, Python, MySql, VS Code or Jupyter Notebook
- Clone this repo 

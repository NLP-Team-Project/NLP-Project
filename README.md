# NLP-Project
Scraping githubs and discovering most frequent program languages.

## Description


## Goal
* The purpose of this model is to predict borrowers that default.
* My goal is to find specific features that drive defaults.

## Initial hypotheses

* Null Hypothesis: Features do not drive borrowers to default
* Alternative Hypothesis: Features drive borrowers to default

## Data dictionary

| Column         | Column_type | Data_type| Description              |
|----------------|-------------|----------|--------------------------|
|repo            |Feature      |string    |Name of the repositiory.  |
|language        |Target       |string    |Programming language used.|
|readme_contents |Feature      |string    |Contents for every readme.|

## Planning:
- Generate questions to ask about the data set based off of what I want my model to predict. Do any features have an impact on defaults?. What features significantly drive defaults?
- Determine the format. Final report should be in .ipynb, Modules should be in .py, Predictions should be in .csv.
- Determine audience and develop speech and presention accordingly. Audience will be lead data scientist.
- Determine significance between features and defaults.
- Develop my null hypothsisis and alternative hypothesis. 
- Determine what model to create
  
## Acquisition:
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

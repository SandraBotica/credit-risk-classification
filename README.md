## Module-20-Challenge

## Description

### Identify the creditworthiness of borrowers (Are borrowers a loan risk?) by training a Logistic Regression Model (supervised learning) on a dataset of historical lending activity from a peer-to-peer lending services company.

### Technologies used

 - Python notebook

## Getting Started

Open the following file <credit_risk_classification.ipynb>
1. The <lending_data.csv> was loaded in as a Pandas DataFrame called df_lending_data.

2. This data was split into training and testing data after:
 - label set y based on 'loan status' column. 
 - `0` healthy label, `1` high risk of defaulting.
 - features X DataFrame based on all other columns.

3. A Logistic Regression Model was run on the training data and predictions made using the test data. The models performance at identifying borrowers at risk of defaulting was evaluated based on:
 - Calculated Accuracy Score
 - Confusion Matrix generated
 - Classification Report (Accuracy, Precision, Recall)


4. Questions were answered in the notebook in markdown cell blocks.

5. The credit analysis report can be found in the file <report-template.md>

6. Training data was then resampled using the RandomOverSampler module from the imblanaced-learn library. The resampled data was fit to a Logistic Rgression Model and predictions made. Steps 3/4/5 above were repeated.  

7. Additionally I thought it would be interesting to run the Logistic Regression model on Scaled Data, so this was done at the end of the notebook, more for my own investigation of the difference with or without scaling of the original dataset.


### Enjoy marking!
### Sandra
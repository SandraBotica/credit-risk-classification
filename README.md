## Module 20 Report.

## Overview.

## The purpose of this analysis was to use various techniques to train and evaluate machine learning models based on loan risk.

The financial information of borrowers provided in the <lending_data.csv> included (8 columns and 77536 rows, excluding the Header):
 - loan-size
 - interest-rate
 - borrower-income
 - debt-to-income
 - num-of-accounts
 - derogatory-marks
 - total-debt
 - loan-status (`0` label = healthy, `1` label = high risk of defaulting)
 
### The aim is to predict which borrowers with loans were at high risk of defaulting.

 - The "loan-status" column, described above, will become the dependent, y target variable, the labels.
 - The variables that are to be predicted are `0` and `1`. 
 - All other columns will be the X variable, the features.
 - The aim is to classify using a binary classifier (logic regression and randomsampler) borrowers membership to either category i.e. `0` healthy loan status i.e. low credit risk,`1` high risk of defaulting i.e. a high credit risk. 
 - One could suggest that these categories could also be linked to borrowers being further labelled: `0` - creditworthy if applying for loans, `1` - high credit-risk and unqualified for loan application.

 - After data was separated y.value_counts() were:
 * `0` - 75036
 * `1` - 2500
 - Considering the label `1` is what we need to identify, it comprises just 31% of total data (rows/borrowers). This demonstrates a Class imbalance, with the dataset biased to `0`, the healthy loan label. It is highly probable that the features of this CLASS will drive the prediction model. 


### The stages of the machine learning process and methods used, included:

 1. Separating data (described above).
 2. Splitting data using "train_test_split", with the training features X_Train.shape() = 58152 (`0`),7 (`1`).
 3. Creating an instance of a `LogisticRegression` model.
 4. Fitting the model using using training data.
 5. Making predictions using the testing data, saving these values with the actual value in a DataFrame.
 6. Scoring the training and testing data.
 7. Evaluating the models performance (described in results). 

 - These stages were repeated using the RandomOverSampler module from the imbalanced-learn library to resample data with equal number of data points for each class.

 - You will also see at the end of the notebook my exploration of scaling data for the first machine learning model, as the features "loan-size", "borrower-income" and "total-debt" were much larger values.

### I would also consider removing the "debt-to-income" column as it is the same as columns "total-debt"/"borrower-income". 
### As having these features repeated, will impact on prediction capabilities of this model.

## Results

### Machine Learning Model 1:

The logistic regression model was better at predicting the `0` label then the `1` labels. 
This may be due to the bias of data having 18765 "0" and only 619 "1". 

Accuracy
 - A balanced_accuracy_score of 95% makes one think that this is a good prediction model, as does the 99% accuracy score in the classification report. 
  
Confusion matrix:
 - Total predictions = 19384
 - Correct predictions = 19226 (18663 `0`s TN, 563 `1`s TP)
 - Incorrect predictions = 158 (102 `0`s FP, 56 `1`s at FN)

Classification report.
 - Precision: 15% of the time (0.85 precision score) the model predicted a false positive (predicted a `1` but was actually a `0`), hence 102 borrowers (from the confusion matrix) were identified as at risk of defaulting when they were healthy borrowers, predicting a FALSE ALARM (Type 1 error).
 - Recall: 9% of the time (0.91 recall score) the model predicted a false negative (predicted a `0` but was actually a `1`), hence 56 borrowers (from the confusion matrix) who are at risk of defaulting have been MISSED in the prediction (Type 2 error).

### Machine Learning Model 2:

The logistic regression model, fit with oversampled data, continued to be better at predicting the `0` label then the `1` labels, but predictions improved in accuracy, and recall, when using oversampled data, with a slight decline in precision scores.

Accuracy
 - The balanced_accuracy_score improved from 95% to 99% and the accuracy score in the classification report remained at 99%. 
  
Confusion matrix:
 - Total predictions = 19384
 - Correct predictions = 19261 (18652 `0`s TN, 609 `1`s TP)
 - Incorrect predictions = 123 (113 `0`s FP, 10 `1`s at FN)
  
Classification report.
 - Precision: 16% of the time (0.84 precision score) the model predicted a false positive (predicted a `1` but was actually a `0`), hence 113 borrowers (from the confusion matrix) were identified as at risk of defaulting when they were healthy borrowers, predicting a FALSE ALARM (Type 1 error).
 - Recall: 2% of the time (0.98 recall score) the model predicted a false negative (predicted a `0` but was actually a `1`), hence 10 borrowers (from the confusion matrix) who are at risk of defaulting have been MISSED in the prediction (Type 2 error).

## Summary

The resampled data fit to the logistic regression model (Machine Learning Model 2) performed better i.e. made better predictions.

Model 2 had a better balanced_accuracy_score, improving from 95% to 99% accuracy. Keep in mind that accuracy scores are not so valid in these predictions as the target variables are mostly class `0` i.e. the data was biased towards this class.

Model 2 had an improved recall (reduced number of false negatives). Predictions MISSED (Type 2 error) were 10 compared to 56 borrowers as actually being at risk of defaulting. 

Model 2 had a 1% reduction in precision (increased number of false positives). Predictions of FALSE ALARMS (Type 1 error) were 113 compared to 102 borrowers as actually not being at risk of defaulting.


The purpose of this analysis was to train and evaluate machine learning models based on loan risk. Both models were better at predicting `0` healthy borrowers. The concern with both these models is that they MISSED borrowers at risk of defaulting and predicted FALSE ALARMS, where a `0` healthy borrower may have been contacted as at risk of defaulting when they were not.

When you consider that borrowers labeled `1`, at risk of defaulting may face these risks:
 - reduced credit score with future access to credit being denied
 - potential home foreclosure
 - the need to file for bankruptcy
 - legal claims
* Finding a model that better predicted the at risk of defaulting `1` borrowers is critical in preventing the above, or at least providing the creditors the opportunity to get bank assistants to contact and provide advice to prevent this from occuring.


These models performed near perfect predictions for the `0`'s. If trying to generate a list of borrowers to offer additional credit, loans, insurance (income/life/death etc) or financial advice packages, then one would say "the model has been correctly trained to produce comprehensive, accurate and precise predictions" in real life.

### Based on the results the logistic regression model on resampled data seems to be a good algorithm to use in a bank, but may require further pilots with new data to assess it's reliability.

### What is required is a metric that better evaluates the `1` at risk of defaulting class prediction.

### I think it would be interesting to explore the use of a support vector machine learning model, and alternatively a decision tree, as it would help trace back the models logic to see how it reached a prediction, so as to better justify alerting borrowers that they are at risk of defaulting.



### Enjoy marking!
### Sandra
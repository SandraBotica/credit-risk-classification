# Module 20 Report.

## Overview.

* The purpose of this analysis was to use various techniques to train and evaluate machine learning models based on loan risk.

* The financial information of borrowers provided in the <lending_data.csv> included (8 columns and 77536 rows, excluding the Header):
 - loan-size
 - interest-rate
 - borrower-income
 - debt-to-income
 - num-of-accounts
 - derogatory-marks
 - total-debt
 - loan-status (`0` label = healthy, `1` label = high risk of defaulting)
 
* The aim is to predict which borrowers with loans were at high risk of defaulting.
 - The "loan-status" column, described above, will become the dependent y variable, the labels.
 - The variables that are to be predicted are `0` and `1`. 
 - All other columns will be the X variable, the features.
 - The aim is to draw categorical conclusions about data using a binary approach of predicting borrowers membership to either category i.e. "0" healthy loan status,"1" high risk of defaulting. 
 - One could suggest that these categories could also be linked to borrowers being further labelled: "0" - creditworthy if applying for loans, '"1" - credit-risk and unqualified for loan application.

 - After data was separated y.value_counts() were:
 * `0` - 75036
 * `1` - 2500
 - Considering the label `1` is what we need to identify, it comprises just 31% of total data (rows/borrowers). This demonstrates a Class imbalance, with the dataset biased to `0`, the healthy loan label. It is highly probable that the features of this CLASS will drive the prediction model. 


* The stages of the machine learning process and methods used, included:
 1. Separating data (described above).
 2. Splitting data using "train_test_split", with the training features X_Train.shape() = 58152 (`0`),7 (`1`).
 3. Creating an instance of a `LogisticRegression` model.
 4. Fitting the model using using training data.
 5. Making predictions using the testing data, saving these values with the actual value in a DataFrame.
 6. Scoring the training and testing data.
 7. Evaluating the models performance (described in results). 

 - These stages were repeated using the RandomOverSampler module from the imbalanced-learn library to resample data with equal number of data points for each label/CLASS.

 - You will also see at the end of the notebook my exploration of scaling data for the first machine learning model, as the features "loan-size", "borrower-income" and "total-debt" were much larger values.

 - I am also concerned that the "debt-to-income" column is a repeat of columns "borrower-income" + "total-debt". Same features being repeated, which may impact on prediction capabilities of this model.

## Results

* Machine Learning Model 1:

  - The logistic regression model was better at predicting the `0` label then the `1` labels. 
  - This may be due to the bias of data having 18765 "0" and only 619 "1". 

  - Accuracy
  - A balanced_accuracy_score of 95% makes one think that this is a good prediction model, as does the 99% accuracy score in the classification report. This is mis-leading, as one should be concerned that if this model was trying to predict who is at risk of defaulting the followng occurs.
  - Precision
  - 15% of the time (based on 0.85 precision score) the model predicted a false negative (predicted a `0` but was actually a `1`), hence 56 borrowers (from the confusion matrix) who are at risk of defaulting have been MISSED in being identified.
  - Recall
  - 9% of the time (based on 0.91 recall score) the model predicted a false positive (predicted a `0` but was actually a `1`), hence 102 borrowers (from the confusion matrix) who are not at risk of defaulting may have been identified as at risk of defaulting creating a FALSE ALARM.

* Machine Learning Model 2:

  - The logistic regression model, fit with oversampled data, continued to be better at predicting the `0` label then the `1` labels, but predictions improved in accuracy, and recall, when using oversampled data, with a slight decline in precision scores.

  - Accuracy
  - The balanced_accuracy_score improved from 95% to 99% and the accuracy score in the classification report remained at 99%. One should still be concerned that if this model was trying to predict who is at risk of defaulting the following continues to occur.
  - Precision
  - 16% of the time (based on 0.84 precision score) the model predicted a false negative (predicted a `0` but was actually a `1`), hence 10 borrowers (from the confusion matrix) who are at risk of defaulting have been MISSED in being identified.
  - Recall
  - 2% of the time (based on 0.98 recall score) the model predicted a false positive (predicted a `0` but was actually a `1`), hence 113 borrowers (from the confusion matrix) who are not at risk of defaulting may have been identified as at risk of defaulting creating a FALSE ALARM.

## Summary

Summarise the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?


* Performance of this model in predicting the `1`'s was near perfect and if trying to solve the folowing list of problems then one would say "the model has been correctly trained to produce comprehensive, accurate and precise predictions".
 - Extra credit
 - Extra loans
 - Insurance packages (income/life/death etc)
 - Finance advice packages

* I wouldn't recommend either model as the number of false negatives  would mean that some people may discover too late that their 
If you do not recommend any of the models, please justify your reasoning.

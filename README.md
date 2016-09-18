# Identify Fraud from Enron Email
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

Play detective and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

##1. Summary and Overview of dataset
The goal is to use machine learning techniques to build a predictive model that can accurately identify persons of interest from a pool of Enron employees. Available features include information extracted from
employees' email and financial records.

Number of data points: 146
Number of persons of interest: 18
Number of features: 21

The data is unbalanced as the poi to other employees is at a ratio of 1:7.11. It would be very important to not only look at accuracy when evaluating machine learning algorithms but also consider recall and precision as very good indicators.

A spreadsheet quirk outlier 'TOTAL' is identified and removed. Other outliers with extreme values are kept because they might be valuable in detecting person of interest.

##2. Feature selection
I engineered 3 new features, fraction_to_poi, fraction_from_poi and other_compensation.
'fraction_to_poi' is the percentage of emails sent to poi.
'fraction_from_poi' is the percentage of emails received from poi.
'other_compensation' is all incomes minus base salary and bonus.

Then I conducted an univariate feature selection process to identify top k most important features.

Let's see how performance of GaussianNB classifier accuracy changes when different number of features is used.
They are:

**Number of features* | **Accuracy** | **Recall** | **Precision**
-------- | ----------- | ---------
3 | 0.84723 | 0.28350 | 0.50625
4 | 0.86093 | 0.3490 | 0.51973
5 | 0.84929 | 0.3935 | 0.46734
6 | 0.85614 | 0.40150 | 0.49568
7 | 0.86120 | 0.38300 | 0.47460
8 | 0.85793 | 0.36100 | 0.45841

Precision: Percentage of predicted POIs are true POIs.
Recall: Percentage of true POIs are predicted.

In the case of find POI, I think recall is more important than precision because I want to make sure as many POIs are brought to justice as possible. I decided to go with the top 6 variables.
They are: 'fraction_to_poi', 'long_term_incentive', 'deferred_income', 'bonus', 'total_stock_value', 'exercised_stock_options'

##3. Pick Machine learning algorithms and tuning
Supervised learning algorithms will be suitable to my need because I have a clear Y variable / target variable. I have decided to try out GaussianNB and random forest.

I did not scale my features as scaling is not needed. Because changing scaling of features or distance between one point to the other will not affect the results of GaussianNB or Random forest.

Baseline of GaussianNB
Recall: .3265
Precision: 0.49545
Accuracy: 0.85629

Baseline of Random Forest
Accuracy: 0.84921
Precision: 0.43974
Recall: 0.20250

Parameters tuning is critical in developing accurate machine learning model. No matter how good the model is right out the box, there is also room for improvement.

**Algorithm** | **Parameter** | **Accuracy** | **Precision** | **Recall**
------------- | ------------- | ------------ | ------------- | ----------
RF | n_estimators=10 | 0.84921 | 0.43974 | 0.20250
RF | n_estimators=50 | 0.85571 | 0.48973 | 0.23850
RF | n_estimators=100 | 0.85686 | 0.49792 | 0.23900
GaussianNB | NA | 0.85629 | 0.49545 | 0.85629

Overall GaussianNB performed the best and I've decided to go with GauusianNB.

##4. Evaluation
It is important to not only look at accuracy when evaluating different machine learning algorithms but also precision and recall. Precision and recall provide you with a more holistic view of the selected model. Also it is crucial to see if the model returns more relevant results than irrelevant results and whether the model returns most relevant results.
Cross-validation is used with the Stratifiedshifflesplit method. Validation is to split the dataset into training and testing set to make sure that the model performs well with data other than the training set. It is important to see how the model will perform when put into use.

The chosen model has a accuracy of .85614, precision of .49568 and recall of 0.40150.

Precision: Percentage of predicted POIs are true POIs.
Recall: Percentage of true POIs are predicted.

Total predictions: 14000	True positives:  803	False positives:  817	False negatives: 1197	True negatives: 11183

True positives: POIs who are accurately predicted as POIs.
False positives: Innocent employees incorrectly labeled as POI.
False negatives: Innocents employees labeled as Innocent
True negatives: POIs who are incorrectly labeled as innocent.

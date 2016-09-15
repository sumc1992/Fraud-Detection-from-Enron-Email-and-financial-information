# Project 5
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

Play detective and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

##1. Summary and Overview of dataset
The goal is to use machine learning techniques to build a predictive model that can accurately identify persons of interest from a pool of Enron employees. Available features include information extracted from
employees' email and financial records.

Number of data points: 146
Number of persons of interest: 18
Number of features: 21

A spreadsheet quirk outlier 'TOTAL' is identified and removed. Other outliers with extreme values are kept because they might be valuable in detecting person of interest.

##2. Feature selection
I engineered 3 new features, fraction_to_poi, fraction_from_poi and other_compensation.
'fraction_to_poi' is the percentage of emails sent to poi.
'fraction_from_poi' is the percentage of emails received from poi.
'other_compensation' is all incomes minus base salary and bonus.

Then I conducted an univariate feature selection process to identify top five most important features.

They are:
**Rank** | **Feature** | **Score**
-------- | ----------- | ---------
1 | exercised_stock_options | 25.098
2 | total_stock_value | 24.468
3 | bonus | 21.060
4 | salary | 18.576
5 | fraction_to_poi | 16.642

These five important features will then be used to create my predictive models.

##3. Pick Machine learning algorithms and tuning
Supervised learning algorithms will be suitable to my need because I have a clear Y variable / target variable. I have decided to try out GaussianNB and random forest.

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

The chosen model has a accuracy of .85629, precision of .49545 and recall of .85629.

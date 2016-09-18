#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from compute_fraction import computeFraction

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages',
 'from_poi_to_this_person', 'from_messages',
'from_this_person_to_poi', 'shared_receipt_with_poi', 'fraction_from_poi',
'fraction_to_poi', 'other_compensation'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

num_poi = 0
for value in data_dict.values():
    if value['poi'] == True:
        num_poi += 1

print 'Total number of data points: ', len(data_dict)
print 'Number of POI: ', num_poi
print 'Total number of features: ', len(data_dict[data_dict.keys()[0]])
### Task 2: Remove outliers
data_dict.pop('TOTAL')


### Task 3: Create new feature(s)

### Engineer two new features, fraction_to_poi and fraction_from_poi
### Perccentage of emails sent to poi or from poi

### Engineer the third new variable, other_compensation.
### All incomes minus base salary and bonus.

list_of_bonus = ['deferral_payments', 'total_payments', 'loan_advances',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
'restricted_stock', 'director_fees']

for name in data_dict:

    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_dict[name]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_dict[name]["fraction_to_poi"] = fraction_to_poi

    data_dict[name]['other_compensation'] = 0

    for feature in list_of_bonus:
        if data_point[feature] != 'NaN':
            data_dict[name]['other_compensation'] += data_point[feature]

#print data_dict[data_dict.keys()[0]]

### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Conduct k-best univariate feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import f_classif
# Input an sklearn.cross_validation object here, e.g. StratifiedShuffleSplit
splits = StratifiedShuffleSplit(labels, 3, test_size = .3, random_state = 0)
k = 6 # Change this to number of features to use.

# We will include all the features into variable best_features, then group by their
# occurrences.
best_features = []
for i_train, i_test in splits:
    features_train, features_test = [features[i] for i in i_train], [features[i] for i in i_test]
    labels_train, labels_test = [labels[i] for i in i_train], [labels[i] for i in i_test]

    # fit selector to training set
    selector = SelectKBest(f_classif, k = k)
    selector.fit(features_train, labels_train)

    for i in selector.get_support(indices = True):
        best_features.append(features_list[i+1])

# This is the part where we group by occurrences.
# At the end of this step, features_list should have k of the most occurring
# features. In other words they are features that are highly likely to have
# high scores from SelectKBest.
from collections import defaultdict
d = defaultdict(int)
for i in best_features:
    d[i] += 1
import operator
sorted_d = sorted(d.items(), key=operator.itemgetter(1))
features_list = [x[0] for x in sorted_d[-k:]]

### Choose top five most important variables
features_list = ['poi'] + features_list
print features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()

from sklearn.ensemble import RandomForestClassifier
#clf_rf = RandomForestClassifier(n_estimators=10)
#clf_rf = RandomForestClassifier(n_estimators=50)
#clf_rf = RandomForestClassifier(n_estimators=100)
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Fit classifiers
clf_gnb.fit(features_train, labels_train)
#clf_rf.fit(features_train, labels_train)

from sklearn.neighbors import KNeighborsClassifier
clf = clf_gnb
#print clf.score(features_test, labels_test)

#print 'Precision score is: ', precision_score(labels_test, clf.predict(features_test))
#print 'Recall score is: ', recall_score(labels_test, clf.predict(features_test))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

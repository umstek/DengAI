# -*- coding: utf-8 -*-

import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

colna = list(range(0, 24))
colna.remove(1)
colna.remove(3)

features = pandas.read_csv(filepath_or_buffer='dengue_features_train.csv', usecols=colna)
features = pandas.get_dummies(features)
features = features.fillna(value=features.mean())
feature_list = list(features.columns)
features = numpy.array(features)

labels = pandas.read_csv('dengue_labels_train.csv')
labels = numpy.array(labels['total_cases'])

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 100)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 100)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(numpy.mean(errors), 2), 'units')
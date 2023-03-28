# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd

# K-fold setting
# Get all the K-fold data
data = pd.read_csv('train_data.csv')
x_data = data.drop('label', axis=1).values
x_label = data.drop('label', axis=1)
# Get Columns' name
x_columns = [column for column in x_label]
y_data = data.label.values

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
i = 0
for train_index, valid_index in cv.split(x_data,y_data):
        i = i + 1
        x_train, x_valid = x_data[train_index], x_data[valid_index]
        y_train, y_valid = y_data[train_index], y_data[valid_index]
        # train_set = pd.DataFrame({'SMILES': x_train, 'label': y_train})
        train_label = pd.DataFrame({'label': y_train})
        train_feat = pd.DataFrame(x_train, columns=x_columns)
        train_set = pd.concat([train_label, train_feat], axis=1)
        train_set.to_csv('training_data' + '_' + str(i) + '.csv', index=False, sep=',')
        # valid_set = pd.DataFrame({'SMILES': x_valid, 'label': y_valid})
        valid_label = pd.DataFrame({'label': y_valid})
        valid_feat = pd.DataFrame(x_valid, columns=x_columns)
        valid_set = pd.concat([valid_label, valid_feat], axis=1)
        valid_set.to_csv('validation_data' + '_' + str(i) + '.csv', index=False, sep=',')
import pandas as pd
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

# Helper functions ------------------------------------------------

def fill_missing_values(df, column_name):
    most_common_value = df[column_name].value_counts().idxmax()
    df[column_name] = df[column_name].fillna(most_common_value)

def fill_missing_values_median(df, column_name):
    median_value = df[column_name].median()
    df[column_name] = df[column_name].fillna(median_value)

# Load data -------------------------------------------------------
df_train = pd.read_csv('data/train.csv', index_col='Id')
df_test = pd.read_csv('data/test.csv', index_col='Id')

# Dealing with outliers -------------------------------------------

# Remove GrLivArea outliers
df_train = df_train[df_train['GrLivArea'] < 4500]
# Remove GarageYrBlt outliers
df_train = df_train[df_train['GarageYrBlt'] < 2023]

# df_test['GarageYrBlt'].hist(bins=50)
# plt.show()
# 
# print(df_train.shape)

# Dealing with skewed data -----------------------------------------

# SalePrice is skewed so transform it by the natural log function
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

# Preparing features

# Extract features
train_features = df_train.drop(['SalePrice'], axis=1)
test_features = df_test

# Concatenate train with test features into one data frame
features = pd.concat([train_features, test_features]).reset_index(drop=True)

# Strings disguised as numbers --------------------------------------

# Change these int columns to strings
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

# Dealing with nan values in categorical data -----------------------

# set all nan values to a category 'None'
objects = []
for i in features.columns:
    if features[i].dtype == 'object':
        objects.append(i)
features.update(features[objects].fillna('None'))

# Dealing with nan values in numerical data -----------------------

# Get column names of columns with numerical data
numerical_columns = features.dtypes[features.dtypes != 'object'].keys()

# Fill missing values with most common value
fill_missing_values(features, 'MasVnrArea')
fill_missing_values(features, 'BsmtFinSF1')
fill_missing_values(features, 'BsmtFinSF2')
fill_missing_values(features, 'BsmtUnfSF')
fill_missing_values(features, 'TotalBsmtSF')
fill_missing_values(features, 'BsmtFullBath')
fill_missing_values(features, 'BsmtHalfBath')
fill_missing_values(features, 'GarageCars')
fill_missing_values(features, 'GarageArea')

# Fill missing values with median value
fill_missing_values_median(features, 'GarageYrBlt')

# get rid of nans from lot frantage (400+ nans)
# do this by grouping the lot frontages according to neighbourhood and extracing the median
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

print(features[numerical_columns].isnull().sum())

# Show histograms of all numerical data
# features[numerical_columns].hist(bins=40)
# plt.show()

# Getting dummies ------------------------------------------------
final_features = pd.get_dummies(features).reset_index(drop=True)

# Divide the features back into train and test sets -------------
X = final_features.iloc[:len(df_train), :]
Y = final_features.iloc[len(df_train):, :]

print(X.shape, Y.shape)

X['SalePrice'] = df_train['SalePrice'].to_list()

X.to_csv("data/clean_train.csv")
Y.to_csv("data/clean_test.csv")

print(X.head())

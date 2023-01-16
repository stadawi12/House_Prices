import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper functions ------------------------------------------------

def fill_missing_values(df, column_name):
    " Fills missing values with the most common value "
    most_common_value = df[column_name].value_counts().idxmax()
    df[column_name] = df[column_name].fillna(most_common_value)

def fill_missing_values_median(df, column_name):
    " Fills missing values with the median value "
    median_value = df[column_name].median()
    df[column_name] = df[column_name].fillna(median_value)

# Load data -------------------------------------------------------
df_train = pd.read_csv('data/train.csv', index_col='Id')
df_test = pd.read_csv('data/test.csv', index_col='Id')

# Dealing with outliers -------------------------------------------

# Remove GrLivArea outliers
df_train = df_train[df_train['GrLivArea'] < 4000]
# Remove GarageYrBlt outliers
# df_train = df_train[df_train['GarageYrBlt'] < 2023]

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

# Get column names of columns with numerical data
numerical_columns = features.dtypes[features.dtypes != 'object'].keys()

# Drop columns with a lot of missing data ---------------------------

# df_null_counts = features.isnull().sum()
# print(df_null_counts[df_null_counts > 500])

# counts = features['Street'].value_counts(dropna=False)
# print(counts)

features.drop(['Utilities', 'Street','PoolQC'],
        axis=1, inplace=True)

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

# Fill missing values with most common value
# fill_missing_values(features, 'MasVnrArea')
# fill_missing_values(features, 'BsmtFinSF1')
# fill_missing_values(features, 'BsmtFinSF2')
# fill_missing_values(features, 'BsmtUnfSF')
# fill_missing_values(features, 'TotalBsmtSF')
# fill_missing_values(features, 'BsmtFullBath')
# fill_missing_values(features, 'BsmtHalfBath')
# fill_missing_values(features, 'GarageCars')
# fill_missing_values(features, 'GarageArea')

# Fill missing values with median value
fill_missing_values_median(features, 'GarageYrBlt')

# transform nans in lot frantage (400+ nans)
# use the median lot of the neighbourhood to fill the missing value
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
features['MasVnrArea'] = features.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))
features['BsmtFinSF1'] = features.groupby('Neighborhood')['BsmtFinSF1'].transform(lambda x: x.fillna(x.median()))
features['BsmtFinSF2'] = features.groupby('Neighborhood')['BsmtFinSF2'].transform(lambda x: x.fillna(x.median()))
features['BsmtUnfSF'] = features.groupby('Neighborhood')['BsmtUnfSF'].transform(lambda x: x.fillna(x.median()))
features['TotalBsmtSF'] = features.groupby('Neighborhood')['TotalBsmtSF'].transform(lambda x: x.fillna(x.median()))
features['BsmtFullBath'] = features.groupby('Neighborhood')['BsmtFullBath'].transform(lambda x: x.fillna(x.median()))
features['BsmtHalfBath'] = features.groupby('Neighborhood')['BsmtHalfBath'].transform(lambda x: x.fillna(x.median()))
features['GarageCars'] = features.groupby('Neighborhood')['GarageCars'].transform(lambda x: x.fillna(x.median()))
features['GarageArea'] = features.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.median()))

# print(features[numerical_columns].isnull().sum())

# SalePrice is skewed so transform it by the natural log function
# features['LotFrontage'] = np.log1p(features['LotFrontage'])
# features['1stFlrSF'] = np.log1p(features['1stFlrSF'])
# features['GrLivArea'] = np.log1p(features['GrLivArea'])

# Show histograms of all numerical data
# features[numerical_columns].hist(bins=40)
# plt.show()

# Feature engineering ------------------------------------------
# Adding new features . Make sure that you understand this. 

features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']
features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +
                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                              features['WoodDeckSF'])

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Getting dummies ------------------------------------------------
final_features = pd.get_dummies(features).reset_index(drop=True)

# Divide the features back into train and test sets -------------
X = final_features.iloc[:len(df_train), :]
Y = final_features.iloc[len(df_train):, :]

# print(X.shape, Y.shape)

X['SalePrice'] = df_train['SalePrice'].to_list()

X.to_csv("data/clean_train.csv")
Y.to_csv("data/clean_test.csv")

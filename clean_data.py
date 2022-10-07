import pandas as pd
from typing import Any
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load data
    df_train: Any = pd.read_csv('data/train.csv', index_col='Id')
    df_test: Any = pd.read_csv('data/test.csv', index_col='Id')
    df_test_labels: Any = pd.read_csv('data/sample_submission.csv', index_col='Id')

    # Join test data with labels
    df_test = df_test.join(df_test_labels, on='Id')

    # CLEANING TRAIN -------------------------------------------------------------------

    # Turn categorical data into one hot encodings
    df_train_dummies: Any = pd.get_dummies(data=df_train, dummy_na=True, dtype='float')

    # Drop LotFrontage column because 290 out of 1460 rows have missing values
    df_train_dummies = df_train_dummies.drop(columns = ['LotFrontage'])

    # Drop rows with missing values (89 rows dropped)
    df_train_dummies = df_train_dummies.dropna()

    print(f"Raw train dataset shape: {df_train.shape}")
    print(f"Cleaned train dataset shape: {df_train_dummies.shape}\n")

    # CLEANING TEST ----------------------------------------------------------------------

    df_test_dummies: Any = pd.get_dummies(data=df_test, dummy_na=True, dtype='float')

    # Drop LotFrontage because more than 200 out of 1459 rows have missing values
    df_test_dummies = df_test_dummies.drop(columns = ['LotFrontage'])

    # Drop rows with missing values (89 rows dropped)
    df_test_dummies = df_test_dummies.dropna()

    print(f"Raw test dataset shape: {df_test.shape}")
    print(f"Cleaned test dataset shape: {df_test_dummies.shape}\n")

    nan_columns_test: Any = df_test_dummies.isna().any()[lambda x: x]

    # Concatenate train and test data into one table
    df = pd.concat([df_train_dummies, df_test_dummies], axis=0,
                   keys=['train', 'test'])
    print(f"Combined data shape: {df.shape}")

    df = df.fillna(0)

    print(df['SalePrice'].mean())

    # Normalise columns that have a larger max value than 1
    for column in df.columns:
        maximum = df[column].max()
        if maximum > 1:
            df[column] = df[column].div(maximum)
            print(column, df[column].max())

    df.to_csv('data/clean_data.csv')

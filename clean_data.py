import pandas as pd
from typing import Any
import matplotlib.pyplot as plt
from utils import Read_Input

input_data = Read_Input('inputs.yaml')

drop_lot_frontage = input_data['drop_lot_frontage']
train_dropna      = input_data['train_dropna']
test_dropna       = input_data['test_dropna']

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
    if drop_lot_frontage:
        df_train_dummies = df_train_dummies.drop(columns = ['LotFrontage'])

    # Drop rows with missing values (89 rows dropped)
    if train_dropna:
        df_train_dummies = df_train_dummies.dropna()

    df_train_dummies[["SalePrice","LotFrontage"]].plot(
            kind='scatter', 
            x = ['SalePrice'],
            y = ['LotFrontage','SalePrice'],
            sharex=False,
            sharey=False)
    plt.show()

    # vars = []
    # for column in df_train_dummies.columns:
    #     var = df_train_dummies[column].var()
    #     if var == 0:
    #         print(column)
        
        # vars.append(var)
    # print(sorted(vars))

    print(f"Raw train dataset shape: {df_train.shape}")
    print(f"Cleaned train dataset shape: {df_train_dummies.shape}\n")

    # CLEANING TEST ----------------------------------------------------------------------

    df_test_dummies: Any = pd.get_dummies(data=df_test, dummy_na=True, dtype='float')

    # Drop LotFrontage because more than 200 out of 1459 rows have missing values
    if drop_lot_frontage:
        df_test_dummies = df_test_dummies.drop(columns = ['LotFrontage'])

    # Drop rows with missing values (94 rows dropped)
    if test_dropna:
        df_test_dummies = df_test_dummies.dropna()

    print(f"Raw test dataset shape: {df_test.shape}")
    print(f"Cleaned test dataset shape: {df_test_dummies.shape}\n")

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
            # print(column, df[column].max())

    df.to_csv('data/clean_data.csv')

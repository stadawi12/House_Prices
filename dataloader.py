import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class HousePriceDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        df_X = self.dataframe.drop(columns=["SalePrice"])
        df_Y = self.dataframe["SalePrice"]

        X = torch.from_numpy(df_X.iloc[idx].to_numpy().astype(np.float32))
        Y = torch.tensor([df_Y.iloc[idx]], dtype=torch.float)

        return X, Y

if __name__ == '__main__':

    df = pd.read_csv('data/clean_train.csv', index_col=[0])

    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)

    training_data = HousePriceDataset(train)
    testing_data = HousePriceDataset(test)

    dataloader_train = DataLoader(training_data, batch_size=64, num_workers=4)
    dataloader_test = DataLoader(testing_data, batch_size=64, num_workers=4)

    X, y = next(iter(dataloader_train))

    print(X[:5])
    print(train.head())
    print(torch.exp(y[:5]))

    print(X.shape)
    print(y.shape)

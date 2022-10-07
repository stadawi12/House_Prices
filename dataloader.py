import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class HousePriceDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, dataset: str):
        self.dataframe = dataframe
        self.dataset = dataset
        
        self.df = self.dataframe.loc[self.dataset]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_X = self.df.drop(columns=["SalePrice"])
        df_Y = self.df["SalePrice"]

        X = torch.from_numpy(df_X.iloc[idx].to_numpy().astype(np.float32))
        Y = torch.tensor([df_Y.iloc[idx]], dtype=torch.float)

        return X, Y

if __name__ == '__main__':

    df = pd.read_csv('data/clean_data.csv', index_col=[0,1])

    training_data = HousePriceDataset(df, 'train')

    dataloader_train = DataLoader(training_data, batch_size=64, num_workers=4)

    X, y = next(iter(dataloader_train))

    print(X.dtype)
    print(y.dtype)

    print(X.shape)
    print(y.shape)

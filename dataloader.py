import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

class HousePriceDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe : pd.DataFrame = dataframe
        
    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        df_features : pd.DataFrame = self.dataframe.drop(columns=["SalePrice"])
        df_labels   : pd.DataFrame = self.dataframe["SalePrice"]

        features : Tensor = torch.from_numpy(df_features.iloc[idx].to_numpy().astype(np.float32))
        labels   : Tensor = torch.tensor([df_labels.iloc[idx]], dtype=torch.float)

        return features, labels

if __name__ == '__main__':
    from torch.utils.data import DataLoader

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

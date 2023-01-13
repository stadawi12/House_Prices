import torch
import torch.nn as nn

class LinearModel(nn.Module):
    
    def __init__(self, input_size):
        super(LinearModel, self).__init__()

        self.model_size = 40

        self.linear1 = nn.Linear(input_size, self.model_size)
        self.linear2 = nn.Linear(self.model_size, self.model_size)
        self.activation = nn.ReLU()
        self.out = nn.Linear(self.model_size, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):

        x = self.linear1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.activation(x)

        x = self.out(x)
        return x

if __name__ == '__main__':

    import pandas as pd
    from dataloader import HousePriceDataset
    from torch.utils.data import DataLoader

    df = pd.read_csv('data/clean_train.csv')
    train = df.sample(frac=0.8, random_state=42)

    train_set = HousePriceDataset(train)
    train_dl = DataLoader(train_set, batch_size=10, num_workers=4)

    X, y = next(iter(train_dl))

    print(X)

    print(X.shape)
    
    model = LinearModel(X.shape[1])

    out = model(X)
    print(out)

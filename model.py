import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam

from torch.utils.data import DataLoader

HIDDEN_LAYERS = 2
HIDDEN_SIZE = 300

def Layers(input_size: int, output_size: int, times: int = 1, is_out: bool = True):
    components = []
    for _ in range(times):
        layers.append(nn.Linear(input_size, output_size))
        if is_out:
            layers.append(nn.ReLU())
    layers = nn.Sequential(*components)
    return layers

class LinearModel(pl.LightningModule):
    
    def __init__(self, input_size: int):
        super(LinearModel, self).__init__()

        self.linear_in      = Layers(input_size, HIDDEN_SIZE)
        self.linear_hidden  = Layers(HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS)
        self.linear_out     = Layers(HIDDEN_SIZE, 1, is_out=False)
        self.activation_out = nn.Sigmoid()
        self.criterion      = nn.MSELoss() 

    def forward(self, x):
        x = self.linear_in(x)
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return x

    def training_step(self, batch, batch_idx):
        features, labels = batch
        y_hat = self(features)
        loss = self.criterion(y_hat, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.01)

    def train_dataloader(self):
        dataset = HousePriceDataset()


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

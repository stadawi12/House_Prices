import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar

from dataloader import HousePriceDataset

# Hyper-parameters
HIDDEN_SIZE     : int = 300
N_HIDDEN_LAYERS : int = 2
DATA_SPLIT      : float = 0.8
LEARNING_RATE   : float = 0.01
BATCH_SIZE      : int = 32
N_EPOCHS        : int = 100

# Train and test data
All_data : pd.DataFrame = pd.read_csv('data/clean_train.csv', index_col=[0])
Train    : pd.DataFrame = All_data.sample(frac=DATA_SPLIT, random_state=42)
Valid    : pd.DataFrame = All_data.drop(Train.index)

steps_per_epoch = len(Train) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS

warmup_steps = total_training_steps // 5

def Layer(input_size: int, output_size: int, times: int = 1, is_out: bool = False):
    """ Generate neural network layer with the specified input and output size and do not include
    activation function if it is an output layer """
    components = []
    for _ in range(times):
        components.append(nn.Linear(input_size, output_size))
        if not is_out:
            components.append(nn.ReLU())
    layers = nn.Sequential(*components)
    return layers

class LinearModel(pl.LightningModule):
    """ Pytorch lighning model """
    def __init__(self, input_size: int, hidden_size: int, n_hidden_layers: int):
        super(LinearModel, self).__init__()
        self.save_hyperparameters()
        self.input_size      : int = input_size
        self.hidden_size     : int = hidden_size
        self.n_hidden_layers : int = n_hidden_layers

        self.linear_in      = Layer(self.input_size, self.hidden_size)
        self.linear_hidden  = Layer(self.hidden_size, self.hidden_size, times=self.n_hidden_layers)
        self.linear_out     = Layer(self.hidden_size, 1, is_out=True)
        self.criterion      = nn.MSELoss()

    def forward(self, x):
        x = self.linear_in(x)
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return x

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps 
                )
        return dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler,
                    interval='step'
                    )
                )

    def training_step(self, batch, batch_idx):
        features, labels = batch
        y_hat = self(features)
        loss = self.criterion(y_hat, labels)
        self.log("train_loss", loss)
        return loss

    def train_dataloader(self):
        dataset = HousePriceDataset(Train)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
        return loader

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        y_hat = self(features)
        loss = self.criterion(y_hat, labels)
        self.log("valid_loss", loss)
        return loss

    def val_dataloader(self):
        dataset = HousePriceDataset(Valid)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
        return loader

if __name__ == '__main__':
    trainer = Trainer(callbacks=[EarlyStopping(monitor='valid_loss', mode='min', patience=10),
                                 TQDMProgressBar(process_position=2)],
                      max_epochs=N_EPOCHS,
                      log_every_n_steps=10,
                      fast_dev_run=False
                      )
    input_size: int = Train.shape[1] - 1
    print(input_size)
    model = LinearModel(input_size, HIDDEN_SIZE, N_HIDDEN_LAYERS)
    trainer.fit(model)

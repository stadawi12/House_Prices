import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader import HousePriceDataset
from model import LinearModel
from utils import test, train

df = pd.read_csv('data/clean_data.csv', index_col=[0, 1])

training_data = HousePriceDataset(df, "train")
testing_data = HousePriceDataset(df, "test")

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size = 64, shuffle=True)

model = LinearModel(len(df.columns)-1)

loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

analytics = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": [],
             "epochs": [] 
            }

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimiser, epoch)
    test_loss, test_acc = test(train_dataloader, model, loss_fn, epoch)

    if epoch > 1:

        analytics["train_loss"].append(train_loss)
        analytics["train_acc"].append(train_acc)
        analytics["test_loss"].append(test_loss)
        analytics["test_acc"].append(test_acc)
        analytics["epochs"].append(epoch)

# Generate figure and axis
fig, ax = plt.subplots(2)

# construct plots
ax[0].plot(analytics["epochs"], analytics["train_loss"], label = 'train_loss')
ax[1].plot(analytics["epochs"], analytics["train_acc"],  label = 'train_acc')
ax[0].plot(analytics["epochs"], analytics["test_loss"],  label = 'test_loss')
ax[1].plot(analytics["epochs"], analytics["test_acc"],   label = 'test_acc')

# create legends
ax[0].legend()
ax[1].legend()

# set titles
ax[0].set_title('Loss curves')
ax[1].set_title('Accuracy curves')

plt.show()

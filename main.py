import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader import HousePriceDataset
from model import LinearModel
from utils import test, train

from sklearn.model_selection import train_test_split

df = pd.read_csv('data/clean_train.csv', index_col=[0])

df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

training_data = HousePriceDataset(df_train)
testing_data = HousePriceDataset(df_test)

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size = 64, shuffle=True)

model = LinearModel(len(df_train.columns)-1)

loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

analytics = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": [],
             "epochs": [] 
            }

EPOCHS = 100
for epoch in range(1, EPOCHS + 1):

    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimiser, epoch)
    test_loss, test_acc = test(test_dataloader, model, loss_fn, epoch)

    if epoch > 1:

        analytics["train_loss"].append(train_loss)
        analytics["train_acc"].append(train_acc)
        analytics["test_loss"].append(test_loss)
        analytics["test_acc"].append(test_acc)
        analytics["epochs"].append(epoch)

torch.save(model.state_dict(), 'model.pt')

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

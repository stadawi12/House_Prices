""" Here we train the model """
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataloader import HousePriceDataset
from model import LinearModel
from utils import test, train

torch.manual_seed(42)

EPOCHS = 100
LR = 0.01
SPLIT = 0.9
DECAY=0.6

df = pd.read_csv('data/clean_train.csv', index_col=[0])

df_train = df.sample(frac=SPLIT, random_state=42)
df_test = df.drop(df_train.index)
print(f"train set shape: {df_train.shape}, test set shape: {df_test.shape}")

training_data = HousePriceDataset(df_train)
testing_data = HousePriceDataset(df_test)

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size = 64, shuffle=True)

model = LinearModel(len(df_train.columns)-1)

loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimiser, step_size=10, gamma=DECAY)

analytics = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": [],
             "epochs": [] 
            }

for epoch in range(1, EPOCHS + 1):

    print(scheduler.get_last_lr())
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimiser, epoch)
    test_loss, test_acc = test(test_dataloader, model, loss_fn, epoch)
    scheduler.step()

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

plt.savefig('images/acc_loss_curves.pdf', format='pdf')

plt.show()

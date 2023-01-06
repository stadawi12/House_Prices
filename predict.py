import torch
from model import LinearModel
import pandas as pd
from dataloader import HousePriceDataset
from torch.utils.data import DataLoader
import numpy as np

df = pd.read_csv('data/clean_test.csv', index_col=[0])

X = torch.from_numpy(df.to_numpy().astype(np.float32))

print(type(X))

model = LinearModel(len(df.columns))
model.load_state_dict(torch.load('model.pt'))
model.eval()

pred = torch.exp(model(X))

sample_submission = pd.read_csv('data/sample_submission.csv', index_col=[0])

sample_submission['SalePrice'] = pred.detach().numpy()

sample_submission.to_csv('data/my_submission.csv')

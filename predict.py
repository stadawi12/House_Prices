import torch
from model import LinearModel
import pandas as pd
import numpy as np

df = pd.read_csv('data/clean_test.csv', index_col=[0])

X = torch.from_numpy(df.to_numpy().astype(np.float32))

print(type(X))

model = LinearModel(len(df.columns))
model.load_state_dict(torch.load('saved_models/model4.pt'))
model.eval()

pred = torch.exp(model(X))

sample_submission = pd.read_csv('data/sample_submission.csv', index_col=[0])

sample_submission['SalePrice'] = pred.detach().numpy()

sample_submission.to_csv('data/model4_prediction.csv')

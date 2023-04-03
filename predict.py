import torch
from model import LinearModel
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/test.csv', index_col=[0])

X = torch.from_numpy(df.to_numpy().astype(np.float32))

print(type(X))

checkpoint = './lightning_logs/version_1/checkpoints/epoch=38-step=1443.ckpt'
model = LinearModel.load_from_checkpoint(checkpoint)
model.eval()

pred = torch.exp(model(X))

sample_submission = pd.read_csv('data/sample_submission.csv', index_col=[0])

sample_submission['SalePrice'] = pred.detach().numpy()

sample_submission.to_csv('data/prediction.csv')

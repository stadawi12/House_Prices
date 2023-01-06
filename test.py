import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader import HousePriceDataset
from model import LinearModel
from utils import test, train

df = pd.read_csv('data/clean_data.csv', index_col=[0, 1])

print(df.shape)

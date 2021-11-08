import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

'''
'../蛋壳颜色打分表.csv'
'''


def my_collate_fn(batch):
    score = []
    rgb = []
    for i in batch:
        score.append(i[0]-95)
        rgb.append(i[1:4])
    rgb = torch.tensor(rgb).reshape(50, 3)
    return torch.tensor(score).reshape(50, 1), rgb


def default_loader(data_path):
    return pd.read_csv(data_path)


class csvDataSet(Dataset):
    def __init__(self, data_path, img_transform=None, loader=default_loader):
        self.df = loader(data_path)

    def __getitem__(self, index):
        Score = self.df['score']
        R = self.df['R']
        G = self.df['G']
        B = self.df['B']
        return Score[index], R[index], G[index], B[index]

    def __len__(self):
        return self.df.__len__()


def prepare_data(data_path):
    DataSet = csvDataSet(data_path)
    return DataSet

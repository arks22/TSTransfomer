import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        assert len(data) == len(labels), "The sizes of data and labels must match"
        
        # チャンネルの次元を追加
        data = data[:, :, np.newaxis] # (80, 1250) -> (80, 1250, 1)
        # NumPy配列をPyTorchテンソルに変換
        self.data   = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx番目のシークエンスデータとラベルを取得
        # 時系列データは(80,)の形状で、対応するラベルはスカラー
        data = self.data[idx]
        label = self.labels[idx]

        return data, label

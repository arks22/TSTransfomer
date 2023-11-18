import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

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


# ラベルの作成
def label_price_movement(data, threshold):
    labels = []
    for i in range(len(data)):
        if i == len(data) - 1:
            break # 最後のシークエンスはラベルを作成できないのでスキップ

        current_price = data[i][-1] # 現在の価格(シークエンスの最後の価格)
        future_prices = data[i+1][0] # 未来の価格(次のシークエンスの最初の価格)
        
        # ラベルを決定
        if future_prices > current_price * (1 + threshold):
            labels.append(0) # 上昇
        elif future_prices < current_price * (1 - threshold):
            labels.append(1) # 下落
        else:
            labels.append(2) # 変化なし
            
    # 全体の分布をprint
    print(f'Labels: {np.bincount(labels)}')
        
    labels = np.array(labels)

    return labels
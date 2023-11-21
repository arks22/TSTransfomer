from tqdm import tqdm
import torch
import numpy as np

def train_reg(params, model, dataloader, criterion, optimizer):
    model.train() # モデルを訓練モードに設定
    total_loss = 0
    total_acc = 0

    with tqdm(dataloader, unit="batch", leave=False) as tepoch:
        for train_data, train_labels in tepoch:

            # GPUにデータを送る
            train_data, train_labels= train_data.to('cuda'), train_labels.float().to('cuda')
            
            optimizer.zero_grad() # 勾配の初期化
            outputs = model(train_data).squeeze() # 順伝播 # 次元を削減
            loss = criterion(outputs, train_labels) # 損失関数の計算
            total_loss += loss.item()

            # 逆伝播
            loss.backward() # 勾配計算
            optimizer.step() # パラメータの更新
            
            # 正解率の計算
            predicted_labels = torch.zeros(train_labels.size()).to('cuda')
            predicted_labels = torch.where(outputs < 0, 0, 1)
            train_labels     = torch.where(train_labels < 0, 0, 1)
            correct_predictions = (predicted_labels == train_labels).sum().item()
            samples_n = train_data.size(0)
            total_acc += correct_predictions / samples_n
            
            
    loss = total_loss / len(dataloader)
    acc =  total_acc / len(dataloader)
    
    return loss, acc
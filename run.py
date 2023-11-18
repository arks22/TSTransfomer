import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

from model import TimeSeriesTransformer
from dataset import TimeSeriesDataset, label_price_movement



def main():
    # ----------- Data Preparation -----------
    print('Data Preparation....')
    
    data, labels = generate_dummy_data(10000) # ダミーの時系列データの生成
    train_data, train_labels = data[:8000], labels[:8000] # 訓練データとラベル
    val_data, val_labels = data[8000:9000], labels[8000:9000] # 検証データとラベル  
    test_data, test_labels = data[9000:], labels[9000:] # テストデータとラベル

    train_dataset = TimeSeriesDataset(train_data, train_labels) # データセットの作成
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) # DataLoaderの作成
    
    val_dataset = TimeSeriesDataset(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    test_dataset = TimeSeriesDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f'Train Data: {len(train_dataset)} | Val Data: {len(val_dataset)} | Test Data: {len(test_dataset)}')

    # ----------- Model Definition -----------
    print('Model Definition....')
    input_dim = 1  # Modify this as per your input features
    time_window = 80  # Modify this as per your time series window
    output_classes = 3  # Modify this for "up", "down", "flat" classes
    initial_lr = 0.1

    model = TimeSeriesTransformer(input_dim=input_dim, time_window=time_window, output_classes=output_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)

    # ----------- Training -----------
    print('Training....')
    max_epochs = 100
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    val_loss = 10
    for epoch in range(max_epochs):

        # Train
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer) 

        # Validation
        val_loss, val_acc = test(model, val_dataloader, criterion)
        val_loss, val_acc = round(val_loss, 3), round(val_acc, 3)

        # 学習率の更新
        if epoch > 30:
            scheduler.step(val_loss)
        
        # Print
        print(f'Epoch {epoch+1} | Train Loss {train_loss} | Train Acc {train_acc} | Val Loss {val_loss} | Val Acc {val_acc}')

        # プロット
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        plot_loss(train_loss_list, val_loss_list, train_acc_list, val_acc_list)
        
        # モデルの保存
        if (epoch + 1) % 10 == 0:
            pass
            #torch.save(model.state_dict(), f'time_series_transformer-{epoch}.pth')
            

def generate_dummy_data(data_len):
    step = 0.1  # 時点間のステップ
    time_window = 80  # サンプルごとの時系列データの長さ

    def sinnp(n, line):
        return np.sin(n * line / 4)

    def cosnp(n, line):
        return np.cos(n * line / 4)

    t = np.arange(0, data_len * time_window * step, step)
    raw_data = (sinnp(1, t) + sinnp(3, t) + sinnp(10, t) + cosnp(5, t) + cosnp(7, t)) / 5
    raw_data = raw_data + (np.random.rand(len(t)) * 0.05)# ノイズ項

    # 正規化
    raw_data = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))
    
    # 確認のためにプロット、保存
    plt.figure(figsize=(20, 4))  # プロットのサイズを1:5に変更
    plt.plot(raw_data[:160])
    plt.savefig('time_series.png')

    # 要素数がtime_windowになるように分割
    len_data = len(raw_data) // time_window # 100000 / 80 = 1250
    raw_data = raw_data[:len_data * time_window] # 100000 -> 1250 * 80 = 100000
    data = raw_data.reshape(len_data, time_window) #  (80, 1250)

    # ラベルの作成
    threshold = 0.05 # 10%の変化を閾値とする
    labels = label_price_movement(data, threshold) # ラベルの作成

    data = data[:-1] # 最後のシーケンスはラベルを作成できないので削除
    
    return data, labels


def train(model, dataloader, criterion, optimizer):
    model.train() # モデルを訓練モードに設定
    total_loss = 0
    total_acc = 0

    with tqdm(dataloader, unit="batch", leave=False) as tepoch:
        for batch_data, batch_labels in tepoch:

            # GPUにデータを送る
            #batch_data = batch_data.to('cuda')
            #batch_labels = batch_labels.to('cuda')
            
            optimizer.zero_grad() # 勾配の初期化
            outputs = model(batch_data) # 順伝播
            loss = criterion(outputs, batch_labels) # 損失関数の計算

            # 逆伝播
            loss.backward() # 勾配計算
            optimizer.step() # パラメータの更新

            # 正解数の計算
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions = (predicted_labels == batch_labels).sum().item()
            samples_n = batch_data.size(0)
            
            # 統計情報の更新
            total_loss += loss.item()
            total_acc += correct_predictions / samples_n
            
    loss_epoch = round(total_loss / len(dataloader), 3)
    acc_epoch = round(total_acc / len(dataloader) ,3)
    
    return loss_epoch, acc_epoch


# モデルをテストしてlossとaccuracyを計算
def test(model, dataloader, critertion):
    model.eval()
    
    total_loss, total_acc = 0, 0

    for test_data, test_labels in dataloader:
        with torch.no_grad():
            outputs = model(test_data) # 順伝播
            total_loss += critertion(outputs, test_labels) # 損失関数の計算

            # accuracyの計算
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions = (predicted_labels == test_labels).sum().item()
            samples_n = test_data.size(0)
            total_acc += correct_predictions / samples_n
    
    loss = total_loss / len(dataloader)
    accuracy = total_acc / len(dataloader)

    return loss.item(), accuracy


# 損失とaccuracyのプロット
def plot_loss(train_loss_list, test_loss_list, train_acc_list, test_acc_list):
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='train')
    plt.plot(test_loss_list, label='test')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='train')
    plt.plot(test_acc_list, label='test')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('loss_acc.png')
    plt.close()
    

if __name__ == '__main__':
    main()
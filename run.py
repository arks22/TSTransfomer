import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from model import TimeSeriesTransformer
from dataset import TimeSeriesDataset
from dummy_data import generate_dummy_data


def split_data(data, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # データの分割
    train_data = data[:int(len(data) * train_ratio)]
    train_labels = labels[:int(len(data) * train_ratio)]
    val_data = data[int(len(data) * train_ratio):int(len(data) * (train_ratio + val_ratio))]
    val_labels = labels[int(len(data) * train_ratio):int(len(data) * (train_ratio + val_ratio))]
    test_data = data[int(len(data) * (train_ratio + val_ratio)):]
    test_labels = labels[int(len(data) * (train_ratio + val_ratio)):]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def main():
    # ----------- Data Preparation -----------
    print('Data Preparation....')
    batach_size = 64
    
    data = np.load('data/timeseries.npy')
    labels = np.load('data/label.npy')
    
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(data, labels)
    print(train_data.shape, train_labels.shape)
    print(val_data.shape, val_labels.shape)
    print(test_data.shape, test_labels.shape)

    plt.figure(figsize=(10, 4)) 
    plt.plot(train_data[0])
    plt.savefig('time_series.png')
    
    train_dataset = TimeSeriesDataset(train_data, train_labels) # データセットの作成
    train_dataloader = DataLoader(train_dataset, batch_size=batach_size, shuffle=True) # DataLoaderの作成
    
    val_dataset = TimeSeriesDataset(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batach_size, shuffle=False)
    
    test_dataset = TimeSeriesDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batach_size, shuffle=False)
    print(f'Train Data: {len(train_dataset)} | Val Data: {len(val_dataset)} | Test Data: {len(test_dataset)}')

    # ----------- Model Definition -----------
    print('Model Definition....')
    input_dim = 1  # Modify this as per your input features
    time_window = 80  # Modify this as per your time series window
    output_classes = 3  # Modify this for "up", "down", "flat" classes
    initial_lr = 0.1

    model = TimeSeriesTransformer(input_dim=input_dim, time_window=time_window, output_classes=output_classes).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)

    # ----------- Training -----------
    print('Training....')
    max_epochs = 80
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    val_loss = 10
    for epoch in range(max_epochs):

        # Train
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer) 

        # Validation
        val_loss, val_acc, _ = test(model, val_dataloader, criterion)
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

    # ----------- Test -----------
    test_loss, test_acc, class_accuracy = test(model, test_dataloader, criterion, mode='test')
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")

    for i, cl_acc in enumerate(class_accuracy):
        print(f"Class {i} Accuracy: {cl_acc:.2f}")

            
def train(model, dataloader, criterion, optimizer):
    model.train() # モデルを訓練モードに設定
    total_loss = 0
    total_acc = 0

    with tqdm(dataloader, unit="batch", leave=False) as tepoch:
        for batch_data, batch_labels in tepoch:

            # GPUにデータを送る
            batch_data, batch_labels= batch_data.to('cuda'), batch_labels.to('cuda')
            
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
def test(model, dataloader, critertion, mode='val'):
    model.eval()
    
    total_loss, total_acc = 0, 0
    all_labels = []
    all_preds = []

    for test_data, test_labels in dataloader:

        test_data, test_labels = test_data.to('cuda'), test_labels.to('cuda')
        with torch.no_grad():
            outputs = model(test_data) # 順伝播
            total_loss += critertion(outputs, test_labels) # 損失関数の計算

            # accuracyの計算
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions = (predicted_labels == test_labels).sum().item()
            samples_n = test_data.size(0)
            total_acc += correct_predictions / samples_n
            
            # 混同行列の計算
            all_labels.extend(test_labels.cpu().numpy())
            all_preds.extend(predicted_labels.cpu().numpy())
            
            # 確信度スコアの取得
            scores = torch.softmax(outputs, dim=1)
            print("Confidence scores:\n", scores)

    loss = total_loss / len(dataloader)
    accuracy = total_acc / len(dataloader)
   
    # 混同行列の計算とプロット
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{mode}.png')
    plt.close()
    
    # クラスごとの正解率の計算
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    return loss.item(), accuracy, class_accuracy


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
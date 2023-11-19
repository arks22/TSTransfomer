import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import argparse
import os
from datetime import datetime
import shutil
import optuna

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

def objective(trial):
    # ----------- Hyperparameters -----------
    print('Hyperparameters....')
    params = {
        "data_name": "btc",
        "output_classes_n": 2,
        "output_classes": ["Up", "Down"],
        "initial_lr": trial.suggest_float('initial_lr', 1e-3, 5e-1),
        "batch_size": trial.suggest_int('batch_size', 16, 128),
        "max_epochs": 50,
        "time_window": 80,
        "confidence_threshold": 0.8,
        "cnn_input_dim": 1,
        "cnn_kernel_size": 16,
        "cnn_stride": 8,
        "transformer_depth": trial.suggest_int('transfomer_depth', 4,16),
        "transformer_heads": 4,
        "embedding_dim": 128,
        "dropout_rate": 0.3
    }
    gen_dir = ""

    # ----------- Data Preparation -----------
    print('Data Preparation....')
    batach_size = params['batch_size']
    
    if params['data_name'] == 'dummy':
        data, labels = generate_dummy_data(100000)
    elif params['data_name'] == 'btc':
        data = np.load('data/timeseries.npy')
        labels = np.load('data/label_2class.npy')
    
    
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(data, labels)

    plt.figure(figsize=(10, 4)) 
    plt.plot(train_data[0])
    #plt.savefig(os.path.join(gen_dir, 'time_series.png'))
    
    train_dataset = TimeSeriesDataset(train_data, train_labels) # データセットの作成
    train_dataloader = DataLoader(train_dataset, batch_size=batach_size, shuffle=True) # DataLoaderの作成
    
    val_dataset = TimeSeriesDataset(val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batach_size, shuffle=False)
    
    test_dataset = TimeSeriesDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batach_size, shuffle=False)
    print(f'Train Data: {len(train_dataset)} | Val Data: {len(val_dataset)} | Test Data: {len(test_dataset)}')

    # ----------- Model Definition -----------
    print('Model Definition....')
    model = TimeSeriesTransformer(params).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params["initial_lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)

    # ----------- Training -----------
    print('Training....')
    max_epochs = params["max_epochs"]
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    val_loss = 10
    for epoch in range(max_epochs):

        # Train
        train_loss, train_acc = train(params, model, train_dataloader, criterion, optimizer) 

        # Validation
        val_loss, val_acc, _ = test(params, model, val_dataloader, criterion, gen_dir)
        val_loss, val_acc = round(val_loss, 3), round(val_acc, 3)

        # Print
        print(f'Epoch {epoch+1} | Train Loss {train_loss} | Train Acc {train_acc} | Val Loss {val_loss} | Val Acc {val_acc}')

        
    # ----------- Test -----------
    test_loss, test_acc, class_accuracy = test(params, model, test_dataloader, criterion, gen_dir, mode='test')

    return test_acc
    

            
def train(params, model, dataloader, criterion, optimizer):
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
def test(params, model, dataloader, critertion, gen_dir, mode='val'):
    model.eval()
    
    total_loss, total_acc = 0, 0
    all_labels = []
    all_preds = []
    high_confidence_preds = []
    high_confidence_labels = []
    confidence_threshold = params['confidence_threshold']

    for test_data, test_labels in dataloader:

        test_data, test_labels = test_data.to('cuda'), test_labels.to('cuda')
        with torch.no_grad():
            outputs = model(test_data) # 順伝播
            total_loss += critertion(outputs, test_labels) # 損失関数の計算

            # 確信度スコアの取得
            scores = torch.softmax(outputs, dim=1)

            # accuracyの計算
            _, predicted_labels = torch.max(outputs, 1)
            high_confidence = scores.max(dim=1).values > confidence_threshold

            high_confidence_preds.extend(predicted_labels[high_confidence].cpu().numpy())
            high_confidence_labels.extend(test_labels[high_confidence].cpu().numpy())

            correct_predictions = (predicted_labels == test_labels).sum().item()
            samples_n = test_data.size(0)
            total_acc += correct_predictions / samples_n
            
            # 混同行列の計算
            all_labels.extend(test_labels.cpu().numpy())
            all_preds.extend(predicted_labels.cpu().numpy())
            
    # 高確信度の予測に基づく混同行列と正解率の計算
    if high_confidence_preds:
        cm = confusion_matrix(high_confidence_labels, high_confidence_preds)
        high_confidence_acc = np.mean(np.array(high_confidence_labels) == np.array(high_confidence_preds))
        #print("High Confidence Accuracy: ", high_confidence_acc)

    loss = total_loss / len(dataloader)
    accuracy = total_acc / len(dataloader)
   
    # 混同行列の計算とプロット
    cm = confusion_matrix(all_labels, all_preds)
    
    # クラスごとの正解率の計算
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    return loss.item(), accuracy, class_accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', storage='sqlite:///db.sqlite3', load_if_exists=True, study_name='btc_50epochs')
    study.optimize(objective, n_trials=100)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
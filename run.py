import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json
import argparse
import os
from datetime import datetime
import shutil

from model import TimeSeriesTransformer
from dataset import TimeSeriesDataset
from dummy_data import generate_dummy_data


def main(args):
    # ----------- Hyperparameters -----------
    print('Hyperparameters....')
    params = load_params(args.config)
    gen_dir = create_dir_with_timestamp(params)
    shutil.copy(args.config, os.path.join(gen_dir, 'config.json'))

    # ----------- Data Preparation -----------
    print('Data Preparation....')
    batch_size = params['batch_size']
    
    if params['data_name'] == 'dummy':
        data, labels = generate_dummy_data(100000)
    elif params['data_name'] == 'btc':
        train_timeseries = np.load(params['train_timeseries_path'])
        train_labels     = np.load(params['train_labels_path'])
        val_timeseries   = np.load(params['val_timeseries_path'])
        val_labels       = np.load(params['val_labels_path'])
        test_timeseries  = np.load(params['test_timeseries_path'])
        test_labels      = np.load(params['test_labels_path'])
        
    
    plt.figure(figsize=(10, 4)) 
    plt.plot(train_timeseries[0])
    plt.savefig(os.path.join(gen_dir, 'time_series.png'))

    train_dataset = TimeSeriesDataset(train_timeseries, train_labels) # データセットの作成
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # DataLoaderの作成

    val_dataset = TimeSeriesDataset(val_timeseries, val_labels)  # Fix: Replace val_data with val_timeseries
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TimeSeriesDataset(test_timeseries, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f'Train Data: {len(train_dataset)} | Val Data: {len(val_dataset)} | Test Data: {len(test_dataset)}')

    # ----------- Model Definition -----------
    print('Model Definition....')
    model = TimeSeriesTransformer(params).to('cuda')
    if params["output_classes_n"] == 2:
        criterion = nn.BCELoss()
    elif params["output_classes_n"] > 2:
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
        val_loss, val_acc, val_class_acc = test(params, model, val_dataloader, criterion, gen_dir)
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
        plot_loss(train_loss_list, val_loss_list, train_acc_list, val_acc_list, gen_dir)

    # ----------- Test -----------
    test_loss, test_acc, test_class_acc = test(params, model, test_dataloader, criterion, gen_dir, mode='test')
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")

    for i, cl_acc in enumerate(test_class_acc):
        print(f"Class {i} Accuracy: {cl_acc:.2f}")

            
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

            if params['output_classes_n'] == 2:
                batch_labels = batch_labels.float() # BCELossの場合はラベルをfloatに変換
                outputs = outputs.squeeze()  # 出力の次元を調整

            loss = criterion(outputs, batch_labels) # 損失関数の計算

            # 逆伝播
            loss.backward() # 勾配計算
            optimizer.step() # パラメータの更新
            
            # 正解率の計算
            if params['output_classes_n'] == 2:
                predicted_labels = outputs.round()  # 出力を0か1に丸める
            else:
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
        if params['output_classes_n'] == 2:
            test_labels = test_labels.float()  # ラベルをfloatに変換
        with torch.no_grad():
            outputs = model(test_data) # 順伝播
            if params['output_classes_n'] == 2:
                outputs = outputs.squeeze()  # 出力の次元を調整
            total_loss += critertion(outputs, test_labels) # 損失関数の計算

            # accuracyの計算
            if params['output_classes_n'] == 2:
                predicted_labels = outputs.round()  # 二値分類の場合、出力を0か1に丸める
            else:
                _, predicted_labels = torch.max(outputs, 1)  # 多クラス分類の場合

            # 高確信度の予測を抽出
            if params['output_classes_n'] == 2: 
                high_confidence = (outputs > confidence_threshold)
            else:
                high_confidence = outputs.max(dim=1).values > confidence_threshold

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
        print("High Confidence Accuracy: ", high_confidence_acc)
        plot_confusion_matrix(cm, classes=params["output_classes"], gen_dir=gen_dir, title=f'cm_{mode}_high_confidence', cmap=plt.cm.Blues)

    loss = total_loss / len(dataloader)
    accuracy = total_acc / len(dataloader)
   
    # 混同行列の計算とプロット
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes=params["output_classes"], gen_dir=gen_dir, title=f'cm_{mode}', cmap=plt.cm.Blues)
    
    # クラスごとの正解率の計算
    class_accuracy = [np.mean(np.array(all_labels) == 0), np.mean(np.array(all_labels) == 1)] if params['output_classes_n'] == 2 else cm.diagonal() / cm.sum(axis=1)

    return loss.item(), accuracy, class_accuracy

# 混同行列のプロット
def plot_confusion_matrix(cm, classes, gen_dir, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',  xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    path = os.path.join(gen_dir, f'{title}.png')
    plt.savefig(path)
    plt.close()


# 損失とaccuracyのプロット
def plot_loss(train_loss_list, test_loss_list, train_acc_list, test_acc_list, gen_dir):
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
    
    path = os.path.join(gen_dir, 'loss_acc.png')
    plt.savefig(path)
    plt.close()
    

def load_params(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params


def create_dir_with_timestamp(params):
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    gen_dir = os.path.join('results', params['data_name'], timestamp)
    
    # フォルダを作成（存在しない場合）
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
        os.chmod(gen_dir, 0o777)
        print(f'フォルダ "{timestamp}" を作成しました。')
    else:
        print(f'フォルダ "{timestamp}" は既に存在します。')
        
    return gen_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    main(args)
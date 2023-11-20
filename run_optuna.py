import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import optuna

from model.model import TimeSeriesTransformer
from data_provider.dataset import TimeSeriesDataset


def objective(trial):
    # ----------- Hyperparameters -----------
    print('Hyperparameters....')
    params = {
        "data_name": "btc",
        "output_classes_n": 2,
        "output_classes": ["Up", "Down"],
        "train_timeseries_path": "data/timeseries_train.npy",
        "train_labels_path": "data/label_2class_train.npy",
        "val_timeseries_path": "data/timeseries_val.npy",
        "val_labels_path": "data/label_2class_val.npy",
        "test_timeseries_path": "data/timeseries_test.npy",
        "test_labels_path": "data/label_2class_test.npy",
        "initial_lr": trial.suggest_float('initial_lr', 1e-3, 9e-1),
        "batch_size": trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024, 2048]),
        "optimizer": trial.suggest_categorical('optimizer', ['SGD', 'AdamW']),
        "scheduler_gamma": trial.suggest_float('scheduler_gamma', 0.8, 1),
        "max_epochs": 20,
        "time_window": 80,
        "confidence_threshold": 0.8,
        "cnn_input_dim": 1,
        "cnn_kernel_size": 16,
        "cnn_stride": 8,
        "transformer_depth": trial.suggest_categorical('transfomer_depth', [2, 4, 8, 16, 32]),
        "transformer_heads": trial.suggest_categorical('transformer_heads', [2, 4, 8, 16, 32]),
        "embedding_dim": 128,
        "dropout_rate": trial.suggest_float('dropout_rate', 0, 0.5),
        "last_layer": "mlp"
    }
    gen_dir = ""

    # ----------- Data Preparation -----------
    print('Data Preparation....')
    batach_size = params['batch_size']
    
    train_timeseries = np.load(params['train_timeseries_path'])
    train_labels     = np.load(params['train_labels_path'])
    val_timeseries   = np.load(params['val_timeseries_path'])
    val_labels       = np.load(params['val_labels_path'])
    test_timeseries  = np.load(params['test_timeseries_path'])
    test_labels      = np.load(params['test_labels_path'])

    train_dataset = TimeSeriesDataset(train_timeseries, train_labels)  # データセットの作成
    train_dataloader = DataLoader(train_dataset, batch_size=batach_size, shuffle=True, num_workers=4)  # DataLoaderの作成

    val_dataset = TimeSeriesDataset(val_timeseries, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batach_size, shuffle=False, num_workers=4)

    test_dataset = TimeSeriesDataset(test_timeseries, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batach_size, shuffle=False, num_workers=4)
    print(f'Train Data: {len(train_dataset)} | Val Data: {len(val_dataset)} | Test Data: {len(test_dataset)}')

    # ----------- Model Definition -----------
    print('Model Definition....')
    model = TimeSeriesTransformer(params).to('cuda')
    criterion = nn.BCELoss()
    optimizer_class = getattr(torch.optim, params["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=params["initial_lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

    # ----------- Training -----------
    print('Training....')
    max_epochs = params["max_epochs"]
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    val_loss = 10
    for epoch in range(max_epochs):

        # Train
        train_loss, train_acc = train(params, model, train_dataloader, criterion, optimizer) 

        # Scheduler
        scheduler.step()

        # Validation
        val_loss, val_acc = test(params, model, val_dataloader, criterion, gen_dir)
        val_loss, val_acc = round(val_loss, 3), round(val_acc, 3)

        # Print
        print(f'Epoch {epoch+1} | Train Loss {train_loss} | Train Acc {train_acc} | Val Loss {val_loss} | Val Acc {val_acc}')

        
    # ----------- Test -----------
    test_loss, test_acc = test(params, model, test_dataloader, criterion, gen_dir, mode='test')

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
            outputs = model(batch_data).squeeze() # 順伝播 # 次元を削減
            batch_labels = batch_labels.float() # BCELossの場合はラベルをfloatに変換

            loss = criterion(outputs, batch_labels) # 損失関数の計算

            # 逆伝播
            loss.backward() # 勾配計算
            optimizer.step() # パラメータの更新

            # 正解数の計算
            predicted_labels = outputs.round()

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
        test_labels = test_labels.float()

        with torch.no_grad():
            outputs = model(test_data).squeeze() # 順伝播
            total_loss += critertion(outputs, test_labels) # 損失関数の計算

            predicted_labels = outputs.round()

            correct_predictions = (predicted_labels == test_labels).sum().item()
            samples_n = test_data.size(0)
            total_acc += correct_predictions / samples_n
            
    loss = total_loss / len(dataloader)
    accuracy = total_acc / len(dataloader)
   
    return loss.item(), accuracy


if __name__ == '__main__':

    study = optuna.create_study(direction='maximize', storage='sqlite:///db.sqlite3', load_if_exists=True, study_name='btc_20')
    study.optimize(objective, n_trials=100)



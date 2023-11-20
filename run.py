import torch
from torch import nn
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import shutil

from model.model import TimeSeriesTransformer
from data_provider.dataset import TimeSeriesDataset
from process_dummy import generate_dummy_data
from run_tst.utils import save_model, plot_loss, load_params, create_dir_with_timestamp
from run_tst.train import train
from run_tst.test import test


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
        train_timeseries, train_labels, val_timeseries, val_labels, test_timeseries, test_labels = generate_dummy_data(100000)  
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

        # Validation
        val_loss, val_acc, val_class_acc = test(params, model, val_dataloader, criterion, gen_dir)
        val_loss, val_acc = round(val_loss, 3), round(val_acc, 3)

        # 学習率の更新
        if epoch > 30:
            scheduler.step(val_loss)

        # プロット
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        plot_loss(train_loss_list, val_loss_list, train_acc_list, val_acc_list, gen_dir)

        # モデルの保存
        if val_acc > max(val_loss_list):
            save_model(model, gen_dir, epoch)

        print(f'Epoch {epoch+1} | Train Loss {train_loss} | Train Acc {train_acc} | Val Loss {val_loss} | Val Acc {val_acc}')

    # ----------- Test -----------
    test_loss, test_acc, test_class_acc = test(params, model, test_dataloader, criterion, gen_dir, mode='test')
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")

    for i, cl_acc in enumerate(test_class_acc):
        print(f"Class {i} Accuracy: {cl_acc:.3f}")

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    main(args)
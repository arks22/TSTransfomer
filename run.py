import torch
from torch import nn

from matplotlib import pyplot as plt
import argparse
import os
import shutil

from model.model import TimeSeriesTransformer
from data_provider.get_dataloader import get_dataloader
from run_tst.utils import save_model, plot_loss, load_params, create_dir_with_timestamp
from run_tst.train import train
from run_tst.test import test


def main(args):
    params = load_params(args.config)
    gen_dir = create_dir_with_timestamp(params)
    shutil.copy(args.config, os.path.join(gen_dir, 'config.json'))
    
    # ----------- Data Loading -----------
    print('Data Loading....')
    train_dataloader, finetune_dataloader, val_dataloader, test_dataloader = get_dataloader(params)

    # ----------- Model Definition -----------
    model = TimeSeriesTransformer(params).to(params['device'])
    criterion = getattr(nn, params["criterion"])()
    
    # 事前学習されたモデルの読み込み
    if params['pretrained_model_path']:
        model.load_state_dict(torch.load(params['pretrained_model_path']))
        print(f'Pretrained model loaded from {params["pretrained_model_path"]}')

    # ----------- Training -----------
    print('Training....')
    optimizer_class = getattr(torch.optim, params["train_optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=params["train_initial_lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["train_scheduler_gamma"])

    train_wrapper(params, gen_dir, train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, params['train_epochs'])

    # ----------- Finetuning -----------
    if params['finetune']:
        print('Finetuning....')
        optimizer_class = getattr(torch.optim, params["finetune_optimizer"])
        optimizer = optimizer_class(model.parameters(), lr=params["finetune_initial_lr"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["finetune_scheduler_gamma"])
        
        train_wrapper(params, gen_dir, finetune_dataloader, val_dataloader, model, criterion, optimizer, scheduler, params['finetune_epochs'], mode='finetune')

    # ----------- Test -----------
    test_loss, test_acc, test_class_acc, _ = test(params, model, test_dataloader, criterion, gen_dir, mode='test')
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")

    for i, cl_acc in enumerate(test_class_acc):
        print(f"Class {i} Accuracy: {cl_acc:.3f}")


def train_wrapper(params, gen_dir, train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, max_epochs, mode='train'):
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    val_loss = 10
    for epoch in range(1, max_epochs+1):

        # Train
        train_loss, train_acc = train(params, model, train_dataloader, criterion, optimizer) 

        # Validation
        val_loss, val_acc, _, high_confidence_acc = test(params, model, val_dataloader, criterion, gen_dir)
        print(f'-- Epoch {epoch} | Train Loss {train_loss} | Train Acc {train_acc:.3f}')
        print(f'           | Val Loss   {val_loss} | Val Acc   {val_acc:.3f}')
        print("            | High Confidence Accuracy: ", high_confidence_acc)

        # 学習率の更新
        scheduler.step()

        # モデルの保存
        if  val_acc >= max(val_acc_list, default=0):
            save_model(model, gen_dir, f'model_{mode}_{epoch}.pth')

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        # プロット
        plot_loss(train_loss_list, val_loss_list, train_acc_list, val_acc_list, gen_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    main(args)
import torch

from matplotlib import pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

def save_model(model, gen_dir, epoch):
    model_path = os.path.join(gen_dir, f'model_{epoch}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    

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

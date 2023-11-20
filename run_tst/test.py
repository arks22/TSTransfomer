import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch

def calculate_class_accuracy(cm):
    class_accuracy = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        class_accuracy.append(TP / (TP + FN) if TP + FN > 0 else 0)
    
    return class_accuracy

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


# モデルをテストしてlossとaccuracyを計算
def test(params, model, dataloader, critertion, gen_dir, mode='val'):
    total_loss, total_acc = 0, 0
    high_confidence_acc = 0
    all_labels = []
    all_preds = []
    high_confidence_preds = []
    high_confidence_labels = []
    confidence_threshold = params['confidence_threshold']

    model.eval()

    for test_data, test_labels in dataloader:
        test_data, test_labels = test_data.to('cuda'), test_labels.to('cuda')

        with torch.no_grad():
            outputs = model(test_data) # 順伝播
            outputs = outputs.squeeze()  # 出力の次元を調整
            total_loss += critertion(outputs, test_labels.float()) # 損失関数の計算

            # accuracyの計算
            predicted_labels = outputs.round()  # 二値分類の場合、出力を0か1に丸める

            # 高確信度の予測を抽出
            high_confidence = torch.logical_or(outputs > confidence_threshold, outputs < 1 - confidence_threshold)


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
    class_accuracy = calculate_class_accuracy(cm)

    return loss.item(), accuracy, class_accuracy, high_confidence_acc
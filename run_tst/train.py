from tqdm import tqdm

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
            loss = criterion(outputs, batch_labels.float()) # 損失関数の計算

            # 逆伝播
            loss.backward() # 勾配計算
            optimizer.step() # パラメータの更新
            
            # 正解率の計算
            predicted_labels = outputs.round()  # 出力を0か1に丸める
            correct_predictions = (predicted_labels == batch_labels).sum().item()
            samples_n = batch_data.size(0)
            
            # 統計情報の更新
            total_loss += loss.item()
            total_acc += correct_predictions / samples_n
            
    loss_epoch = round(total_loss / len(dataloader), 3)
    acc_epoch = round(total_acc / len(dataloader) ,3)
    
    return loss_epoch, acc_epoch
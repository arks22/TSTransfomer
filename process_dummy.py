import numpy as np
import matplotlib.pyplot as plt

def generate_dummy_data(data_len):
    timeseries = dummy_data(data_len+ 1)

    # ラベルの作成
    threshold = 0.05 # 10%の変化を閾値とする
    labels = label_price_movement(timeseries, threshold) # ラベルの作成

    timeseries = normalize(timeseries) # 正規化
    timeseries  = timeseries[:-1] # 最後のシーケンスはラベルを作成できないので削除

    # 分割
    train_timeseries, val_timeseries, test_timeseries = split_data(timeseries)
    train_labels, val_labels, test_labels = split_data(labels)
    
    return train_timeseries, train_labels, val_timeseries, val_labels, test_timeseries, test_labels

    
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # データの分割
    train_data   = data[:int(len(data) * train_ratio)]
    val_data     = data[int(len(data) * train_ratio):int(len(data) * (train_ratio + val_ratio))]
    test_data    = data[int(len(data) * (train_ratio + val_ratio)):]

    return train_data, val_data, test_data


def dummy_data(data_len):
    step = 0.1  # 時点間のステップ
    time_window = 80  # サンプルごとの時系列データの長さ

    def sinnp(n, line):
        return np.sin(n * line / 4)

    def cosnp(n, line):
        return np.cos(n * line / 4)

    t = np.arange(0, data_len * time_window * step, step)
    raw_data = sinnp(1, t) + sinnp(3, t) + sinnp(11, t) + cosnp(5, t) + cosnp(7, t) + cosnp(17, t) + sinnp(23, t) + cosnp(41, t)
    raw_data = raw_data + (np.random.rand(len(t)) * 0.05)# ノイズ項

    
    # 確認のためにプロット、保存
    plt.figure(figsize=(20, 4))  # プロットのサイズを1:5に変更
    plt.plot(raw_data[:160])
    plt.savefig('time_series.png')

    # 要素数がtime_windowになるように分割
    len_data = len(raw_data) // time_window # 100000 / 80 = 1250
    raw_data = raw_data[:len_data * time_window] # 100000 -> 1250 * 80 = 100000
    data = raw_data.reshape(len_data, time_window) #  (80, 1250)
    
    return data


def normalize(data):
    # 正規化
    for i in range(len(data)):
        data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))
    
    return data


# ラベルの作成
def label_price_movement(data, threshold):
    labels = []
    for i in range(len(data) - 1):
        current_price = data[i][-1] # 現在の価格(シークエンスの最後の価格)
        future_prices = sum(data[i + 1]) / len(data[i + 1]) # 次のシークエンスの平均価格
        
        # ラベルを決定
        if future_prices > current_price:
            labels.append(0) # 上昇
        else:
            labels.append(1) # 下落
            
    # 全体の分布をprint
    print(f'Labels: {np.bincount(labels)}')
        
    labels = np.array(labels)

    return labels
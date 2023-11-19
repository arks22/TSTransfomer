import numpy as np
import matplotlib.pyplot as plt

def generate_dummy_data(data_len):
    data = dummy_data(10000 + 1)

    # ラベルの作成
    threshold = 0.05 # 10%の変化を閾値とする
    labels = label_price_movement(data, threshold) # ラベルの作成

    data = data[:-1] # 最後のシーケンスはラベルを作成できないので削除
    
    return data, labels


def dummy_data(data_len):
    step = 0.1  # 時点間のステップ
    time_window = 80  # サンプルごとの時系列データの長さ

    def sinnp(n, line):
        return np.sin(n * line / 4)

    def cosnp(n, line):
        return np.cos(n * line / 4)

    t = np.arange(0, data_len * time_window * step, step)
    raw_data = sinnp(1, t) + sinnp(3, t) + sinnp(11, t) + cosnp(5, t) + cosnp(7, t) + cosnp(17, t) + sinnp(23, t) + cosnp(31, t)
    raw_data = raw_data + (np.random.rand(len(t)) * 0.03)# ノイズ項

    # 正規化
    raw_data = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))
    
    # 確認のためにプロット、保存
    plt.figure(figsize=(20, 4))  # プロットのサイズを1:5に変更
    plt.plot(raw_data[:160])
    plt.savefig('time_series.png')

    # 要素数がtime_windowになるように分割
    len_data = len(raw_data) // time_window # 100000 / 80 = 1250
    raw_data = raw_data[:len_data * time_window] # 100000 -> 1250 * 80 = 100000
    data = raw_data.reshape(len_data, time_window) #  (80, 1250)
    
    return data


# ラベルの作成
def label_price_movement(data, threshold):
    labels = []
    for i in range(len(data)):
        if i == len(data) - 1:
            break # 最後のシークエンスはラベルを作成できないのでスキップ

        current_price = data[i][-1] # 現在の価格(シークエンスの最後の価格)
        future_prices = data[i+1][0] # 未来の価格(次のシークエンスの最初の価格)
        
        # ラベルを決定
        if future_prices > current_price * (1 + threshold):
            labels.append(0) # 上昇
        elif future_prices < current_price * (1 - threshold):
            labels.append(2) # 下落
        else:
            labels.append(1) # 変化なし
            
    # 全体の分布をprint
    print(f'Labels: {np.bincount(labels)}')
        
    labels = np.array(labels)

    return labels
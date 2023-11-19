import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def open_column_as_numpy(file_path):
    print(f'Open: {file_path}')
    # CSVファイルの読み込み
    data = pd.read_csv(file_path, index_col=0, header=[0, 1]).sort_index(axis=1)

    # 'open'の選択
    pandas_data = data['Weighted_Price']
    pandas_data = pandas_data.dropna()

    # NumPy配列に変換
    numpy_data = pandas_data.to_numpy()

    return numpy_data

    
def reshape_data(data, time_window, shift=0):
    if shift < 0:
        raise ValueError("Shift must be a non-negative integer")

    print(f'Reshape: {data.shape} -> {(data.shape[0] - shift) // time_window, time_window}')
    data_len = (len(data) - shift) // time_window
    # 正のシフトを適用
    reshaped_data = data[shift:data_len * time_window + shift].reshape(-1, time_window)
    
    return reshaped_data
    
    
def label_price_movement_2class(data):
    print(f'Labeling.... ')
    labels = []
    for i in range(len(data) - 1):
        current_price = data[i, -1]
        future_price = data[i+1, 0]

        if future_price > current_price:
            labels.append(0)  # 上昇
        else:
            labels.append(1)  # 下落

    print(f'Labels: {np.bincount(labels)}')
    return np.array(labels)

    
def label_price_movement_3class(data, threshold):
    print(f'Labeling.... ')
    labels = []
    for i in range(len(data) - 1):
        current_price = data[i, -1]
        future_price = data[i+1, 0]
        
        if future_price > current_price * (1 + threshold):
            labels.append(0) # 上昇
        elif future_price < current_price * (1 - threshold):
            labels.append(2) # 下落
        else:
            labels.append(1) # 変化なし


    print('[ Up | Flat | Down ]')
    print(np.bincount(labels))
    return np.array(labels)


# 標準化
def preprocessing(data):
    print(f'Preprocessing.... ')
    # 各timewindow内で標準化
    """
    for i in range(len(data)):
        data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])
    """
    # 各timewindow内で正規化
    for i in range(len(data)):
        data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))

    return data

def plot_data(data, time_window, labels):

    plt.figure(figsize=(40, 8))

    # 0から240のデータポイントをプロット
    for i in range(20000,20010):
        color = 'g' if labels[i] == 0 else 'r'  # 緑色は上昇、赤色は下落を示す
        
        window_data = data[i]
        time_steps = np.arange(i * time_window, (i + 1) * time_window)
        plt.plot(time_steps, window_data, color=color)
        plt.axvline(x=(i + 1) * time_window - 1, color='gray', linestyle='--')

    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title('Time Series Data Plot')
    plt.legend()
    plt.savefig('time_series.png')


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # データの分割
    train_data   = data[:int(len(data) * train_ratio)]
    val_data     = data[int(len(data) * train_ratio):int(len(data) * (train_ratio + val_ratio))]
    test_data    = data[int(len(data) * (train_ratio + val_ratio)):]

    return train_data, val_data, test_data

    
def main():
    source_path = 'data/btc.csv'

    time_window = 80
    threshold = 0.0002

    # 時系列データの作成
    timeseries = open_column_as_numpy(source_path)
    timeseries = reshape_data(timeseries , time_window)
    
    # valとtestの分割
    train_timeseries, val_timeseries, test_timeseries = split_data(timeseries)
    
    # ラベルの作成
    train_label = label_price_movement_2class(train_timeseries)
    val_label   = label_price_movement_2class(val_timeseries)
    test_label  = label_price_movement_2class(test_timeseries)

    # 最後のデータのみラベルがないため削除
    train_timeseries = train_timeseries[:-1]
    val_timeseries   = val_timeseries[:-1]
    test_timeseries  = test_timeseries[:-1]
    
    # データ拡張
    augument_timeseries_list = []
    augument_label_list = []
    for i in range(1, 10):
        # データのシフト
        shift = 8 * i  # シフト値を 8, 16, 24, ... と増加させる
        print(f'Shift: {shift}')
        train_timeseries_shited = reshape_data(train_timeseries, time_window, shift=shift)
        
        # ラベルの作成
        train_label_shifted = label_price_movement_2class(train_timeseries_shited)

        # 最後のデータはラベルがないため削除
        train_timeseries_shited = train_timeseries_shited[:-1]

        augument_timeseries_list.append(train_timeseries_shited)
        augument_label_list.append(train_label_shifted)
    
    # データの結合
    train_timeseries = np.concatenate([train_timeseries] + augument_timeseries_list, axis=0)
    train_label      = np.concatenate([train_label]      + augument_label_list, axis=0)

    # 標準化(元のデータでラベルを作成した後に行う)
    train_timeseries = preprocessing(train_timeseries)
    val_timeseries   = preprocessing(val_timeseries)
    test_timeseries  = preprocessing(test_timeseries)
    
    print(f'TRAIN | Time Series Data: {train_timeseries.shape} | Label Data: {train_label.shape}')
    print(f'VAL   | Time Series Data: {val_timeseries.shape} | Label Data: {val_label.shape}')
    print(f'TEST  | Time Series Data: {test_timeseries.shape} | Label Data: {test_label.shape}')

    train_timeseries_path = 'data/timeseries_train.npy'
    val_timeseries_path   = 'data/timeseries_val.npy'
    test_timeseries_path  = 'data/timeseries_test.npy'
    train_label_path      = 'data/label_train.npy'
    val_label_path        = 'data/label_val.npy'
    test_label_path       = 'data/label_test.npy'
    
    np.save(train_timeseries_path, train_timeseries)
    np.save(val_timeseries_path, val_timeseries)
    np.save(test_timeseries_path, test_timeseries)
    np.save(train_label_path, train_label)
    np.save(val_label_path, val_label)
    np.save(test_label_path, test_label)
    
if __name__ == '__main__':
    main()
import pandas as pd

import pandas as pd
import numpy as np


def open_column_as_numpy(file_path):
    print(f'Open: {file_path}')
    # CSVファイルの読み込み
    data = pd.read_csv(file_path, index_col=0, header=[0, 1]).sort_index(axis=1)
    print(data.head())

    # 'open'の選択
    pandas_data = data['Weighted_Price']
    pandas_data = pandas_data.dropna()

    # NumPy配列に変換
    numpy_data = pandas_data.to_numpy()

    return numpy_data

    
def reshape_data(data, time_window):
    print(f'Reshape: {data.shape} -> {data.shape[0] // time_window, time_window}')
    data_len = len(data) // time_window
    # データの再構成
    reshaped_data = data[:data_len * time_window].reshape(-1, time_window)
    
    return reshaped_data
    
    
def label_price_movement(data, threshold):
    print(f'Labeling.... ')
    labels = []
    for i in range(len(data) - 1):
        current_price = data[i, -1]
        future_price = data[i+1, 0]

        if future_price > current_price:
            labels.append(0)  # 上昇
        else:
            labels.append(1)  # 下落

        """
        if future_price > current_price * (1 + threshold):
            labels.append(0)  # 上昇
        elif future_price < current_price * (1 - threshold):
            labels.append(2)  # 下落
        else:
            labels.append(1)  # 変化なし
        """

    print(f'Labels: {np.bincount(labels)}')
    return np.array(labels)


# 標準化
def preprocessing(data):
    print(f'Preprocessing.... ')
    # 各timewindow内で標準化
    for i in range(len(data)):
        data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])

    return data

    
def main():
    file_path = 'data/btc.csv'
    timeseries_path = 'data/timeseries.npy'
    label_path = 'data/label.npy'

    time_window = 80
    threshold = 0.001

    # 時系列データの作成
    timeseries_data = open_column_as_numpy(file_path)
    timeseries_data = reshape_data(timeseries_data, time_window)
    timeseries_data = preprocessing(timeseries_data)

    # ラベルの作成
    label_data = label_price_movement(timeseries_data, threshold)
    
    # 最後のデータはラベルがないため削除
    timeseries_data = timeseries_data[:-1] 
    print(timeseries_data.shape, label_data.shape)
    
    
    np.save(timeseries_path, timeseries_data)
    np.save(label_path, label_data)
    

if __name__ == '__main__':
    main()
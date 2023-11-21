import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def open_column_as_numpy(file_path):
    print(f'Open: {file_path}')
    # CSVファイルの読み込み
    df = pd.read_csv(file_path, index_col=0, header=[0, 1]).sort_index(axis=1)

    # 変化率の計算
    df['Returns'] = df['Weighted_Price'].pct_change()
    
    window_size = 30
    df['Volatility'] = df['Returns'].rolling(window=window_size).std()
    
    # 欠損値の削除
    df = df.dropna()
    
    weighted_price = df['Close']

    # NumPy配列に変換
    numpy_data = weighted_price.to_numpy()
    np_df = df.to_numpy()

    return numpy_data

    
def reshape_data(data, time_window, shift=0):
    if shift < 0:
        raise ValueError("Shift must be a non-negative integer")

    data_len = (len(data) - shift) // time_window
    # 正のシフトを適用
    reshaped_data = data[shift:data_len * time_window + shift].reshape(-1, time_window)
    
    return reshaped_data
    
def label_regression(data, time_window):
    labels = []
    for i in range(len(data) - 1):
        quaterted_time_window = time_window // 4
        current_price = data[i, -1]
        future_window = data[i+1, :quaterted_time_window]
        future_price = sum(future_window) / len(future_window)
        
        change_rate = future_price / current_price - 1
        labels.append(change_rate)

    return np.array(labels)
    
    
def label_price_movement_2class(data, time_window):
    labels = []
    for i in range(len(data) - 1):
        current_price = data[i, -1] # 現在のタイムウィンドウの最後の値
        future_window = data[i+1, :5]
        future_price = sum(future_window) / len(future_window)

        if future_price > current_price:
            labels.append(0)  # 上昇
        else:
            labels.append(1)  # 下落

    return np.array(labels)

    
def label_price_movement_3class(data, threshold):
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

    return np.array(labels)


# 標準化
def preprocessing(data):
    # 各timewindow内で標準化
    """
    for i in range(len(data)):
        data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])
    """
    # 各timewindow内で正規化
    for i in range(len(data)):
        data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))

    return data

def plot_data(data, time_window, labels, slice, title='time_series.png'):

    plt.figure(figsize=(40, 8))

    # 0から240のデータポイントをプロット
    for i in range(slice[0], slice[1]):
        color = 'g' if labels[i] == 0 else 'r'  # 緑色は上昇、赤色は下落を示す
        time_steps = np.arange(i * time_window, (i + 1) * time_window)
        window_data = data[i]

        plt.plot(time_steps, window_data, color='orange')
        plt.axvline(x=(i + 1) * time_window - 1, color='gray', linestyle='--')

        # 次のタイムウィンドウ内における平均価格を水平線でプロット
        next_lavel_window = data[i+1, :5]
        avg_price = sum(next_lavel_window) / len(next_lavel_window)
        plt.hlines(y=avg_price, xmin=(i+1) * time_window, xmax=(i+1) * time_window + 5, color=color, linestyle='-')

        # 現在のタイムウィンドウ内の最後の価格を水平線でプロット
        last_price = data[i, -1]
        plt.hlines(y=last_price, xmin=(i+1) * time_window, xmax=(i+1) * time_window + 5, color='gray', linestyle='-')

        #plt.text(x=i * time_window - 1, y=avg_price, s=f'{label_price:.3f}', color='black', fontsize=20)

    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title('Time Series Data Plot')
    plt.savefig(title)


# データの分割
def split_data(data, ratio_list):
    added_ratio = 0
    dataset_list = []

    for i in range(len(ratio_list)):
        ratio = added_ratio + int(len(data) * ratio_list[i]) # 0 + 10000 * 0.9 = 9000
        dataset_list.append(data[added_ratio:added_ratio + ratio])

        print(f'{i}: {added_ratio} ~ {ratio}')
        added_ratio += ratio
    
    return dataset_list


def delete_last_seq(timeseries):
    timeseries = timeseries[:-1]
    return timeseries
    

def main():
    source_path = 'data/btc.csv'

    time_window = 80
    threshold = 0.0002
    problem = '2class'

    # 時系列データの作成
    timeseries = open_column_as_numpy(source_path)
    timeseries = reshape_data(timeseries , time_window)
    
    # valとtestの分割
    timeseries_list = split_data(timeseries, [0.9, 0.1])

    #finetune_rate = 0.2
    #finetune_timeseries = timeseries_dataset_list[-int(len(train_timeseries) * finetune_rate):]
    
    # ラベルの作成
    label_list = []
    for i in range(len(timeseries_list)):
        if problem == '2class':
            label_list.append(label_price_movement_2class(timeseries_list[i], time_window))
        elif problem == '3class':
            label_list.append(label_price_movement_3class(timeseries_list[i], threshold))
        elif problem == 'regression':
            label_list.append(label_regression(timeseries_list[i], time_window))

        # 最後のデータのみラベルがないため削除
        timeseries_list[i] = delete_last_seq(timeseries_list[i]) 
    
    # データ拡張
    for i in range(len(timeseries_list)):
        augument_timeseries_list = []
        augument_label_list = []
        shift_step = 5
        arange_shift = np.arange(shift_step, time_window, shift_step) # 4, 8, 12, ... , 76
        for shift in arange_shift:
            # データのシフト
            timeseries_shited = reshape_data(timeseries_list[i], time_window, shift=shift)
            
            # ラベルの作成
            if problem == '2class':
                label_shifted = label_price_movement_2class(timeseries_shited, time_window)
            elif problem == '3class':
                label_shifted = label_price_movement_3class(timeseries_shited, threshold)
            elif problem == 'regression':
                label_shifted = label_regression(timeseries_shited, time_window)

            # 最後のデータはラベルがないため削除
            timeseries_shited = delete_last_seq(timeseries_shited)

            # リストに追加
            augument_timeseries_list.append(timeseries_shited)
            augument_label_list.append(label_shifted)
    
        # データの結合
        timeseries_list[i] = np.concatenate([timeseries_list[i]] + augument_timeseries_list, axis=0)
        label_list[i]      = np.concatenate([label_list[i]]      + augument_label_list, axis=0)

    # プロット
    data_name_list = ['train', 'test', 'finetune']
    for i in range(len(timeseries_list)):
        for j in range(7):
            k = j * 100
            slice = [k, k + 10]
            plot_data(timeseries_list[i], time_window, label_list[i], slice=slice, title=f'{data_name_list[i]}_{j}.png')

        # 標準化(元のデータでラベルを作成した後に行う)
        timeseries_list[i] = preprocessing(timeseries_list[i])
     
        #　データの統計情報 
        print(f'{data_name_list[i]} | Timesereis: shape-{timeseries_list[i].shape} Label: shape{label_list[i].shape}, dist-{np.bincount(label_list[i])}')
    
        # データの保存
        np.save(f'data/timeseries_{data_name_list[i]}.npy', timeseries_list[i])
        np.save(f'data/label_{problem}_{data_name_list[i]}.npy', label_list[i])

    
if __name__ == '__main__':
    main()
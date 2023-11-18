import torch
from torch import nn
from torch.nn import functional as F


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, time_window, output_classes, kernel_size=16, stride=8, transformer_depth=4, 
                 transformer_heads=4, embedding_dim=128, dropout_rate=0.3):
        super(TimeSeriesTransformer, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=embedding_dim, 
                                kernel_size=kernel_size, stride=stride)
        
        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=transformer_heads, 
                                                       dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=transformer_depth)
        
        self.positional_encodings = nn.Parameter(torch.zeros(time_window // stride, embedding_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),  # Adjust the size as necessary
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_classes)
        )

    def forward(self, x):
        # xは(batch_size, time_window, input_dim)の形状であることを確認
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, time_window)に変更
        x = self.conv1d(x)
        
        # 位置エンコーディングを追加する際に次元が一致することを確認
        # xの形状が(batch_size, embedding_dim, new_time_window)になるように調整
        # self.positional_encodingsの形状は(new_time_window, embedding_dim)とする
        x = x.permute(0, 2, 1)  # (batch_size, new_time_window, embedding_dim)に変更
        x = x + self.positional_encodings[:x.size(1), :].expand_as(x)

        # Transformerに渡す前に次元を変更
        x = x.permute(1, 0, 2)  # (new_time_window, batch_size, embedding_dim)に変更
        x = self.transformer_encoder(x)
        
        # Transformerの出力をMLPに渡す前に次元を変更
        x = x.permute(1, 0, 2)  # (batch_size, new_time_window, embedding_dim)に変更
        x = x.mean(dim=1)
        
        # MLPを適用
        x = self.mlp(x)
        return x

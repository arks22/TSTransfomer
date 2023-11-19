import torch
from torch import nn
from torch.nn import functional as F


class TimeSeriesTransformer(nn.Module):
    def __init__(self, params):
        super(TimeSeriesTransformer, self).__init__()

        self.input_dim         = params['cnn_input_dim']
        self.time_window       = params['time_window']
        self.output_classes    = params['output_classes_n']
        self.kernel_size       = params['cnn_kernel_size']
        self.stride            = params['cnn_stride']
        self.transformer_depth = params['transformer_depth']
        self.transformer_heads = params['transformer_heads']
        self.embedding_dim     = params['embedding_dim']
        self.dropout_rate      = params['dropout_rate']
        
        self.conv1d = nn.Conv1d(in_channels=self.input_dim, out_channels=self. embedding_dim, 
                                kernel_size=self.kernel_size, stride=self.stride)
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.transformer_heads, dropout=self.dropout_rate)

        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=self.transformer_depth)

        self.positional_encodings = nn.Parameter(torch.zeros(self.time_window // self.stride, self.embedding_dim))
        
        # MLPの層を定義
        mlp_layers = [
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        ]
         # output_classesが1の場合のみシグモイド活性化関数を追加
        if self.output_classes == 2:
            mlp_layers.append(nn.Linear(64, 1))
            mlp_layers.append(nn.Sigmoid())
        elif self.output_classes > 2:
            mlp_layers.append(nn.Linear(64, self.output_classes))
            
        self.mlp = nn.Sequential(*mlp_layers)
            

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

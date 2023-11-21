import torch
from torch import nn
from torch.nn import functional as F


class MLPHead(nn.Module):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.output_classes = params['output_classes_n']
        self.embedding_dim  = params['embedding_dim']
        self.dropout_rate   = params['dropout_rate']
        self.mlp_hidden_dim = params['mlp_hidden_dim']
        self.mlp_input_token = params['mlp_input_token']
        
        layers = [
            nn.Linear(self.embedding_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        ]

        # output_classesが2の場合のみシグモイド活性化関数を追加
        if self.output_classes == 1:  # 回帰問題
            layers.append(nn.Linear(self.mlp_hidden_dim, 1))
        elif self.output_classes == 2: # 2値分類問題
            layers.append(nn.Linear(self.mlp_hidden_dim, 1)) #(batch_size, 1)
            layers.append(nn.Sigmoid())
        elif self.output_classes > 2: # 多クラス分類問題
            layers.append(nn.Linear(64, self.output_classes)) #(batch_size, output_classes)
            
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.mlp_input_token == 'mean':
            x = x.mean(dim=1)               # -> (batch_size, embedding_dim)
        elif self.mlp_input_token == 'last':
            x = x[:, -1, :]                  # -> (batch_size, embedding_dim)

        x = self.mlp(x)                 # -> (batch_size, output_classes)
        
        return x


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
        
        # CNNの層を定義
        self.conv1d = nn.Conv1d(in_channels=self.input_dim,
                                out_channels=self.embedding_dim, 
                                kernel_size=self.kernel_size,
                                stride=self.stride)

        # Positional Encodingを定義
        self.positional_encodings = nn.Parameter(torch.zeros(self.time_window // self.stride, self.embedding_dim))
        
        # Transformer Encoder Layerを定義
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                            nhead=self.transformer_heads,
                                                            dropout=self.dropout_rate,
                                                            batch_first=True)

        # Transformer Encoderを定義
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer,
                                                         num_layers=self.transformer_depth)

        self.head = MLPHead(params)
    

    def forward(self, x):
        x = x.permute(0, 2, 1)          # 　-> (batch_size, input_dim, time_window)
        x = self.conv1d(x)              # -> (batch_size, embedding_dim, new_time_window)
        
        x = x.permute(0, 2, 1)          # -> (batch_size, new_time_window, embedding_dim)
        x = x + self.positional_encodings[:x.size(1), :].expand_as(x)

        # Transformerに渡す前に次元を変更
        x = x.permute(1, 0, 2)          # -> (new_time_window, batch_size, embedding_dim)
        x = self.transformer_encoder(x)
        
        x = x.permute(1, 0, 2)          # -> (batch_size, new_time_window, embedding_dim)
        
        x = self.head(x)                 # -> (batch_size, output_classes)

        return x

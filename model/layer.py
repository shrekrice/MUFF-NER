import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MUFFmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, dropout=0.5, gpu=True):
        super(MUFFmodel, self).__init__()
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, padding=0),
            *[nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1) for _ in range(num_layer - 1)]
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, input_feature):
        input_feature = input_feature.transpose(2, 1).contiguous()
        cnn_output = self.drop(torch.tanh(self.cnn_layers[0](input_feature)))

        for layer in self.cnn_layers[1:]:
            cnn_output = self.drop(torch.tanh(layer(cnn_output)))

        return cnn_output.transpose(2, 1).contiguous()

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(4)
        ])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MuffNermodel(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, num_layer, dropout=0.5, gpu=True, biflag=True):
        super(MuffNermodel, self).__init__()
        self.model_type = 'lstm'
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layer, batch_first=True, bidirectional=biflag)
        self.drop = nn.Dropout(dropout)

    def forward(self, input, mask=None):
        if self.model_type == 'lstm':
            hidden = None
            feature_out, hidden = self.lstm(input, hidden)
            feature_out_d = self.drop(feature_out)

        return feature_out_d

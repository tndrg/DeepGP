import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model,max_len=25000, position = None):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.


        if position is None:
            pe = torch.zeros(max_len, d_model).float()
            position = torch.arange(0, max_len).float().unsqueeze(1)
        else:
            pe = torch.zeros(len(position), d_model).float()
            position = torch.from_numpy(position).float().unsqueeze(1)


        pe.require_grad = False
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(1e6) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding='same', padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, pos, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model,position=pos)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class DataEmbeddingVocab(nn.Module):
    def __init__(self, c_in, d_model, pos, dropout=0.1):
        super(DataEmbeddingVocab, self).__init__()
        self.value_embedding = nn.Embedding(num_embeddings = 201, embedding_dim=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model,position=pos)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x[:,:,0]
        x = torch.round(x,decimals=2)*100
        x = x.int()
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_woPE(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_woPE, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)
    
class CHRDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        if c_in == d_model:
            self.value_embedding  = nn.Identity()
        else:
            self.value_embedding  = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.chr_embedding = nn.Embedding(22, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x,chroms):
        x = self.value_embedding(x) + self.position_embedding(x)+self.chr_embedding(chroms)
        # x = self.value_embedding(x) + self.chr_embedding(chroms)
        return self.dropout(x)
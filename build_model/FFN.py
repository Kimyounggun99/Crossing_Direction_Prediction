from torch import nn


class FFN(nn.Module): # 前馈网络
    def __init__(self, d_model, hidden_dim, rate=0.3, layer_norm_eps=1e-5, device=None):
        super(FFN, self).__init__()

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.linear1 = nn.Linear(d_model, hidden_dim).to(device)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(rate)
        self.linear2 = nn.Linear(hidden_dim, d_model).to(device)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        y = self.linear2(self.dropout1(self.relu(self.linear1(x)))) # 前馈网络
        out = x + self.dropout2(y) 
        out = self.norm(out) # 归一化
        return out
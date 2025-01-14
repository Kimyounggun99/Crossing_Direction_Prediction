import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Bottlenecks_origin(nn.Module): # 瓶颈结构
    def __init__(self, dims):
        super(Bottlenecks_origin, self).__init__()
        self.dims = dims
        self.num_bnks = 3 # 单元数目
        self.num_layers = 4 # 层数
        self.bbox = nn.ModuleList()

        self.bbox.append(nn.Linear(dims, dims + self.num_bnks, bias=True)) 
        

        for _ in range(self.num_layers - 1):
            self.bbox.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def cut(self, x): # 切片
        return x[:, :, :self.dims], x[:, :, -self.num_bnks:] # 从第0个到第dims个，从倒数第num_bnks个到最后一个

    def forward(self, bbox):
        bbox, bnk_bbox = self.cut(self.dropout(self.relu(self.bbox[0](bbox)))) # 生成下一层然后切片，得到bbox下一层和bnk_bbox
        bottlenecks = bnk_bbox 

        for i in range(self.num_layers - 1):
            bbox = torch.cat((bbox, bottlenecks), dim=-1)
            bbox, bnk_bbox = self.cut(self.dropout(self.relu(self.bbox[i + 1](bbox))))
            bottlenecks = bnk_bbox

        return bottlenecks

class Bottlenecks(nn.Module): # 瓶颈结构
    def __init__(self, dims):
        super(Bottlenecks, self).__init__()
        self.dims = dims
        self.num_bnks = 3
        self.num_layers = 4
        self.center = nn.ModuleList()
        self.kp = nn.ModuleList()

        self.center.append(nn.Linear(dims, dims + self.num_bnks, bias=True)) 
        self.kp.append(nn.Linear(dims, dims + self.num_bnks, bias=True))
        for _ in range(self.num_layers - 1):
            self.center.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))
            self.kp.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def cut(self, x): # 切片
        return x[:, :, :self.dims], x[:, :, -self.num_bnks:] 

    def forward(self, center,  kp):
        center, bnk_center = self.cut(self.dropout(self.relu(self.center[0](center))))  
        kp, bnk_kp = self.cut(self.dropout(self.relu(self.kp[0](kp))))
        bottlenecks = bnk_center +  bnk_kp 

        for i in range(self.num_layers - 1):
            center = torch.cat((center, bottlenecks), dim=-1)
            center, bnk_center = self.cut(self.dropout(self.relu(self.center[i + 1](center))))
            kp, bnk_kp = self.cut(self.dropout(self.relu(self.kp[i + 1](torch.cat((kp, bottlenecks), dim=-1)))))
            
            bottlenecks = bnk_center +  bnk_kp

        return bottlenecks
import torch
from torch import nn
import numpy as np
import math
from .model_blocks import EmbedPosEnc, AttentionBlocks, Time_att
from .FFN import FFN
from .BottleNecks import Bottlenecks, Bottlenecks_origin
from einops import repeat

class Transformer_based_model(nn.Module):
    def __init__(self,device, args):
        super(Transformer_based_model, self).__init__()
        self.sigma_cls = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True)
        nn.init.kaiming_normal_(self.sigma_cls, mode='fan_out') 
        self.sigma_reg = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True) 
        nn.init.kaiming_normal_(self.sigma_reg, mode='fan_out')  
        self.device=device
        d_model = args.d_model #128
        hidden_dim = args.hidden_dim #256
        modal_nums = 3
        self.num_layers = args.num_layers#4

        self.num_heads= args.num_heads #8

        self.token = nn.Parameter(torch.ones(1, 1, d_model)).to(device)

        self.center_embedding = EmbedPosEnc(4, d_model,device) 
        self.center_token = nn.Parameter(torch.ones(1, 1, d_model)).to(device)


        self.kp_embedding = EmbedPosEnc(60, d_model,device)  
        self.kp_token = nn.Parameter(torch.ones(1, 1, d_model)).to(device)

        self.center_att = nn.ModuleList() 
        self.center_ffn = nn.ModuleList()

        self.kp_att = nn.ModuleList()
        self.kp_ffn = nn.ModuleList()
        self.cross_att = nn.ModuleList()
        self.cross_ffn = nn.ModuleList()

        for _ in range(self.num_layers):
            self.center_att.append(AttentionBlocks(d_model, self.num_heads, device=device)) 
            self.center_ffn.append(FFN(d_model, hidden_dim, device=device))
            self.kp_att.append(AttentionBlocks(d_model, self.num_heads, device=device))
            self.kp_ffn.append(FFN(d_model, hidden_dim, device=device))
            self.cross_att.append(AttentionBlocks(d_model, self.num_heads, device=device)) 
            self.cross_ffn.append(FFN(d_model, hidden_dim, device=device))

        self.dense = nn.Linear(modal_nums * d_model, 4).to(device)
        self.bottlenecks = Bottlenecks(d_model).to(device)
        self.time_att = Time_att(dims=3).to(device) 
        self.endp = nn.Linear(modal_nums * d_model, 4).to(device) 
        self.relu = nn.ReLU()
        self.last = nn.Linear(3, 1).to(device) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, kp,  center):

        b = center.shape[0]

        center= center
        kp= kp

        token = repeat(self.token, '() s e -> b s e', b=b) 
 
  
        center = self.center_embedding(center, self.center_token) 

        kp = self.kp_embedding(kp, self.kp_token) 
     
        center = self.center_att[0](center) 
        token = torch.cat([token, center[:, 0:1, :]], dim=1)  
       
        kp = self.kp_att[0](kp) 
        token = torch.cat([token, kp[:, 0:1, :]], dim=1) 

        token = self.cross_att[0](token) 
        token_new = token[:, 0:1, :] 
        center = torch.cat([token_new, center[:, 1:, :]], dim=1) 
        kp = torch.cat([token_new, kp[:, 1:, :]], dim=1)
        center = self.center_ffn[0](center)  
        kp = self.kp_ffn[0](kp) 

        token = self.cross_ffn[0](token)[:, 0:1, :] 

        for i in range(self.num_layers - 1):
            center = self.center_att[i + 1](center)
            token = torch.cat([token, center[:, 0:1, :]], dim=1) 
            kp = self.kp_att[i + 1](kp)
            token = torch.cat([token, kp[:, 0:1, :]], dim=1)
            token = self.cross_att[i + 1](token)
            token_new = token[:, 0:1, :]

            center = torch.cat([token_new, center[:, 1:, :]], dim=1)
            kp = torch.cat([token_new, kp[:, 1:, :]], dim=1)
            
            center = self.center_ffn[i + 1](center)
            kp = self.kp_ffn[i + 1](kp)
            token = self.cross_ffn[i + 1](token)[:, 0:1, :]

        bnk = self.relu(self.time_att(self.bottlenecks(center, kp))) 
        tmp = self.last(bnk)
        pred = self.sigmoid(tmp)

        return pred 





class GCN_based_model(nn.Module):
    def __init__(self,device, args):
        super(GCN_based_model, self).__init__()
        self.sigma_cls = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True) 
        nn.init.kaiming_normal_(self.sigma_cls, mode='fan_out') 
        self.sigma_reg = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True) 
        nn.init.kaiming_normal_(self.sigma_reg, mode='fan_out')  
        self.device=device
        d_model = args.d_model

        self.num_layers = args.num_layers
        self.token = nn.Parameter(torch.ones(1, 1, d_model)).to(device)

        self.center_embedding = EmbedPosEnc(4, d_model,device) 
        self.center_token = nn.Parameter(torch.ones(1, 1, d_model)).to(device)



        A = np.stack([np.eye(16)] * args.num_adj_subset, axis=0)
        self.kp_bn = nn.BatchNorm1d(4 * 16).to(device)
        self.kp_pool = nn.AdaptiveAvgPool2d(1).to(device)


        bn_init(self.kp_bn, 1)

        self.kp1 = TCN_GCN_unit(4, 32, A, residual=False, device=device)
        self.kp2 = TCN_GCN_unit(32, 64, A, device=device).to(device)
        self.kp3 = TCN_GCN_unit(64, 128, A, device=device).to(device)
        self.kp4 = TCN_GCN_unit(128, d_model, A, device=device).to(device)
        self.kp_att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, 128, bias=False).to(device),
            nn.BatchNorm1d(128).to(device), 
            nn.Sigmoid()
        )
        


        self.relu = nn.ReLU()
        self.last = nn.Linear(d_model, 1).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, kp,  center):


        center= center

        kp= torch.concat((kp,center), axis=-1)
        N, C, T, V = kp.shape
        kp = kp.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        kp=kp.float()
        kp = self.kp_bn(kp)
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
 
        kp = self.kp1(kp)
        kp = self.kp2(kp)
        kp = self.kp3(kp)
        kp = self.kp4(kp)

        kp= self.kp_pool(kp).squeeze(-1).squeeze(-1)
        
        kp= self.kp_att(kp).mul(kp) +kp

        tmp = self.last(kp)
        pred = self.sigmoid(tmp)

        return pred 






class Transformer_GCN_mixing_model(nn.Module):
    def __init__(self,device, args):
        super(Transformer_GCN_mixing_model, self).__init__()
        self.sigma_cls = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True) # 生成一个可训练的分类损失参数
        nn.init.kaiming_normal_(self.sigma_cls, mode='fan_out') 
        self.sigma_reg = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True) # 生成一个可训练的回归损失参数
        nn.init.kaiming_normal_(self.sigma_reg, mode='fan_out')  
        self.device=device
        d_model = args.d_model #128
        hidden_dim = args.hidden_dim #256
        modal_nums = 3
        self.num_layers = args.num_layers #4

        self.num_heads= args.num_heads#8

        self.token = nn.Parameter(torch.ones(1, 1, d_model)).to(device)

        self.center_embedding = EmbedPosEnc(4, d_model,device) 
        self.center_token = nn.Parameter(torch.ones(1, 1, d_model)).to(device)



        A = np.stack([np.eye(15)] * args.num_adj_subset, axis=0)
        self.kp_bn = nn.BatchNorm1d(4 * 15).to(device)
        self.kp_pool = nn.AdaptiveAvgPool2d(1).to(device)
        bn_init(self.kp_bn, 1)
        self.kp1 = TCN_GCN_unit(4, 32, A, residual=False, device=device)
        self.kp2 = TCN_GCN_unit(32, 64, A, device=device).to(device)
        self.kp3 = TCN_GCN_unit(64, 128, A, device=device).to(device)
        self.kp4 = TCN_GCN_unit(128, d_model, A, device=device).to(device)
        self.kp_att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, 128, bias=False).to(device),
            nn.BatchNorm1d(128).to(device), 
            nn.Sigmoid()
        )
        self.kp_linear=nn.Linear(d_model,3).to(device)

        self.center_att = nn.ModuleList() 
        self.center_ffn = nn.ModuleList()


        for _ in range(self.num_layers):
            self.center_att.append(AttentionBlocks(d_model, self.num_heads, device=device)) 
            self.center_ffn.append(FFN(d_model, hidden_dim, device=device))

        self.dense = nn.Linear(modal_nums * d_model, 4).to(device)
        self.bottlenecks = Bottlenecks_origin(d_model).to(device)
        self.time_att = Time_att(dims=3).to(device) # Time_att
        self.endp = nn.Linear(modal_nums * d_model, 4).to(device) 
        self.relu = nn.ReLU()
        self.last = nn.Linear(6, 1).to(device) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, kp,  center):

        b = center.shape[0]

        center= center
        
        N, C, T, V = kp.shape
        kp = kp.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        kp=kp.float()
        kp = self.kp_bn(kp)
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        kp = self.kp1(kp)
        kp = self.kp2(kp)
        kp = self.kp3(kp)
        kp = self.kp4(kp)

        token = repeat(self.token, '() s e -> b s e', b=b) 
 
  
        center = self.center_embedding(center, self.center_token) 



        center = self.center_att[0](center) 
        token = torch.cat([token, center[:, 0:1, :]], dim=1)  

        token_new = token[:, 0:1, :] 
        center = torch.cat([token_new, center[:, 1:, :]], dim=1) 
        
        center = self.center_ffn[0](center) 


        for i in range(self.num_layers - 1):
            center = self.center_att[i + 1](center)
            token = torch.cat([token, center[:, 0:1, :]], dim=1) 
            token_new = token[:, 0:1, :]

            center = torch.cat([token_new, center[:, 1:, :]], dim=1)
            
            center = self.center_ffn[i + 1](center)
      
        kp= self.kp_pool(kp).squeeze(-1).squeeze(-1)
        
        kp= self.kp_att(kp).mul(kp) +kp
        kp= self.relu(self.kp_linear(kp))

        bnk = self.relu(self.time_att(self.bottlenecks(center))) 
        
        all_feat= torch.concat((kp,bnk),axis=-1)
        tmp = self.last(all_feat)
        pred = self.sigmoid(tmp)

        return pred 

    



def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)



class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,device=None):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1)).to(device)

        self.bn = nn.BatchNorm2d(out_channels).to(device)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, device=None):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True).to(device)
        else:
            self.A = torch.autograd.Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False).to(device)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1).to(device))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1).to(device),
                nn.BatchNorm2d(out_channels).to(device)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels).to(device)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)
    
    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y



class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True,device=None):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive,device=device)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride, device=device)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

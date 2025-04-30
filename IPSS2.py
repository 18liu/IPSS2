import torch
import torch.nn as nn
import math
from einops import rearrange
from torchvision import transforms as transforms, models
import warnings

warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")

class MultiHead_SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        assert input_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        B, L, D = x.size()
        Q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.output_linear(out)
    
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
 
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
    
class CrossPatchAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrossPatchAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.addnorm = AddNorm(128, 0.2)
        self.ma = MultiHead_SelfAttention(128,8)
        self.atten = IPSelf_Attention(256)
        
    def forward(self, x, y):
        B, C, H, W = x.size()
        dis = rearrange(x,'b c h w -> b (h w) c')    
        ref = rearrange(y,'b c h w -> b (h w) c')         
        dis_ma = self.ma(dis)
        ref_ma = self.ma(ref)
        dis_ad = self.addnorm(dis, dis_ma)
        ref_ad = self.addnorm(ref, ref_ma)
        x = rearrange(dis_ad, 'b (h w) c -> b c h w', h=7, w=7)
        y = rearrange(ref_ad, 'b (h w) c -> b c h w', h=7, w=7)
        
        k_out = self.key_conv(x).view(B, -1, W*H) 
        q_y = self.query_conv(y).view(B, -1, W*H).permute(0,2,1)
        energy_ = torch.bmm(q_y,k_out)
        attention_ = self.softmax(energy_)
        v_out = self.value_conv(x).view(B, -1, W*H)
        out_x = torch.bmm(v_out,attention_.permute(0,2,1))
        out_x = out_x.view(B, C, H, W)
        out_x = self.gamma*out_x + x
        
        k_out_ = self.key_conv(y).view(B, -1, W*H) 
        q_y_ = self.query_conv(x).view(B, -1, W*H).permute(0,2,1)
        energy = torch.bmm(q_y_,k_out_)
        attention = self.softmax(energy)
        v_out_ = self.value_conv(y).view(B, -1, W*H)
        out_y = torch.bmm(v_out_,attention.permute(0,2,1))
        out_y = out_y.view(B, C, H, W)
        out_y = self.gamma*out_y + y
        
        out = torch.cat((out_x, out_y),dim=1)
        out = self.atten(out)
        
        return out
    
class IPSelf_Attention(nn.Module):

    def __init__(self, in_dim):
        super(IPSelf_Attention, self).__init__()

        self.qConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.kConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.vConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.position_encoding = PositionalEncoding(d_model=in_dim)  
        self.softmax = nn.Softmax(dim=-1)
        
    def INF(self, B, N, device):
        inf_tensor = torch.full((N, N), float("inf"), device=device)
        inf_matrix = -torch.diag_embed(inf_tensor[0]).unsqueeze(0).repeat(B, 1, 1)
        return inf_matrix

    def forward(self, x):
        bs, C, w, h = x.size()

        x_flattened = x.reshape(bs, C, w * h).permute(0, 2, 1)  

        x_encoded = self.position_encoding(x_flattened)
        x_encoded = x_encoded.permute(0, 2, 1).view(bs, C, w, h)  

        proj_query = self.qConv(x_encoded).view(bs, -1, w * h).permute(0, 2, 1)
        proj_key = self.kConv(x_encoded).view(bs, -1, w * h)
        proj_value = self.vConv(x_encoded).view(bs, -1, w * h)  

        energy = torch.bmm(proj_query, proj_key)
        inf_matrix = self.INF(bs, w * h, x.device)
        energy = energy + inf_matrix

        attention = self.softmax(energy)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, w, h)

        out = self.gamma * out + x

        return out                                      
    
class IPSS2(nn.Module):
    def __init__(self):
        super(IPSS2,self).__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.stage1 = nn.Sequential(*backbone.features[:3])
        self.stage2 = nn.Sequential(*backbone.features[3:5])
        self.stage3 = nn.Sequential(*backbone.features[5:7])
        self.stage4 = nn.Sequential(*backbone.features[7:])

        self.conv_1 = nn.Conv2d(1280, 128, 1)
        self.conv_2 = nn.Conv2d(296, 128, 1)
        self.conv_3 = nn.Conv2d(256, 128, 1)    
                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        self.cpa = CrossPatchAttention(128) 
        self.atten = IPSelf_Attention(128)  
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64, 1)
    
    def fuse(self, x):

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        x1 = nn.functional.adaptive_avg_pool2d(x1, (7, 7))
        x2 = nn.functional.adaptive_avg_pool2d(x2, (7, 7))
        
        feat_local = self.conv_2(torch.cat((x1, x2, x3), dim=1))
        feat_global = self.conv_1(x4)
        
        x_fuse = torch.cat((feat_global, feat_local), dim=1)
        x_fuse = self.conv_3(x_fuse)
        
        return x_fuse
    
    def forward_vector(self, x, y):
        B, N, C, H, W = x.shape
        feature_dis = torch.tensor([]).to(x.device)
        feature_ref = torch.tensor([]).to(x.device)

        for i in range(N):
     
            dis = self.atten(self.fuse(x[:,i,:,:,:]))
            ref = self.atten(self.fuse(y[:,i,:,:,:]))
            
            feature_dis = torch.cat((dis, feature_dis),dim=1)
            feature_ref = torch.cat((ref, feature_ref),dim=1)
            
        return feature_dis, feature_ref
    
    def forward(self, x, y): 

        dis, ref = self.forward_vector(x, y)  
        dis = self.conv_1(dis)                      
        ref = self.conv_1(ref)                      
        feat = self.cpa(dis, ref)
        feat = self.conv_3(feat)
        feat = self.gap(feat)
        feat = self.flatten(feat)
        feat = self.fc1(feat)
        feat = self.fc2(feat)
        score = torch.squeeze(feat,dim=-1)
        
        return score

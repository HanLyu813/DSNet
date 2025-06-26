import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention_HighFreq(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention_HighFreq, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    """x: query, y: key and value"""

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(y)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(y)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + y


class CrissCrossAttention_LowFreq(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention_LowFreq, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    """x: query, y: key and value"""

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(y)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(y)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        """ Low-Freq Filter"""
        att_H_max_indices = att_H.argmax(dim=-1, keepdim=True)  
        att_H_mask = torch.zeros_like(att_H).scatter_(-1, att_H_max_indices, 1)  
        att_H = att_H * att_H_mask  

        att_W_max_indices = att_W.argmax(dim=-1, keepdim=True)  
        att_W_mask = torch.zeros_like(att_W).scatter_(-1, att_W_max_indices, 1)  
        att_W = att_W * att_W_mask  

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + y


class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)  
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        # X_l2h = self.upsample(X_l2h)
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')
        # print('X_l2h:{}'.format(X_l2h.shape))
        # print('X_h2h:{}'.format(X_h2h.shape))
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h)  
        X_l2h = self.l2h(X_l)  
        
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_h2h + X_l2h  
        return X_h  


class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()
        
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x0 = x
        x_h, x_l = self.fir(x)  # (1,64,64,64) ,(1,64,32,32)
        x_hh, x_ll = x_h, x_l,
        # x_1 = x_hh +x_ll
        x_h_1, x_l_1 = self.mid1((x_h, x_l))  # (1,64,64,64) ,(1,64,32,32)
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1))  # (1,64,64,64) ,(1,64,32,32)
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2))  # (1,32,64,64) ,(1,32,32,32)
        x_ret = self.lst((x_h_5, x_l_5))  # (1,64,64,64)
        return x_h_5, x_l_5

        # x_l_11 = F.interpolate(x_l_1, (int(x_h_1.size()[2]), int(x_h_1.size()[3])), mode='bilinear')
        # x_ret, x_h_6, x_l_6 = self.lst((x_h_5, x_l_5)) # (1,64,64,64)
        # return x0, x_ret,x_hh, x_ll,x_h_1, x_l_1

        # return x0, x_ret, x_hh, x_ll, x_h_6, x_l_6
        # return x0, x_ret
    # fea_name = ['_before','_after', '_beforeH', '_beforeL', '_afterH', '_afterL', '_afterH0', '_afterL0']

class Octave_middle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)) -> None:
        super(Octave_middle, self).__init__()
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)
    def forward(self, x_h, x_l):
        x_h_1, x_l_1 = self.mid1((x_h, x_l))  # (1,64,64,64) ,(1,64,32,32)
        x_h_2, x_l_2 = self.mid2((x_h_1, x_l_1))  # (1,64,64,64) ,(1,64,32,32)
        return x_h_2, x_l_2
        


class Distraction_Supression(nn.Module):
    def __init__(self, in_channels, out_channels, backbone, recurrence=1) -> None:
        super(Distraction_Supression, self).__init__()
        self.octave = Octave(in_channels, out_channels)
        self.lf_cc_attention = CrissCrossAttention_LowFreq(out_channels // 2)
        self.hf_cc_attention = CrissCrossAttention_HighFreq(out_channels // 2)
        self.recurrence = recurrence
        backbone_dict = {'ResNet50': 1024, 'PVT': 512}
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(backbone_dict.get(backbone), out_channels // 2, 1, 1, 0),
            nn.BatchNorm2d(out_channels // 2), nn.ReLU())

    def forward(self, f, ORI):
        ORI = self.channel_reduction(ORI)

        f_H, f_L = self.octave(f)
        ORI_H = F.interpolate(ORI, (int(f_H.size()[2]), int(f_H.size()[3])), mode='bilinear')
        ORI_L = F.interpolate(ORI, (int(f_L.size()[2]), int(f_L.size()[3])), mode='bilinear')
        for i in range(self.recurrence):
            f_H = self.hf_cc_attention(ORI_H, f_H)
            f_L = self.lf_cc_attention(ORI_L, f_L)
        f_Hds = f_H
        f_Lds = f_L
        return f_Hds, f_Lds

class Distraction_Supression_middle(nn.Module):
    def __init__(self, in_channels, out_channels, backbone, recurrence=1) -> None:
        super(Distraction_Supression_middle, self).__init__()
        self.octave = Octave_middle(in_channels, out_channels)
        self.lf_cc_attention = CrissCrossAttention_LowFreq(out_channels // 2)
        self.hf_cc_attention = CrissCrossAttention_HighFreq(out_channels // 2)
        self.recurrence = recurrence
        backbone_dict = {'ResNet50': 1024, 'PVT': 512}
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(backbone_dict.get(backbone), out_channels // 2, 1, 1, 0),
            nn.BatchNorm2d(out_channels // 2), nn.ReLU())
    def forward(self, f_h, f_l, ORI):
        ORI = self.channel_reduction(ORI)
        f_h, h_l = self.octave(f_h, f_l)
        ORI_H = F.interpolate(ORI, (int(f_h.size()[2]), int(f_h.size()[3])), mode='bilinear')
        ORI_L = F.interpolate(ORI, (int(f_l.size()[2]), int(f_l.size()[3])), mode='bilinear')
        for i in range(self.recurrence):
            f_h = self.hf_cc_attention(ORI_H, f_h)
            f_l = self.lf_cc_attention(ORI_L, f_l)
        f_Hds = f_h
        f_Lds = f_l
        return f_Hds, f_Lds

class iterativeDS(nn.Module):
    def __init__(self, in_channels, out_channels, backbone, num_iter=1, use_residual=False, recurrence=1) -> None:
        super(iterativeDS, self).__init__()
        self.num_iter = num_iter
        self.ds1 = Distraction_Supression(in_channels, out_channels, backbone, recurrence)
        self.ds_list = None
        self.use_residual = use_residual
        if num_iter > 1:
            self.ds_list = [Distraction_Supression_middle(out_channels, out_channels, backbone, recurrence) for _ in range(num_iter - 1)]
            self.ds_list = nn.ModuleList(self.ds_list)
        else:
            self.ds_list = None
        # self.ds_final = nn.Sequential(self.ds1, *self.ds_list)
    
    def forward(self, f, ORI):
        f_Hds, f_Lds = self.ds1(f, ORI)
        # if self.ds_list is not None:
        #     for ds in self.ds_list:
        #         f_Hds, f_Lds = ds(f_Hds, f_Lds, ORI)

        if self.ds_list is not None:
            for ds in self.ds_list:
                if self.use_residual:
                
                    f_Hds_prev = f_Hds
                    f_Lds_prev = f_Lds
                
                f_Hds, f_Lds = ds(f_Hds, f_Lds, ORI)
                
                if self.use_residual:
                    f_Hds = f_Hds + f_Hds_prev
                    f_Lds = f_Lds + f_Lds_prev
        
        return f_Hds, f_Lds
                        

        


if __name__ == '__main__':
    # model = CrissCrossAttention_LowFreq(64).to('cuda:0')
    # x = torch.randn(2, 64, 5, 6).to('cuda:0')
    # y = torch.randn(2, 64, 5, 6).to('cuda:0')
    # out = model(x, y)
    # print(out.shape)

    # x = torch.randn(2, 320, 24, 24).to('cuda:0')
    # octave = Octave(320, 640).to('cuda:0')
    # out = octave(x)
    # print(out[1].shape)

    # model = Distraction_Supression(320, 640).to('cuda:0')
    # ori = torch.randn(2, 512, 12, 12).to('cuda:0')
    # f = torch.randn(2, 320, 24, 24).to('cuda:0')
    # out = model(f, ori)
    # print(out[0].shape)

    imodel = iterativeDS(320, 640, num_iter=3, backbone='PVT', use_residual=False).to('cuda:0')
    ori = torch.randn(2, 512, 12, 12).to('cuda:0')
    f = torch.randn(2, 320, 24, 24).to('cuda:0')
    out = imodel(f, ori)
    print(out[0].shape, out[1].shape)
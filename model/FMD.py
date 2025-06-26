import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


class CRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CRB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x
    
class CBR(nn.Module):
    def __init__(self, num_channels):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AdaptiveNormalization(nn.Module):
    def __init__(self, in_channels_high, in_channels_low):
        super(AdaptiveNormalization, self).__init__()
        self.crb_sigma = CBR(in_channels_low)
        self.crb_mu = CBR(in_channels_low)
        
        self.conv3_sigma = nn.Conv2d(in_channels_high, in_channels_high, kernel_size=3, padding=1)
        self.conv3_mu = nn.Conv2d(in_channels_high, in_channels_high, kernel_size=3, padding=1)

    def forward(self, fi_high, fi_low):
        sigma_k = self.conv3_sigma(self.crb_sigma(fi_low))
        mu_k = self.conv3_mu(self.crb_mu(fi_low))

        sigma_k = F.interpolate(sigma_k, size=fi_high.shape[2:], mode='bilinear', align_corners=False)
        mu_k = F.interpolate(mu_k, size=fi_high.shape[2:], mode='bilinear', align_corners=False)

        modulated_fi_high = sigma_k * fi_high + mu_k
        return modulated_fi_high
    

class iAFF(nn.Module):
  

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MS_CAM(nn.Module):


    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei
    

class iAFF_FMD(nn.Module):


    def __init__(self, channels=64, r=4):
        super(iAFF_FMD, self).__init__()
        inter_channels = int(channels // r)

        
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, fi_H, fi_L):
        fi_L = F.interpolate(fi_L, (int(fi_H.size()[2]), int(fi_H.size()[3])), mode='bilinear')
        xa = fi_H + fi_L
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = fi_H * wei + fi_L * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        Fi_M = fi_H * wei2 + fi_L * (1 - wei2)
        return Fi_M

class FMD(nn.Module):
    def __init__(self, in_channels) -> None:
        super(FMD, self).__init__()
        self.CBR1 = CBR(in_channels)
        self.CBR2 = CBR(in_channels)
        self.AN = AdaptiveNormalization(in_channels, in_channels)
        self.iAFF = iAFF_FMD(in_channels)
    
    def forward(self, f_Hds, f_Lds, x_mask):
        x_mask = torch.sigmoid(x_mask)
        x_mask_H = F.interpolate(x_mask, (int(f_Hds.size()[2]), int(f_Hds.size()[3])), mode='bilinear')
        x_mask_L = F.interpolate(x_mask, (int(f_Lds.size()[2]), int(f_Lds.size()[3])), mode='bilinear')
        mask_FH = self.CBR1(f_Hds + f_Hds * x_mask_H)
        mask_FL = self.CBR2(f_Lds + f_Lds * x_mask_L)
    
        mask_FH_mdt = self.AN(mask_FH, mask_FL)
        Fi_M = self.iAFF(mask_FH_mdt, mask_FL)
        return Fi_M
    
class FMD_woiAFF(nn.Module):
    def __init__(self, in_channels) -> None:
        super(FMD_woiAFF, self).__init__()
        self.CBR1 = CBR(in_channels)
        self.CBR2 = CBR(in_channels)
        self.AN = AdaptiveNormalization(in_channels, in_channels)
    
    def forward(self, f_Hds, f_Lds, x_mask):
        x_mask = torch.sigmoid(x_mask)
        x_mask_H = F.interpolate(x_mask, (int(f_Hds.size()[2]), int(f_Hds.size()[3])), mode='bilinear')
        x_mask_L = F.interpolate(x_mask, (int(f_Lds.size()[2]), int(f_Lds.size()[3])), mode='bilinear')
        mask_FH = self.CBR1(f_Hds + f_Hds * x_mask_H)
        mask_FL = self.CBR2(f_Lds + f_Lds * x_mask_L)
        mask_FH_mdt = self.AN(mask_FH, mask_FL)

        return mask_FH_mdt

if __name__ == '__main__':
    # model = AdaptiveNormalization(320, 320).to('cuda:0')
    # fi_high = torch.randn(2, 320, 24, 24).to('cuda:0')
    # fi_low = torch.randn(2, 320, 12, 12).to('cuda:0')
    # out = model(fi_high, fi_low)
    # print(out.shape)

    # model = iAFF_FMD(320).to('cuda:0')
    # fi_high = torch.randn(2, 320, 24, 24).to('cuda:0')
    # fi_low = torch.randn(2, 320, 12, 12).to('cuda:0')
    # out = model(fi_high, fi_low)
    # print(out.shape)
    
    model = FMD(320).to('cuda:0')
    fi_high = torch.randn(2, 320, 24, 24).to('cuda:0')
    fi_low = torch.randn(2, 320, 12, 12).to('cuda:0')
    x_mask = torch.randn(2, 1, 224, 224).to('cuda:0')
    out = model(fi_high, fi_low, x_mask)
    print(out.shape)
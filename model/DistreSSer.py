import torch
import torch.nn as nn
import torch.nn.functional as F

# from DSM import Distraction_Supression

from model.DSM import Distraction_Supression, iterativeDS
from model.object_mining import Object_Mining
from model.encoder.pvtv2_encoder import pvt_v2_b4
from model.FMD import FMD

# from DSM import Distraction_Supression, iterativeDS
# from object_mining import Object_Mining
# from encoder.pvtv2_encoder import pvt_v2_b4
# from FMD import FMD

import timm

from thop import profile




class CBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConcatAndPred(nn.Module):
    def __init__(self, C1, C2) -> None:
        super(ConcatAndPred, self).__init__()
        self.conv1 = nn.Conv2d(C1, C2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C2, 1, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(C2*2, C2, kernel_size=3, padding=1)
        self.cbr = CBR(C2 * 2, C2)

    def forward(self, f2, f1):
        f1 = F.interpolate(f1, (int(f2.size()[2]), int(f2.size()[3])), mode='bilinear')
        N2, C2, H2, W2 = f2.shape
        N1, C1, H1, W1 = f1.shape
        # conv1 = nn.Conv2d(C1, C2, kernel_size=3, padding=1).to('cuda:0')
        # conv2 = nn.Conv2d(C2*2, 1, kernel_size=3, padding=1).to('cuda:0')
        f1 = self.conv1(f1)
        F2_M = torch.cat((f2, f1), dim=1)
        F2_M = self.cbr(F2_M)
        p = self.conv2(F2_M)
        return F2_M, p


class DSNet(nn.Module):
    def __init__(self, backbone='ResNet50', num_iter=1, pvt_load_path=None, use_residual=False, recurrence=1) -> None:
        super(DSNet, self).__init__()
        self.backbone = backbone
        self.use_residual = use_residual
        if backbone == 'PVT':
            self.encoder = pvt_v2_b4()
        else:
            self.encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        if backbone == 'PVT': 
            if pvt_load_path is not None:
                pretrained_dict = torch.load(pvt_load_path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
                self.encoder.load_state_dict(pretrained_dict)
                print('Pretrained encoder loaded.')

        # self.DSM1 = Distraction_Supression(64, 64 * 2)
        # self.DSM2 = Distraction_Supression(128, 128 * 2)
        # self.DSM3 = Distraction_Supression(320, 320 * 2)
        channel_numbers_pvt = [64, 128, 320, 512]
        channel_numbers_res = [64, 256, 512, 1024, 2048]

        # self.DSM1 = iterativeDS(64, 64 * 2, backbone, num_iter)
        # self.DSM2 = iterativeDS(128, 128 * 2, backbone, num_iter)
        # self.DSM3 = iterativeDS(320, 320 * 2, backbone, num_iter)

        # self.h_cbr1 = CBR(64, 64)
        # self.h_cbr2 = CBR(128, 128)
        # self.h_cbr3 = CBR(320, 320)

        # self.l_cbr1 = CBR(64, 64)
        # self.l_cbr2 = CBR(128, 128)
        # self.l_cbr3 = CBR(320, 320)

        # self.cbr4 = CBR(512, 512)

        # self.OMM = Object_Mining(512)

        # self.FMD1 = FMD(64)
        # self.FMD2 = FMD(128)
        # self.FMD3 = FMD(320)

        # self.cap4 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        # self.cap3 = ConcatAndPred(512, 320)
        # self.cap2 = ConcatAndPred(320, 128)
        # self.cap1 = ConcatAndPred(128, 64)

        # if backbone == 'ResNet50':
        #     self.channel_reduction1 = nn.Sequential(
        #     nn.Conv2d(channel_numbers_res[0], channel_numbers_pvt[0], 1, 1, 0),
        #     nn.BatchNorm2d(channel_numbers_pvt[0]), nn.ReLU())

        #     self.channel_reduction2 = nn.Sequential(
        #     nn.Conv2d(channel_numbers_res[1], channel_numbers_pvt[1], 1, 1, 0),
        #     nn.BatchNorm2d(channel_numbers_pvt[1]), nn.ReLU())

        #     self.channel_reduction3 = nn.Sequential(
        #     nn.Conv2d(channel_numbers_res[2], channel_numbers_pvt[2], 1, 1, 0),
        #     nn.BatchNorm2d(channel_numbers_pvt[2]), nn.ReLU())

        #     self.channel_reduction4 = nn.Sequential(
        #     nn.Conv2d(channel_numbers_res[3], channel_numbers_pvt[3], 1, 1, 0),
        #     nn.BatchNorm2d(channel_numbers_pvt[3]), nn.ReLU())

        if self.backbone == 'PVT':
            self.DSM1 = iterativeDS(64, 64 * 2, backbone, num_iter, self.use_residual, recurrence)
            self.DSM2 = iterativeDS(128, 128 * 2, backbone, num_iter, self.use_residual, recurrence)
            self.DSM3 = iterativeDS(320, 320 * 2, backbone, num_iter, self.use_residual, recurrence)

            self.h_cbr1 = CBR(64, 64)
            self.h_cbr2 = CBR(128, 128)
            self.h_cbr3 = CBR(320, 320)

            self.l_cbr1 = CBR(64, 64)
            self.l_cbr2 = CBR(128, 128)
            self.l_cbr3 = CBR(320, 320)

            self.cbr4 = CBR(512, 512)

            self.OMM = Object_Mining(512)

            self.FMD1 = FMD(64)
            self.FMD2 = FMD(128)
            self.FMD3 = FMD(320)

            self.cap4 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
            self.cap3 = ConcatAndPred(512, 320)
            self.cap2 = ConcatAndPred(320, 128)
            self.cap1 = ConcatAndPred(128, 64)
        else:
            self.DSM1 = iterativeDS(channel_numbers_res[0], channel_numbers_res[0] * 2, backbone, num_iter, self.use_residual, recurrence)
            self.DSM2 = iterativeDS(channel_numbers_res[1], channel_numbers_res[1] * 2, backbone, num_iter, self.use_residual, recurrence)
            self.DSM3 = iterativeDS(channel_numbers_res[2], channel_numbers_res[2] * 2, backbone, num_iter, self.use_residual, recurrence)

            self.h_cbr1 = CBR(channel_numbers_res[0], channel_numbers_res[0])
            self.h_cbr2 = CBR(channel_numbers_res[1], channel_numbers_res[1])
            self.h_cbr3 = CBR(channel_numbers_res[2], channel_numbers_res[2])

            self.l_cbr1 = CBR(channel_numbers_res[0], channel_numbers_res[0])
            self.l_cbr2 = CBR(channel_numbers_res[1], channel_numbers_res[1])
            self.l_cbr3 = CBR(channel_numbers_res[2], channel_numbers_res[2])

            self.cbr4 = CBR(channel_numbers_res[3], channel_numbers_res[3])

            self.OMM = Object_Mining(channel_numbers_res[3])

            self.FMD1 = FMD(channel_numbers_res[0])
            self.FMD2 = FMD(channel_numbers_res[1])
            self.FMD3 = FMD(channel_numbers_res[2])

            self.cap4 = nn.Conv2d(channel_numbers_res[3], 1, kernel_size=3, padding=1)
            self.cap3 = ConcatAndPred(channel_numbers_res[3], channel_numbers_res[2])
            self.cap2 = ConcatAndPred(channel_numbers_res[2], channel_numbers_res[1])
            self.cap1 = ConcatAndPred(channel_numbers_res[1], channel_numbers_res[0])

            # self.channel_reduction4 = nn.Sequential(
            # nn.Conv2d(channel_numbers_res[3], channel_numbers_pvt[3], 1, 1, 0),
            # nn.BatchNorm2d(channel_numbers_pvt[3]), nn.ReLU())

            # self.channel_reduction3 = nn.Sequential(
            # nn.Conv2d(channel_numbers_res[3], 256, 1, 1, 0),
            # nn.BatchNorm2d(256), nn.ReLU())

        

    def forward(self, x):
        outs = self.encoder(x)
        if self.backbone == 'PVT':
            f1 = outs[3] # 64, 96, 96
            f2 = outs[2] # 128, 48, 48
            f3 = outs[1] # 320, 24, 24
            f4 = outs[0] # 512, 12, 12
        else: 
            # f1 = outs[1] # 256,96，96
            # f2 = outs[2] # 512,48, 48
            # f3 = outs[3] # 1024，24，24
            # f4 = outs[4] # 2048, 12, 12

            # f1 = self.channel_reduction1(f1)
            # f2 = self.channel_reduction2(f2)
            # f3 = self.channel_reduction3(f3)
            # f4 = self.channel_reduction4(f4)

            f1 = outs[0] # 64,192，192
            f2 = outs[1] # 256,96, 96
            f3 = outs[2] # 512，48，48
            f4 = outs[3] # 1024, 24, 24

        

        ori = self.OMM(f4)


        f3_Hds, f3_Lds = self.DSM3(f3, ori)
        f2_Hds, f2_Lds = self.DSM2(f2, ori)
        f1_Hds, f1_Lds = self.DSM1(f1, ori)

        f3_Hds = self.h_cbr3(f3_Hds)
        f2_Hds = self.h_cbr2(f2_Hds)
        f1_Hds = self.h_cbr1(f1_Hds)

        f3_Lds = self.h_cbr3(f3_Lds)
        f2_Lds = self.h_cbr2(f2_Lds)
        f1_Lds = self.h_cbr1(f1_Lds)

        f4 = self.cbr4(f4)


        p4 = self.cap4(f4)
        F3_M = self.FMD3(f3_Hds, f3_Lds, p4)
        F3_M, p3 = self.cap3(F3_M, f4)

        F2_M = self.FMD2(f2_Hds, f2_Lds, p3)
        F2_M, p2 = self.cap2(F2_M, F3_M)
        F1_M = self.FMD1(f1_Hds, f1_Lds, p2)
        F1_M, p1 = self.cap1(F1_M, F2_M)

        p4 = F.interpolate(p4, size=x.size()[2:], mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, size=x.size()[2:], mode='bilinear', align_corners=True)
        p2 = F.interpolate(p2, size=x.size()[2:], mode='bilinear', align_corners=True)
        p1 = F.interpolate(p1, size=x.size()[2:], mode='bilinear', align_corners=True)
        return p1, p2, p3, p4
        # return torch.sigmoid(p1), torch.sigmoid(p2), torch.sigmoid(p3), torch.sigmoid(p4)

    def forward_test(self, x):
        outs = self.encoder(x)
        for out in outs:
            print(out.shape)


if __name__ == '__main__':
    model = DSNet(num_iter=3, backbone='ResNet50', pvt_load_path=None).to('cuda:0')
    input = torch.randn(1, 3, 384, 384).to('cuda:0')
    # model.forward_test(input)

    # p1, p2, p3, p4 = model(input)
    # print(p1.shape, p2.shape, p3.shape, p4.shape)

    flops, params = profile(model, inputs=(input, ))
    print('FLOPs = ' + str((flops/1000**3)/2) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

    Trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {Trainable_params/ 1e6}M')





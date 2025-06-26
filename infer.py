import gc
import os

from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
# from model import DSNet, distraction_supression
import sys
from model.DSM import Distraction_Supression
# from model.DSNet import DSNet
from model.DistreSSer import DSNet
from utils.data_val import get_loader
from utils.utils import clip_gradient, adjust_lr
import logging
import argparse
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import time
import datetime
from tqdm import tqdm


def main(results_root, img_transform, test_root, dict_path, datasets, device_ids):
    net = DSNet(num_iter=3, backbone='PVT', use_residual=False, recurrence=1)
    state_dict = torch.load(dict_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  
        new_state_dict[name] = v  
    # load params
    net.load_state_dict(new_state_dict, strict=True) 
    net.cuda()
    net.eval()
    # print(state_dict)
    # net.load_state_dict(torch.load("snapshot/Net_epoch_200.pth"))
    # print('Load {} succeed!'.format('Net_epoch_200.pth'))
    # net.eval()
    for dataset in datasets:
         results_path = os.path.join(results_root, dataset)
         if not os.path.exists(results_path): 
            os.makedirs(results_path)
         test_path = os.path.join(test_root, dataset)
         with torch.no_grad():
            image_path = os.path.join(test_path, 'Image')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            # print(img_list)
            for idx, img_name in enumerate(tqdm(img_list, desc="Processing Images")):
                img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])
                # print(img_var.shape)
                # start_each = time.time()
                prediction, _, _, _= net(img_var)
                #热力图
                # Image.fromarray(f1)
                prediction = torch.sigmoid(prediction)
                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
                Image.fromarray(prediction).convert('L').save(os.path.join(results_path, img_name + '.png'))
                # print("{} saved!".format(img_name))

if __name__ == '__main__':
    results_root = 'res_pvt_retrain_24/'
    device_ids = [0]
    img_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    to_pil = transforms.ToPILImage()
    test_root = "dataset/TestDataset/"
    datasets = ['COD10K', 'NC4K', 'CHAMELEON', 'CAMO']
    # datasets = ['CAMO']

    dict_path = 'snapshot/pvt_val_retrain24/Net_epoch_10_val_mae_0.023.pth'
    main(results_root, img_transform, test_root, dict_path, datasets, device_ids)
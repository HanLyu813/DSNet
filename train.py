import gc
import os
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
# from tensorboardX import SummaryWriter
from tensorboardX import SummaryWriter
import loss
import torch.optim.lr_scheduler as lr_scheduler
from utils.data_val import test_dataset
from torch.optim.lr_scheduler import LambdaLR
from evaltools import metrics
# def structure_loss(pred, mask):
#     """
#     loss function (ref: F3Net-AAAI-2020)
#     """
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     # wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
#
#     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#     # 计算IoU损失
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask) * weit).sum(dim=(2, 3)) # 交
#     union = ((pred + mask) * weit).sum(dim=(2, 3)) # 和
#     wiou = 1 - (inter + 1) / (union - inter + 1)
#     return (wbce + wiou).mean()

bce_loss = nn.BCEWithLogitsLoss()
structure_loss = loss.structure_loss()
iou_loss = loss.IOU()


def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss


def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def poly_decay_scheduler(optimizer, num_epochs, power=0.9, warmup_epochs=10):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # Linear Warmup
        else:
            return (1 - (epoch - warmup_epochs) / (num_epochs - warmup_epochs)) ** power

    return LambdaLR(optimizer, lr_lambda)


def train(train_loader, model, optimizer, epoch, save_path, writer, hyy, loss_list):
    """
    train function
    """
    global step  
    model.train()
    loss_all = 0  
    epoch_step = 0  
    try:
    
        for i, (images, gts) in enumerate(train_loader, start=1):  
            optimizer.zero_grad()
            if hyy == 'cloud':
                images = images.cuda()
                gts = gts.cuda()
            else:
                images = images.to('cuda:0')
                gts = gts.to('cuda:0')
            images = images
            gts = gts
            p1, p2, p3, p4 = model(images) 

            loss_4 = bce_iou_loss(p4, gts)
            loss_3 = structure_loss(p3, gts)
            loss_2 = structure_loss(p2, gts)
            loss_1 = structure_loss(p1, gts)
            loss = loss_4 + 1 * loss_3 + 2 * loss_2 + 4 * loss_1
            # loss_init = structure_loss(preds[0], gts) + structure_loss(preds[1], gts)
            # loss_final = structure_loss(preds[2], gts)
            # loss = loss_init + loss_final

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}  '
                    .
                    format(epoch, opt.epoch, i, total_step, loss.data))

        loss_all /= epoch_step
        loss_list.append(loss_all)
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 5 == 0:
            # torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
            torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_{}.pth'.format(epoch)))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_{}.pth'.format(epoch)))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer, hyy, mae_list, opt):
    """
    validation function
    """
    global best_mae, best_epoch, best_Score
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        WFM = metrics.WeightedFmeasure()
        SM = metrics.Smeasure()
        EM = metrics.Emeasure()
        MAE = metrics.MAE()
        if opt.val_methods == 'mae':
            for i in range(test_loader.size):
                image, gt, name, img_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                if hyy == 'cloud':
                    image = image.cuda()
                # image = image
                res = model(image)

                res = F.interpolate(res[0], size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()  
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            mae = mae_sum / test_loader.size  
            mae_list.append(mae)
            writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
            print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
            if epoch == 1:
                best_mae = mae
            else:
                if mae < best_mae:
                    best_mae = mae
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_{}_val_mae_{}.pth'.format(epoch, round(best_mae, 3))))
                    print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
            logging.info(
                '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
        elif opt.val_methods == 'metrics':
            for i in range(test_loader.size):
                image, gt, name, img_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                if hyy == 'cloud':
                    image = image.cuda()
                # image = image
                res = model(image)

                res = F.interpolate(res[0], size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()  # 352*352
                WFM.step(pred=res, gt=gt)
                SM.step(pred=res, gt=gt)
                EM.step(pred=res, gt=gt)
                MAE.step(pred=res, gt=gt)

            em = EM.get_results()['em']
            mae = MAE.get_results()['mae']
            wfm = WFM.get_results()['wfm']
            sm = SM.get_results()['sm']

            Smeasure_r = sm.round(3)
            Wmeasure_r = wfm.round(3)
            adpEm_r = em['adp'].round(3)
            MAE_r = mae.round(3)

            avg_score = (Smeasure_r + Wmeasure_r + adpEm_r + 1 - MAE_r) / 4
            print('Epoch: {}, S: {}, WF: {}, adE: {}, MAE: {}'.format(epoch, str(Smeasure_r), str(Wmeasure_r), str(adpEm_r), str(MAE_r)))

            print('Epoch: {}, score: {}, bestScore: {}, bestEpoch: {}.'.format(epoch, avg_score, best_Score, best_epoch))

            if epoch == 1:
                best_Score = avg_score
            else:
                if avg_score > best_Score:
                    best_Score = avg_score
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_{}_val_score_{}.pth'.format(epoch, round(best_Score, 3))))
                    print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
            logging.info(
                '[Score Info]:Epoch:{} SCORE:{} bestEpoch:{} bestSCORE:{}'.format(epoch, avg_score, best_epoch, best_Score))
            


if __name__ == '__main__':
    # torch.backends.cudnn.enabled=False
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    platform = "cloud"
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--load_pretrained', type=str, default="pretrained/pvt_v2_b4.pth", help='pvtv2 pretrained files')
    if platform == 'cloud':
        parser.add_argument('--batchsize', type=int, default=24, help='training batch size')
        parser.add_argument('--train_root', type=str, default='dataset/TrainDataset/',
                            help='the training rgb images root')
        parser.add_argument('--val_root', type=str, default='dataset/TestDataset/COD10K/',
                            help='the test rgb images root')
        parser.add_argument('--save_path', type=str,
                            default='snapshot/',
                            help='the path to save model and log')
    else:
        parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
        parser.add_argument('--train_root', type=str, default="dataset/COD10K-v3/Test/",
                            help='the training rgb images root')
        parser.add_argument('--val_root', type=str,
                            default='E:\\SHSF\\DL\\database\\test_database\\CHAMELEON_TestingDataset\\',
                            help='the test rgb images root')
        parser.add_argument('--save_path', type=str,
                            default='snapshot/',
                            help='the path to save model and log')
    parser.add_argument('--save_path_val', type=str, default='pvt_val_retrain', help='save path for val')
    parser.add_argument('--save_path_train', type=str, default='pvt_train_retrain', help='save path for train')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')  # 352
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate of learning rate')
    # parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='4', help='train use gpu')
    parser.add_argument('--iter', type=int, default=3, help='the iteration of DSM')
    parser.add_argument('--lr_scheduler', type=str, default='Exponential', help='learning rate update strategy')
    parser.add_argument('--backbone', type=str, default='PVT')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--use_residual', type=bool, default=False)
    parser.add_argument('--recurrence', type=int, default=1)
    parser.add_argument('--adam_weight_decay', type=float, default=0)
    parser.add_argument('--val_methods', type=str, default='mae')


    opt = parser.parse_args()
    print("batchsize:{}".format(opt.batchsize))
    opt.iter = 3
    print("build model")
    model = None
    # opt.load_pretrained = "pretrained/pvt_v2_b4.pth"
    # if opt.load_pretrained is not None:
    #     model = DSNet(opt.iter, pvt_load_path==opt.load_pretrained)
    # if platform == 'cloud':
    #     model = DSNet()
    #     model = nn.DataParallel(model)  # Wrap the model with DataParallel for multi-GPU
    #     model = model.cuda()
    # else:
    #     model = DSNet().to('cuda:0')
    model = DSNet(num_iter=opt.iter, pvt_load_path=opt.load_pretrained, backbone=opt.backbone, use_residual=opt.use_residual, recurrence=opt.recurrence)
    model = nn.DataParallel(model)  # Wrap the model with DataParallel for multi-
    model = model.cuda()
    print("finish model")

    if opt.load is not None:  
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    # optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=1e-4)  
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam([
            {"params": model.module.encoder.parameters(), "lr": opt.lr * 0.1},  # Example: 0.1 * base_lr for encoder
            {"params": [p for n, p in model.module.named_parameters() if 'encoder' not in n], "lr": opt.lr}
        ], weight_decay = opt.adam_weight_decay)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD([
            {"params": model.module.encoder.parameters(), "lr": opt.lr * 0.1},  # Example: 0.1 * base_lr for encoder
            {"params": [p for n, p in model.module.named_parameters() if 'encoder' not in n], "lr": opt.lr}
        ], momentum=0.9, weight_decay=0.0005)
    print('lr: ', opt.lr)

    if opt.lr_scheduler == 'Exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=opt.decay_rate)
        print("Exponential scheduler all set")
    elif opt.lr_scheduler == 'CosineAnnealing':
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=1e-7)
        print("Cosine scheduler all set")
    elif opt.lr_scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        print("StepLR scheduler all set")
    elif opt.lr_scheduler == 'poly':
        scheduler = poly_decay_scheduler(optimizer, num_epochs=opt.epoch, power=0.9, warmup_epochs=10)
        print("poly decay scheduler all set")


    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 二、load data---------------------------------------------------------------------------
    print('load data...')
    train_loader = None
    if platform == 'cloud':
        train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                                  gt_root=opt.train_root + 'GT/',
                                  batchsize=opt.batchsize,
                                  trainsize=opt.trainsize,
                                  num_workers=0)  # 14
    else:
        train_loader = get_loader(image_root=opt.train_root + 'Image/',
                                  gt_root=opt.train_root + 'GT_Object/',
                                  batchsize=opt.batchsize,
                                  trainsize=opt.trainsize,
                                  num_workers=0)  # 14

    val_loader = test_dataset(image_root=opt.val_root + 'Image/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)
    logging.basicConfig(filename=save_path + 'pvt_retrain.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info("lr=0.0001,size:512; logname:mylog.log; witer:summary")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))
    step = 0
    writer = SummaryWriter(save_path + 'summary') 
    best_mae = 1
    best_Score = 0
    best_epoch = 0

    print("Start train...")
    gc.collect()
    torch.cuda.empty_cache()
    loss_list = []
    mae_list = []

    if not os.path.exists(os.path.join(save_path, opt.save_path_train)):
        os.makedirs(os.path.join(save_path, opt.save_path_train))  
        print(f"Directory '{os.path.join(save_path, opt.save_path_train)}' created.")
    if not os.path.exists(os.path.join(save_path, opt.save_path_val)):
        os.makedirs(os.path.join(save_path, opt.save_path_val))  
        print(f"Directory '{os.path.join(save_path, opt.save_path_val)}' created.")

    print("save_path:{}".format(os.path.join(save_path, opt.save_path_train)))
    print("train_root:{}".format(opt.train_root))
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, os.path.join(save_path, opt.save_path_train), writer, platform, loss_list)
    
        val(val_loader, model, epoch, os.path.join(save_path, opt.save_path_val), writer, platform,mae_list, opt)
        logging.info('>>> current lr: {}'.format(scheduler.get_lr()[0]))
        print('current lr: {}'.format(scheduler.get_lr()[0]))
        scheduler.step()
    # cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    
        # writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
    writer.close()

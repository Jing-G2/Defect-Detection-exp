import os
import argparse
import shutil
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter
from utils.data_util import load_data


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name',
                        default='',
                        type=str,
                        help='model name')
    parser.add_argument('--data_name', default='', type=str, help='datas name')
    parser.add_argument('--in_channels',
                        default='',
                        type=int,
                        help='in channels')
    parser.add_argument('--num_classes',
                        default='',
                        type=int,
                        help='num classes')
    parser.add_argument('--model_dir', default='', type=str, help='model dir')
    parser.add_argument('--data_dir', default='', type=str, help='datas dir')
    parser.add_argument('--log_dir', default='', type=str, help='log dir')
    parser.add_argument('--device_index',
                        default='0',
                        type=str,
                        help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16

    print('-' * 50)
    print('DEVICE:', device)
    print('BATCH SIZE:', batch_size)
    print('MODEL DIR:', args.model_dir)
    print('LOG DIR:', args.log_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    train_loader, val_loader, test_loader = load_data(args.data_dir,
                                                      args.data_name,
                                                      batch_size, 'gc')

    model = torch.load(os.path.join(args.model_dir, 'model_ori.pth'))
    model.to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    loss, acc1, auc_ap, class_acc = test(model, test_loader, criterion,
                                         device)
    macc = class_acc().mean()
    auc, ap = auc_ap(args.num_classes == 2)

    torch.save(model, os.path.join(args.model_dir, 'model_ori.pth'))
    print('COMPLETE !!!')
    print('TEST mACC', macc)
    print('TEST ACC', acc1)
    print(class_acc)
    print('TEST AP', ap)
    print('TEST AUC', auc)


def test(model, loader, criterion, device):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(total=len(loader),
                             step=20,
                             prefix='Test',
                             meters=[loss_meter])
    class_acc = metrics.ClassAccuracy()
    auc_ap = metrics.AUC_AP()

    model.eval()
    for i, data in enumerate(loader):
        data = data.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(data.x, data.edge_index, data.batch)
        loss = criterion(outputs, data.y)
        acc1 = metrics.accuracy(outputs, data.y)[0]

        class_acc.update(outputs, data.y)
        auc_ap.update(outputs, data.y)
        loss_meter.update(loss.item(), len(data.y))
        acc1_meter.update(acc1.item(), len(data.y))
        progress.display(i)

    return loss_meter.avg, acc1_meter.avg, auc_ap, class_acc


if __name__ == '__main__':
    main()

import os
import argparse
import shutil
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter
from utils.loader import load_data


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
    parser.add_argument('--num_epochs',
                        default=200,
                        type=int,
                        help='num epochs')
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

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir, ignore_errors=True)

    batch_size = 16

    print('-' * 50)
    print('DEVICE:', device)
    print('EPOCH:', args.num_epochs)
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

    model = models.load_model(args.model_name, args.in_channels,
                              args.num_classes, 'gc')
    model.to(device)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.num_epochs)

    writer = SummaryWriter(args.log_dir)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    best_model = None
    best_macc = None
    best_acc = None
    best_epoch = None

    for epoch in tqdm(range(args.num_epochs)):
        loss, acc1, auc_ap, class_acc = train(model, train_loader, criterion,
                                              optimizer, device)
        auc, ap = auc_ap(args.num_classes == 2)
        macc = class_acc().mean()
        writer.add_scalar(tag='train loss',
                          scalar_value=loss,
                          global_step=epoch)
        # writer.add_scalar(tag='train auc', scalar_value=auc, global_step=epoch)
        # writer.add_scalar(tag='train ap', scalar_value=ap, global_step=epoch)
        writer.add_scalar(tag='train acc',
                          scalar_value=acc1,
                          global_step=epoch)
        writer.add_scalar(tag='train macc',
                          scalar_value=macc,
                          global_step=epoch)

        loss, acc1, auc_ap, class_acc = test(model, val_loader, criterion,
                                             device)
        auc, ap = auc_ap(args.num_classes == 2)
        macc = class_acc().mean()
        writer.add_scalar(tag='val loss',
                          scalar_value=loss,
                          global_step=epoch)
        # writer.add_scalar(tag='val auc', scalar_value=auc, global_step=epoch)
        # writer.add_scalar(tag='val ap', scalar_value=ap, global_step=epoch)
        writer.add_scalar(tag='val acc', scalar_value=acc1, global_step=epoch)
        writer.add_scalar(tag='val macc',
                          scalar_value=macc,
                          global_step=epoch)

        # ----------------------------------------
        # save best model
        # ----------------------------------------
        if best_acc is None or best_acc <= acc1:
            # if best_ap is None or best_ap <= ap:
            best_model = model
            best_macc = macc
            best_acc = acc1
            best_epoch = epoch

        scheduler.step()

    loss, acc1, auc_ap, class_acc = test(best_model, test_loader, criterion,
                                         device)
    macc = class_acc().mean()
    auc, ap = auc_ap(args.num_classes == 2)

    torch.save(best_model, os.path.join(args.model_dir, 'model_ori.pth'))
    print('COMPLETE !!!')
    print('BEST EPOCH', best_epoch)
    print('BEST VAL mACC', best_macc)
    print('BEST VAL ACC', best_acc)
    print('-'*10)
    print('TEST mACC', macc)
    print('TEST ACC', acc1)
    print(class_acc)
    print('TEST AP', ap)
    print('TEST AUC', auc)


def train(model, loader, criterion, optimizer, device):
    loss_meter = AverageMeter('Loss', ':.4f')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(total=len(loader),
                             step=20,
                             prefix='Training',
                             meters=[loss_meter])
    class_acc = metrics.ClassAccuracy()
    auc_ap = metrics.AUC_AP()

    model.train()

    for i, data in enumerate(loader):
        data = data.to(device)
        outputs = model(data.x, data.edge_index, data.batch)
        loss = criterion(outputs, data.y)
        acc1 = metrics.accuracy(outputs, data.y)[0]

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        class_acc.update(outputs, data.y)
        auc_ap.update(outputs, data.y)
        loss_meter.update(loss.item(), len(data.y))
        acc1_meter.update(acc1.item(), len(data.y))
        progress.display(i)

    return loss_meter.avg, acc1_meter.avg, auc_ap, class_acc


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

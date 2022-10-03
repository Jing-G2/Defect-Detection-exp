import os
import argparse
import time
import shutil
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp_name', default='', type=str, help='exp name')
    parser.add_argument('--result_path',
                        default='',
                        type=str,
                        help='result path')
    parser.add_argument('--model_name',
                        default='',
                        type=str,
                        help='model name')
    parser.add_argument('--loader_name',
                        default='',
                        type=str,
                        help='loader name')
    parser.add_argument('--num_classes',
                        default='',
                        type=int,
                        help='num classes')
    parser.add_argument('--num_epochs',
                        default=100,
                        type=int,
                        help='num epochs')
    parser.add_argument('--data_path', default='', type=str, help='data path')
    parser.add_argument('--device_index',
                        default='0',
                        type=str,
                        help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(args.result_path, 'results', args.exp_name,
                              'models')
    log_path = os.path.join(args.result_path, 'runs', args.exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('RESULT PATH:\n', model_path, '\n', log_path)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    model = models.load_model(args.model_name, num_classes=args.num_classes)
    model.to(device)

    train_loader = loaders.load_data(args.loader_name, args.data_path, 'train')
    test_loader = loaders.load_data(args.loader_name, args.data_path, 'test')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(),
                          lr=0.01,
                          momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                     T_max=args.num_epochs)

    writer = SummaryWriter(log_path)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    best_acc = None
    best_epoch = None

    for epoch in tqdm(range(args.num_epochs)):
        losses, top1, top5 = train(train_loader, model, criterion, optimizer,
                                   device)
        writer.add_scalar(tag='training loss',
                          scalar_value=losses.avg,
                          global_step=epoch)
        writer.add_scalar(tag='training acc@1',
                          scalar_value=top1.avg,
                          global_step=epoch)
        losses, top1, top5 = test(test_loader, model, criterion, device)
        writer.add_scalar(tag='test loss',
                          scalar_value=losses.avg,
                          global_step=epoch)
        writer.add_scalar(tag='test acc@1',
                          scalar_value=top1.avg,
                          global_step=epoch)

        # ----------------------------------------
        # save best model
        # ----------------------------------------
        if best_acc is None or best_acc < top1.avg:
            best_acc = top1.avg
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(model_path, 'ori_model.pth'))

        scheduler.step()

    print('COMPLETE !!!')
    print('TIME CONSUMED', time.time() - since)
    print('BEST ACC', best_acc)
    print('BEST EPOCH', best_epoch)


def train(train_loader, model, criterion, optimizer, device):
    """ validate the training model """
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(num_total=len(train_loader),
                             meters=[losses, top1, top5],
                             prefix='Training',
                             step=20)

    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        progress.display(i)

    return losses, top1, top5


def test(test_loader, model, criterion, device):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(num_total=len(test_loader),
                             meters=[losses, top1, top5],
                             prefix='Test',
                             step=20)

    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            progress.display(i)

    return losses, top1, top5


if __name__ == '__main__':
    main()

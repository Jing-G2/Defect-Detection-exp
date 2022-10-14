import argparse
import os
import torch
from torch_scatter import scatter

import models
from utils.pattern_sift import Sift

class PatternSift(Sift):
    def __init__(self, modules, num_classes, num_samples, value_type):
        super().__init__(modules, num_classes, num_samples, value_type)

    def __call__(self, outputs, labels, batch):
        softmax = torch.nn.Softmax(dim=1)(outputs.detach())  # [batch_size, num_classes]
        scores, predicts = torch.max(softmax, dim=1)  # [batch_size]
        nll_loss = torch.nn.NLLLoss()(outputs, labels)  # single value

        for layer, module in enumerate(self.modules):  # each layer
            if self.value_type[0] == 'g':
                values = self.gradient(module, -nll_loss)  # [n, o]
            else:
                values = self.contribution(module)  # [n, o, i]
            if values.shape[0] > len(labels):
                values = scatter(values, batch, dim=0, reduce='mean')
            values = values.detach().cpu().numpy()  # [batch_size, num_channels]
            print('===', layer, len(labels))
            print('===', layer, values.shape)

            for i, label in enumerate(labels):  # each datas
                # if label == predicts[i]:
                if self.nums[layer][label] == self.num_samples:
                    score_min, index = torch.min(self.scores[layer][label], dim=0)
                    if scores[i] > score_min:
                        self.values[layer][label][index] = values[i]
                        self.scores[layer][label][index] = scores[i]
                else:
                    self.values[layer][label].append(values[i])
                    self.scores[layer][label][self.nums[layer][label]] = scores[i]
                    self.nums[layer][label] += 1


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', default='', type=str, help='datas name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='datas dir')
    parser.add_argument('--pattern_dir', default='', type=str, help='pattern dir')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:'+args.device_index  if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.pattern_dir):
        os.makedirs(args.pattern_dir)

    print('-' * 50)
    print('DEVICE:', device)
    print('MODEL PATH:', args.model_path)
    print('PATTERN DIR:', args.pattern_dir)
    print('-' * 50)

    # ----------------------------------------
    # sift configuration
    # ----------------------------------------
    train_loader, _ = datas.load_data(args.data_dir, args.data_name, 'gc')

    model = torch.load(args.model_path)
    model.to(device)
    model.eval()

    modules = models.load_modules(model)

    sift = PatternSift(modules=modules, num_classes=args.num_classes, num_samples=8, value_type='c+')

    for i, data in enumerate(train_loader):
        data = data.to(device)
        outputs = model(data.x, data.edge_index, data.batch)
        print('***o', outputs.shape)
        sift(outputs, data.y, data.batch)

    # sift.sift(args.pattern_dir, threshold=0.2)
    sift.sift_partial(args.pattern_dir, alpha=0.3, beta=0.2)
    sift.sift_std(args.pattern_dir)
    # pv.visualize_global_by_labels(args.pattern_dir,
    #                               layers=[0, 1, 2, 3],
    #                               values_=sift.values,
    #                               value_type='c+')
    # pv.visualize_global_by_images(args.pattern_dir,
    #                               layers=[0, 1, 2, 3],
    #                               labels=[0, 1],
    #                               values_=sift.values,
    #                               value_type='c+')


if __name__ == '__main__':
    main()
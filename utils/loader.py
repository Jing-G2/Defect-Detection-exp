import glob
import os
import random
import pandas as pd
import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.data_util import upsample


def load_data(data_dir, data_name, batch_size, tpye='gc'):
    class_names = os.listdir(os.path.join(data_dir, data_name))
    for dir_name in class_names:
        if os.path.isfile(os.path.join(data_dir, data_name, dir_name)):
            class_names.remove(dir_name)
    print('class_names', class_names)

    train_dataset = []
    val_dataset = []
    test_dataset = []
    graph_num = 0

    for class_name in class_names:
        nodes_files = glob.glob(
            os.path.join(data_dir, data_name, class_name, f'*_nodes.csv'))
        edges_files = glob.glob(
            os.path.join(data_dir, data_name, class_name, f'*_edges.csv'))
        print(class_name + ' #graph:', len(nodes_files))

        dataset = []
        for node_files, edges_file in zip(nodes_files, edges_files):
            x = load_nodes(node_files, 4)
            edge_index = load_edges(edges_file)
            assert x.max() >= edge_index.max(), "{}, {}, {}, {}".format(
                node_files, edges_file, x.max(), edge_index.max())
            y = class_names.index(class_name)
            assert y < 2
            data = Data(x=x, edge_index=edge_index, y=y)

            graph_num += 1
            dataset.append(data)

        random.shuffle(dataset)
        train_dataset += dataset[:int(len(dataset) * 0.6)]
        val_dataset += dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
        test_dataset += dataset[int(len(dataset) * 0.8):]

    # upsample for train_dataset and val_dataset
    # train_dataset = upsample(train_dataset)

    random.shuffle(train_dataset)
    random.shuffle(val_dataset)
    random.shuffle(test_dataset)
    print('dataset len:', len(train_dataset), len(val_dataset),
          len(test_dataset))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    print('dataloader len:', len(train_loader), len(val_loader),
          len(test_loader))

    return train_loader, val_loader, test_loader


def load_nodes(nodes_dir, num_nodes_feature=4):
    nodes = pd.read_csv(nodes_dir, usecols=range(0, num_nodes_feature + 1))
    x = nodes.to_numpy(np.float32)
    # print('x shape:',x.shape)
    x = torch.from_numpy(x).cuda()
    return x


def load_edges(edges_dir):
    edges = pd.read_csv(edges_dir)
    edges_from_index = edges['from_node'].to_numpy()
    edges_to_index = edges['to_node'].to_numpy()
    # print('edges_from_index.shape:',edges_from_index.shape)
    edge_index = np.array([edges_from_index, edges_to_index])
    edge_index = torch.from_numpy(edge_index).long().cuda()
    return edge_index


# if __name__ == '__main__':
#     dataset = fetch_dataset('/nfs4-p1/gj/DEFECT2022/data', 'KSDD')

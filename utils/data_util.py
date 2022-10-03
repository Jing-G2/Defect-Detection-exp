import torch
import random
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import add_remaining_self_loops, degree, dropout_adj
from torch_geometric.loader import DataLoader
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_sparse import SparseTensor



def upsample(dataset):
    y = np.array([dataset[i].y for i in range(len(dataset))])
    classes ,num_class_graph = np.unique(y, return_counts=True)
    max_num_class_graph = max(num_class_graph)

    chosen_samples = []
    for i in range(len(classes)):
        train_index = np.where((y == classes[i]) == True)[0].tolist()
        if len(train_index) == max_num_class_graph:
            continue

        up_sample_ratio = max_num_class_graph / num_class_graph[i]
        up_sample_num = int(num_class_graph[i] * up_sample_ratio -
                            num_class_graph[i])

        int_part = int(up_sample_num / len(train_index))
        up_sample = train_index * int_part
        frac_part = up_sample_num - len(train_index) * int_part
        up_sample.extend(random.sample(train_index, frac_part))
        chosen_samples.extend(up_sample)

    chosen_samples = np.array(chosen_samples)
    extend_data = [dataset[i] for i in chosen_samples]
    data = dataset + extend_data

    return data

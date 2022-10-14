import numpy as np
import os
from tqdm import tqdm
import torch
import torch_geometric

'''
base class
'''
def partial_gcn(gcn: torch_geometric.nn.GCNConv, inp: torch.Tensor):
    weight = gcn.lin.weight
    bias = gcn.bias

    N = inp.shape[0]
    I = weight.shape[1]
    O = weight.shape[0]

    inp = inp.unsqueeze(1).expand(N, O, I)
    weight = weight.unsqueeze(0).expand(N, O, I)
    out = inp * weight

    # out = out.sum(2)
    # out = out + bias
    return out  # [B O I]

def partial_gat(gat: torch_geometric.nn.GATConv, inp: torch.Tensor):
    weight = gat.lin_src.weight

    N = inp.shape[0]
    I = weight.shape[1]
    O = weight.shape[0]

    inp = inp.unsqueeze(1).expand(N, O, I)
    weight = weight.unsqueeze(0).expand(N, O, I)
    out = inp * weight

    return out  # [B O I]


def partial_sage(sage: torch_geometric.nn.SAGEConv, inp: torch.Tensor):
    weight = sage.lin_l.weight

    N = inp.shape[0]
    I = weight.shape[1]
    O = weight.shape[0]

    inp = inp.unsqueeze(1).expand(N, O, I)
    weight = weight.unsqueeze(0).expand(N, O, I)
    out = inp * weight

    return out  # [B O I]


def partial_linear(linear: torch.nn.Linear, inp: torch.Tensor):
    # inp: B I
    weight = linear.weight  # [O I]
    bias = linear.bias  # O

    B = inp.shape[0]
    I = weight.shape[1]
    O = weight.shape[0]

    inp = inp.unsqueeze(1).expand(B, O, I)
    weight = weight.unsqueeze(0).expand(B, O, I)
    out = inp * weight

    # out = out.sum(2)
    # out = out + bias
    return out  # [B O I]


def grads(outputs, inputs, retain_graph=True, create_graph=False):
    return torch.autograd.grad(outputs=outputs,
                               inputs=inputs,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]


def normalization(data, axis=None, bot=False):
    assert axis in [None, 0, 1]
    _max = np.max(data, axis=axis)
    if bot:
        _min = np.zeros(_max.shape)
    else:
        _min = np.min(data, axis=axis)
    _range = _max - _min
    if axis == 1:
        _norm = ((data.T - _min) / (_range + 1e-5)).T
    else:
        _norm = (data - _min) / (_range + 1e-5)
    return _norm


class HookModule:
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.outputs = None
        module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.inputs = inputs[0]
        self.outputs = outputs


class Sift:
    def __init__(self, modules, num_classes, num_samples, value_type):
        self.modules = [HookModule(module) for module in modules]
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.value_type = value_type
        assert value_type in ['a+', 'g+', 'c+']

        # [num_modules, num_classes, num_samples, channels]
        self.values = [[[] for _ in range(num_classes)] for _ in range(len(modules))]
        self.scores = torch.zeros((len(modules), num_classes, num_samples))
        self.nums = torch.zeros((len(modules), num_classes), dtype=torch.long)


    # ----------------------------------------
    # a, g, c
    # ----------------------------------------
    def activation(self, module):
        values = module.outputs
        if self.value_type == 'a+':
            values = torch.relu(values)
        return values

    def gradient(self, module, leafs):
        values = grads(leafs, module.outputs)
        if self.value_type == 'g+':
            if values is not None:
                values = torch.relu(values)
        return values

    def contribution(self, module):
        values = None
        if isinstance(module.module, torch.nn.Linear):
            values = partial_linear(module.module, module.inputs)  # [n, o, i]
        if isinstance(module.module, torch_geometric.nn.GCNConv):
            values = partial_gcn(module.module, module.inputs)  # [n, o, i]
        if isinstance(module.module, torch_geometric.nn.GATConv):
            values = partial_gat(module.module, module.inputs)  # [n, o, i]
        if isinstance(module.module, torch_geometric.nn.SAGEConv):
            values = partial_sage(module.module, module.inputs)  # [n, o, i]

        if self.value_type == 'c+':
            values = torch.relu(values)  # [n, o, i]
        return values

    # ----------------------------------------
    # sitf std
    # ----------------------------------------
    def sift_hidden_value(self, result_path):
        assert self.value_type in ['a+']
        for layer, values in enumerate(self.values):
            values = np.asarray(values)  # [num_classes, num_samples, channels]
            print('***', layer, values.shape)

            sta = np.mean(values, axis=1)
            path = os.path.join(result_path, 'std_layer{}.npy'.format(layer))
            np.save(path, sta)

    # ----------------------------------------
    # sitf std
    # ----------------------------------------
    def sift_std(self, result_path):
        assert self.value_type in ['c+']
        for layer, values in enumerate(self.values):
            values = np.asarray(values)  # [num_classes, num_samples, o, i]
            print('***', layer, values.shape)

            values = np.maximum(values, 0)
            values = np.min(values, axis=1)
            values = np.sum(values, axis=1)
            print(values)

            path = os.path.join(result_path, 'std_layer{}.npy'.format(layer))
            np.save(path, values)

    # ----------------------------------------
    # sift a, g
    # ----------------------------------------
    def sift(self, result_path, threshold):
        print(self.nums)
        for layer, values in enumerate(tqdm(self.values)):  # [num_modules, num_classes, num_samples, channels]
            values = np.asarray(values)  # [num_classes, num_samples, channels]
            print('+++', layer, values.shape)

            values = np.sum(values, axis=1)  # [num_classes, channels]
            values = normalization(values, axis=1)

            mask = np.zeros(values.shape)
            mask[np.where(values > threshold)] = 1
            mask_path = os.path.join(result_path, '{}_layer{}.npy'.format(self.value_type, layer))
            np.save(mask_path, mask)

    # ----------------------------------------
    # sift c
    # ----------------------------------------
    def sift_partial(self, result_path, alpha, beta):
        print(self.nums)

        # layer -1
        mask = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            mask[i][i] = 1
        mask_path = os.path.join(result_path, '{}_layer{}.npy'.format(self.value_type, '-1'))
        np.save(mask_path, mask)
        print(mask)

        # other layers
        for layer, values in enumerate(tqdm(self.values)):  # [num_modules, num_classes, num_samples, channels]
            values = np.asarray(values)  # [num_classes, num_samples, channels] # [channels: o, i]
            print('+++', layer, values.shape)

            values = np.sum(values, axis=1)  # [num_classes, o, i]
            datas = []
            for label, value in enumerate(values):
                value = normalization(value, axis=1)  # [o, i]
                data = np.zeros(value.shape)
                data[np.where(value > alpha)] = 1  # [o, i]
                data = np.sum(data, axis=0)  # [i]
                # data_mask = np.expand_dims(mask[label], 0)  # [1, o]
                # data = np.matmul(data_mask, data)[0]  # [1, o] * [o, i] = [i]
                datas.append(data)

                # value = normalization(value, axis=1)  # [o, i]
                # data_mask = np.expand_dims(mask[label], 0)  # [1, o]
                # data = np.matmul(data_mask, value)[0]  # [1, o] * [o, i] = [i]
                # print(data)
                # datas.append(data)

            datas = np.asarray(datas)  # [num_classes, i]
            datas = normalization(datas, axis=1, bot=True)
            mask = np.zeros(datas.shape)
            mask[np.where(datas > beta)] = 1
            mask_path = os.path.join(result_path, '{}_layer{}.npy'.format(self.value_type, layer))
            np.save(mask_path, mask)

    def distribute(self, result_path):
        from utils import draw_util

        for layer, values in enumerate(self.values):
            values = np.asarray(values)  # [num_classes, num_samples, channels]
            print('***', layer, values.shape)

            # stat = np.sum(values, axis=1)
            # path = os.path.join(result_path, 'stat_layer{}.npy'.format(layer))
            # np.save(path, stat)
            # print(stat.tolist())

            # cate = 3
            # path = '/nfs3-p1/hjc/gcn_lego/figs/gcn_cora/{}_{}.png'.format(layer, cate)
            # draw_util.draw_distribute(values[cate], path)
            # for sample, channels in enumerate(values[cate]):  # [channels]
            #     print(channels)

            # v_path = os.path.join(result_path, 'd_{}_layer{}.npy'.format(self.value_type, layer))
            # np.save(v_path, values)
            print(values.tolist())

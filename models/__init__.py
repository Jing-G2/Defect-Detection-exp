import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import GAE, VGAE, ARGA, ARGVA, SAGEConv

from models import GCN_gc, vgg


def load_model(model_name, num_node_features, num_classes, task_type=None):
    assert task_type in [None, 'nc', 'lp', 'gc']

    model = None
    hidden_channels = 16
    if task_type == 'gc':
        if model_name == 'GCN':
            model = GCN_gc.GCN(num_node_features,hidden_channels,  num_classes)
        # elif model_name == 'GAT':
        #     model = GAT_gc.GAT(num_node_features,hidden_channels,  num_classes)
        # elif model_name == 'SAGE':
        #     model = SAGE_gc.SAGE(num_node_features,hidden_channels,  num_classes)
    else:
        model = vgg.vgg16_bn(1, num_classes)

    return model


def load_modules(model, model_layers=None):
    assert model_layers is None or type(model_layers) is list

    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            modules.append(module)
        if isinstance(module, GCNConv):
            modules.append(module)
        if isinstance(module, GATConv):
            modules.append(module)
        if isinstance(module, SAGEConv):
            modules.append(module)

    modules.reverse()  # reverse order
    if model_layers is None:
        model_modules = modules
    else:
        model_modules = []
        for layer in model_layers:
            model_modules.append(modules[layer])

    print('-' * 50)
    print('Model Layers:', model_layers)
    print('Model Modules:', model_modules)
    print('Model Modules Length:', len(model_modules))
    print('-' * 50)

    return model_modules


if __name__ == '__main__':
    model = load_model('GCN', 7, 2, 'gc')
    print(model)

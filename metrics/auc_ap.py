import torch
from sklearn.metrics import average_precision_score, roc_auc_score


class AUC_AP:
    def __init__(self):
        self.pred = None
        self.y = None

    def update(self, pred, y):
        pred_ = torch.softmax(pred, dim=1).detach().cpu()
        y_ = y.detach().cpu()
        if self.pred is None:
            self.pred = pred_
            self.y = y_
        self.pred = torch.cat((self.pred, pred_), dim=0)
        self.y = torch.cat((self.y, y_), dim=0)

    def __call__(self, bc=False):
        # print(bc)
        if bc:
            pred_ = torch.gather(self.pred, 1, torch.unsqueeze(self.y, 1)).squeeze_(1)
            pred_ = pred_.numpy()
            y_ = self.y.numpy()
            # print(y_.shape, pred_.shape)
            return roc_auc_score(y_, pred_), average_precision_score(y_, pred_)
        else:
            return 0, 0


if __name__ == '__main__':
    import torch
    import numpy as np

    # y = np.asarray([1, 0, 1, 1, 0])
    # pred = np.asarray([0.52, 0.65, 0.63, 0.65, 0.21])
    # y = np.asarray([1, 0, 1, 2, 0])
    # pred = np.asarray([[0.52, 0.14, 0.34], [0.65, 0.01, 0.34], [0.63, 0.03, 0.34], [0.65, 0.01, 0.34], [0.21, 0.45, 0.34]])

    y = torch.asarray([1, 0, 1, 1, 0])
    pred = torch.asarray([[0.52, 0.48], [0.65, 0.35], [0.63, 0.37], [0.65, 0.35], [0.21, 0.79]])

    # print(auc_ap(y, pred))
import csv
import random
from functools import partialmethod

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_precision_and_recall(outputs, targets, pos_label=1):
    import warnings
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(), pred.cpu().numpy())

        # add f1
        epsilon = 1e-7
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    #
    # # check if all targets 0s
    # if torch.sum(targets) == 0:
    #     recall = None
    #     f1 = None
    #
    # # check if all preds 0s
    # if torch.sum(pred) == 0:
    #     precision = None
    #     f1 = None
    #
    # if precision is not None:
    #     precision = precision[pos_label]
    #
    # if recall is not None:
    #     recall = recall[pos_label]
    #
    # if f1 is not None:
    #     f1 = f1[pos_label]
    #
    # return precision, recall, f1
    #
    # # possible to have precision or recall be none, while other is 0
    # if f1 is None:
    #     return precision, recall, f1

    # # import pdb; pdb.set_trace()

    if len(precision) == 1:
        return 0.0, 0.0, 0.0

    return precision[pos_label], recall[pos_label], f1[pos_label]

def get_tp(preds, targets):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = preds / targets
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    # false_positives = torch.sum(confusion_vector == float('inf')).item()
    # true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    # false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives

def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass
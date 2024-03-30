import torch
import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import classification_report
import csv
import os
import torch.nn as nn


############# For Training #################


def save_checkpoint(save_path, model, valid_loss):
    """save model
    """
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(), 'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, args):
    """load model
    """
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location=args.device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def load_partial_dict(load_path, model, args):
    """ load partial dict from load_path
    """
    state_dict = torch.load(load_path, map_location=args.device)
    model.load_state_dict(state_dict['model_state_dict'], strict=False)
    print(f'Model partial parameters loaded from <== {load_path}')


def draw_fig_loss(loss_list_train, loss_list_val, global_steps_list, save_dir):
    """loss~global_steps
    """
    plt.cla()
    plt.plot(global_steps_list, loss_list_train, label='Train Loss')
    plt.plot(global_steps_list, loss_list_val, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'train_valid_loss.jpg'), dpi=300)
    # plt.show()


def draw_fig_acc(acc_list_train, acc_list_val, global_steps_list, save_dir):
    """acc~global_steps
    """
    plt.cla()
    plt.plot(global_steps_list, acc_list_train, label='Train Acc.')
    plt.plot(global_steps_list, acc_list_val, label='Valid Acc.')
    # plt.xlabel('Global Steps')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'train_valid_acc.jpg'), dpi=300)
    # plt.show()
    



def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

############# For Evaluation #################


def eval_classification_report(out_log, do_print=False):
    """使用Classification_report直接计算评价指标(缺少auc值),打印/返回评价指标矩阵

    Args:
        out_log (_type_): 模型处理一次数据集的结果

    Returns:
        _type_: 
    """
    pred_y_list = []
    y_list = []
    for batch in out_log:
        pred_y = batch[0].detach().cpu().numpy().argmax(axis=1).tolist()
        y = batch[1].detach().cpu().numpy().tolist()
        pred_y_list.extend(pred_y)
        y_list.extend(y)
    out = classification_report(y_list, pred_y_list, labels=[1, 0], target_names=['Fake', 'True'], digits=4, output_dict=(not do_print))
    if do_print:
        print(out)
    return out


def create_out_csv(root, name):
    """创建评价指标的csv文件

    Args:
        path (_type_): 
        name (_type_): Model_NAME
    """
    path = os.path.join(root, name + '_out.csv')
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ['Args', 'Accuracy', 'Mac.F1', 'F-P', 'F-R', 'F-F1', 'T-P', 'T-R', 'T-F1']
        csv_write.writerow(csv_head)


def append_out_csv(root, name, out, args):
    """添加评价指标矩阵到csv文件的新一行

    Args:
        path (_type_): 
        name (_type_): Model_NAME
        out (_type_): 评价指标矩阵
    """
    path = os.path.join(root, name + '_out.csv')
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        data_row = [args, "{:.4f}".format(out['accuracy']), "{:.4f}".format(out['macro avg']['f1-score']),
                    "{:.4f}".format(out['Fake']['precision']), "{:.4f}".format(out['Fake']['recall']),
                    "{:.4f}".format(out['Fake']['f1-score']), "{:.4f}".format(out['True']['precision']), "{:.4f}".format(out['True']['recall']), "{:.4f}".format(out['True']['f1-score'])]
        csv_write.writerow(data_row)
        print(f'Out saved to ==> {path}')


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(
        self, optimizer, patience=4, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStoppingAcc():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = float("Inf")
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc == float("Inf"):
            self.best_acc = val_acc
        elif self.best_acc - val_acc < self.min_delta:
            self.best_acc = val_acc
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_acc - val_acc > self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

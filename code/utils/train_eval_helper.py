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


def draw_fig_acc(acc_list_train, acc_list_val, global_steps_list, save_dir):
    """acc~global_steps
    """
    plt.cla()
    plt.plot(global_steps_list, acc_list_train, label='Train Acc.')
    plt.plot(global_steps_list, acc_list_val, label='Valid Acc.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'train_valid_acc.jpg'), dpi=300)


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


def eval_classification_report(out_log, do_print=False):
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
    path = os.path.join(root, name + '_out.csv')
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_head = ['Args', 'Accuracy', 'Mac.F1', 'F-P', 'F-R', 'F-F1', 'T-P', 'T-R', 'T-F1']
        csv_write.writerow(csv_head)


def append_out_csv(root, name, out, args):
    path = os.path.join(root, name + '_out.csv')
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        data_row = [args, out['accuracy'], out['macro avg']['f1-score'],
                    out['Fake']['precision'], out['Fake']['recall'],
                    out['Fake']['f1-score'], out['True']['precision'], out['True']['recall'], out['True']['f1-score']]
        csv_write.writerow(data_row)
        print(f'Out saved to ==> {path}')

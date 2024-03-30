import argparse
from utils.train_eval_helper import *
parser = argparse.ArgumentParser()
parser.add_argument('--a_note', type=str, default=None, help='changes to note')
parser.add_argument('--dataset', type=str, default='weibo', help='dataset, e.g. twitter, weibo')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:1', help='specify cuda devices')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=2e-6, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--conv_out', type=int, default=64, help='weight decay')
parser.add_argument('--crop_num', type=int, default=6, help='# of crops, 1+M')  
parser.add_argument('--st_num', type=int, default=31, help='# of sentences, 1+N')
parser.add_argument('--dropout_p', type=float, default=0, help='dropout proportion')
parser.add_argument('--layer_num', type=int, default=8, help='# of transformer encoder block layers')
parser.add_argument('--conv_kernel', nargs='+', type=int, default=[1, 2, 3], help='an integer array')
parser.add_argument('--epochs', type=int, default=30, help='maximum number of epochs')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--name', type=str, default='default', help='save sub-dir')
parser.add_argument('--model', type=str, default='new', help='choose model to train')
parser.add_argument('--finetune', type=lambda x: x.lower() == 'true', default=False, help='whther to finetune')
parser.add_argument('--lr_scheduler', type=lambda x: x.lower() == 'true', default=False, help='whther to use lr_scheduler')
parser.add_argument('--early_stopping', type=lambda x: x.lower() == 'true', default=False, help='whther to use early_stopping')
parser.add_argument('--loadpath', type=str, default=False, help='pretrained-model dict to load')
parser.add_argument('--checkpoint', type=str, default=False, help='checkpoint to load')

args = parser.parse_args()
print(args)
set_random_seed(args.seed, deterministic=False)
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from utils.data_loader_new import *
from models import *
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import signal
import atexit

if args.dataset == 'weibo':
    processed_dir = "/sda/qiaojiao/code/Weibo16/processed"
    big_processed_dir = "/sda/qiaojiao/code/Weibo16/processed"
    save_dir = "/sda/qiaojiao/code/Weibo16/save"

    df_train = np.load(processed_dir + "/train_EANN_frozen.npy", allow_pickle=True)
    df_valid = np.load(processed_dir + "/valid_EANN_frozen.npy", allow_pickle=True)
    df_test = np.load(processed_dir + "/test_EANN_frozen.npy", allow_pickle=True)
    df_columns = ['original_post', 'label', 'image_id', 'post_id']
    df_train = pd.DataFrame(df_train, columns=df_columns)
    df_valid = pd.DataFrame(df_valid, columns=df_columns)
    df_test = pd.DataFrame(df_test, columns=df_columns)
        
    n_words = np.load(processed_dir + '/n_tokens.npy', allow_pickle=True).item()
    crop_input = np.load(processed_dir + '/clip_image_preprocess.npy', allow_pickle=True).item()
    text_input = np.load(processed_dir + '/word_clipinputs.npy', allow_pickle=True).item()

    train_dataset = FakeNewsDataset(df_train, args.crop_num, args.st_num, args.dataset, n_words, crop_input, text_input)
    valid_dataset = FakeNewsDataset(df_valid, args.crop_num, args.st_num, args.dataset, n_words, crop_input, text_input)
    test_dataset = FakeNewsDataset(df_test, args.crop_num, args.st_num, args.dataset, n_words, crop_input, text_input)


else:    
    processed_dir = "/sda/qiaojiao/code/Mediaeval2015/processed"
    big_processed_dir = "/sda/qiaojiao/code/Mediaeval2015/processed"
    save_dir = "/sda/qiaojiao/code/Mediaeval2015/save"

    df_train = np.load(processed_dir + "/dev_multilingual.npy", allow_pickle=True)  
    df_valid = np.load(processed_dir + "/test_multilingual.npy", allow_pickle=True)  
    df_columns = ['post_id', 'original_post', 'image_id', 'label', 'event', 'imagepath']
    df_train = pd.DataFrame(df_train, columns=df_columns)
    df_valid = pd.DataFrame(df_valid, columns=df_columns)
        
    n_words = np.load(processed_dir + '/n_features.npy', allow_pickle=True).item()
    crop_input = np.load(big_processed_dir + "/clip_crop_feature.npy", allow_pickle=True).item()
    text_input = np.load(big_processed_dir + "/word_clipfeatures.npy", allow_pickle=True).item()

    train_dataset = FakeNewsDataset(df_train, args.crop_num, args.st_num, args.dataset, n_words, crop_input, text_input)
    valid_dataset = FakeNewsDataset(df_valid, args.crop_num, args.st_num, args.dataset, n_words, crop_input, text_input)
    test_dataset = FakeNewsDataset(df_valid, args.crop_num, args.st_num, args.dataset, n_words, crop_input, text_input)

# make dir for each version
save_dir = save_dir + "/" + args.name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False) 
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
print("train size, integer loader length: ", len(train_dataset), str((len(train_loader) - 1) * args.batch_size))
print("valid size, integer loader length: ", len(valid_dataset), str((len(valid_loader) - 1) * args.batch_size))


model_dic = {
    'new': C3N,
}

net = model_dic[args.model]
model = net(args).to(args.device)
if args.checkpoint:
    valid_loss = load_checkpoint(args.checkpoint, model, args)  # training from checkpoint
else:
    valid_loss = float("Inf")  # training from origin
if args.loadpath:
    load_partial_dict(model, args, args.loadpath)


print(model)
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.requires_grad)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

# either initialize early stopping or learning rate scheduler
if args.lr_scheduler:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)
if args.early_stopping:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStoppingAcc()
    

@torch.no_grad()
def compute_test(model_, loader):
    model_.eval()
    loss_test = 0.0
    out_log = []
    for data in loader:
        data = {x: data[x].to(args.device) if not (isinstance(data[x], list) or isinstance(data[x], dict))  else data[x] for x in data}
        out = model_(data)
        out_log.append([F.softmax(out, dim=1), data['label']])
        loss_test += F.nll_loss(out, data['label']).item()
    return out_log, loss_test


def train(model, optimizer, train_loader, valid_loader, valid_loss):
    # Training
    print("Start Training!")
    best_loss_val = valid_loss
    best_acc_val = 0
    loss_list_train = []
    loss_list_val = []
    acc_list_train = []
    acc_list_val = []
    epoch_list = []
    global_step = 0

    for epoch in tqdm(range(args.epochs), file=sys.stdout):
        out_log = []
        loss_train = 0.0
        model.train()
        for data in train_loader:
            data = {x: data[x].to(args.device) if not (isinstance(data[x], list) or isinstance(data[x], dict))  else data[x] for x in data}
            out = model(data)
            loss = F.nll_loss(out, data['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), data['label']])
            global_step += 1

        # evaluate model and print after each epoch
        eval_out_train = eval_classification_report(out_log)
        out_log_val, loss_val = compute_test(model, valid_loader)
        eval_out_valid = eval_classification_report(out_log_val)
        loss_list_train.append(loss_train / len(train_loader))
        loss_list_val.append(loss_val / len(valid_loader))
        acc_list_train.append(eval_out_train['accuracy'])
        acc_list_val.append(eval_out_valid['accuracy'])
        epoch_list.append(epoch)

        print(f'  loss_train: {loss_train/len(train_loader):.4f}, acc_train: {eval_out_train["accuracy"]:.4f}, '
              f'loss_val: {loss_val/len(valid_loader):.4f}, acc_val: {eval_out_valid["accuracy"]:.4f}')

        acc_val = eval_out_valid['accuracy']
        if best_acc_val < acc_val:
            best_acc_val = acc_val
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, best_acc_val)

        if args.lr_scheduler:
            lr_scheduler(acc_val)
        if args.early_stopping:
            early_stopping(acc_val)
            if early_stopping.early_stop:
                break
        drawing_list = [loss_list_train, loss_list_val, acc_list_train, acc_list_val, epoch_list]
        np.save(os.path.join(save_dir, 'drawing_list.npy'), drawing_list)

    print('Finished Training!')
    draw_fig_loss(loss_list_train, loss_list_val, epoch_list, save_dir)
    draw_fig_acc(acc_list_train, acc_list_val, epoch_list, save_dir)

def evaluation_save():
    best_model = net(args).to(args.device)
    load_checkpoint(os.path.join(save_dir, 'model.pt'), best_model, args)
    print('Valid Classification Report')
    out_log_val, _ = compute_test(best_model, valid_loader)
    eval_classification_report(out_log_val, do_print=True)

    print('Test Classification Report')
    out_log_test, _ = compute_test(best_model, test_loader)
    eval_classification_report(out_log_test, do_print=True)

    # save results to .csv
    if not os.path.exists(os.path.join(save_dir, 'Exp1_out.csv')):
        create_out_csv(save_dir, 'Exp1')
    append_out_csv(save_dir, 'Exp1', eval_classification_report(out_log_test), args)

def execute_when_process_is_killed():
    print("Terminated, draw the last figures.")
    arr = np.load(os.path.join(save_dir, 'drawing_list.npy'))
    draw_fig_loss(arr[0].tolist(), arr[1].tolist(), arr[-1].tolist(), save_dir)
    draw_fig_acc(arr[2].tolist(), arr[3].tolist(), arr[-1].tolist(), save_dir)
    evaluation_save()
    print("--------------- end -------------------")

atexit.register(execute_when_process_is_killed)
signal.signal(signal.SIGTERM, execute_when_process_is_killed)
signal.signal(signal.SIGINT, execute_when_process_is_killed)

# TODO
train(model, optimizer, train_loader, valid_loader, valid_loss=valid_loss)
        
# execute_when_process_is_killed()

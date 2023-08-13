import torch
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from utils.train_eval_helper import *
from utils.data_loader import *
from model import C3N
from torch.utils.data import DataLoader
import torch.optim as optim
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weibo', help='dataset, e.g. twitter, weibo')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--conv_out', type=int, default=64, help='# of output channels in convlution')
parser.add_argument('--crop_num', type=int, default=6, help='# of crops, 1+M')  
parser.add_argument('--word_num', type=int, default=200, help='# of words, 1+N') 
parser.add_argument('--dropout_p', type=float, default=0, help='dropout proportion')
parser.add_argument('--layer_num', type=int, default=8, help='# of transformer encoder block layers')
parser.add_argument('--epochs', type=int, default=1, help='maximum number of epochs')
parser.add_argument('--name', type=str, default='C3N', help='save sub-dir')
parser.add_argument('--clip_model', type=str, default="ViT-B-16", help='clip model name')
parser.add_argument('--finetune', type=lambda x: x.lower() == 'true', default=True, help='whether to finetune')
args = parser.parse_args()
print(args)
set_random_seed(args.seed, deterministic=False)

processed_dir = "../data/weibo/processed"
big_processed_dir = "It is recommended to use a special directory to store large files."
save_dir = "Directory to store models and results"

df_train = np.load(processed_dir + "/train.npy", allow_pickle=True)
df_valid = np.load(processed_dir + "/valid.npy", allow_pickle=True)
df_test = np.load(processed_dir + "/test.npy", allow_pickle=True)
df_columns = ['post_id', 'image_id', 'original_post', 'label']
df_train = pd.DataFrame(df_train, columns=df_columns)
df_valid = pd.DataFrame(df_valid, columns=df_columns)
df_test = pd.DataFrame(df_test, columns=df_columns)

crop_features = np.load(big_processed_dir + "/clip_crop_preprocess.npy", allow_pickle=True).item()
word_features = np.load(big_processed_dir + "/word_clipinputs.npy", allow_pickle=True).item() 

train_dataset = FakeNewsDataset(df_train, args.crop_num, crop_features, word_features, args.word_num)
valid_dataset = FakeNewsDataset(df_valid, args.crop_num, crop_features, word_features, args.word_num)
test_dataset = FakeNewsDataset(df_test, args.crop_num, crop_features, word_features, args.word_num)

save_dir = save_dir + "/" + args.name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
print("train size, integer loader length: ", len(train_dataset), str((len(train_loader) - 1) * args.batch_size))
print("valid size, integer loader length: ", len(valid_dataset), str((len(valid_loader) - 1) * args.batch_size))
print("test size, integer loader length: ", len(test_dataset), str((len(test_loader) - 1) * args.batch_size))

net = C3N
model = net(args).to(args.device)

print(model)
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.requires_grad)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)


@torch.no_grad()
def compute_test(model_, loader):
    model_.eval()
    loss_test = 0.0
    out_log = []
    for data in loader:
        out = model_(data)
        out_log.append([F.softmax(out, dim=1), data['label']])
        loss_test += F.nll_loss(out, data['label'].to(args.device)).item()
    return out_log, loss_test


def train(model, optimizer, train_loader, valid_loader):
    print("Start Training!")
    best_acc_val = 0
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
        for i, data in enumerate(train_loader):
            out = model(data)
            loss = F.nll_loss(out, data['label'].to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            assert torch.isnan(loss).sum() == 0, "loss is nan!"
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), data['label']])
            global_step += 1

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
        drawing_list = [loss_list_train, loss_list_val, acc_list_train, acc_list_val, epoch_list]
        np.save(os.path.join(save_dir, 'drawing_list.npy'), drawing_list)

    print('Finished Training!')
    draw_fig_loss(loss_list_train, loss_list_val, epoch_list, save_dir)
    draw_fig_acc(acc_list_train, acc_list_val, epoch_list, save_dir)


train(model, optimizer, train_loader, valid_loader)

best_model = net(args).to(args.device)
load_checkpoint(os.path.join(save_dir, 'model.pt'), best_model, args)
print('Valid Classification Report')
out_log_val, _ = compute_test(best_model, valid_loader)
eval_classification_report(out_log_val, do_print=True)

print('Test Classification Report')
out_log_test, _ = compute_test(best_model, test_loader)
out = eval_classification_report(out_log_test, do_print=True)

if not os.path.exists(os.path.join(save_dir, 'Exp1_out.csv')):
    create_out_csv(save_dir, 'Exp1')
append_out_csv(save_dir, 'Exp1', eval_classification_report(out_log_test), args)

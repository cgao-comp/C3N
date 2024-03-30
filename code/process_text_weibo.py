import os
import pickle
import pandas as pd
import numpy as np
import torch
import re
from PIL import Image
import cn_clip.clip as clip
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

data_dir = "/sda/qiaojiao/code/Weibo16/row"
processed_dir = "/sda/qiaojiao/code/Weibo16/processed"
big_processed_dir = "/sda/qiaojiao/code/Weibo16/processed"

CLIP_MODEL_NAME = "ViT-B-16"


def read_image():
    """ check image whether to open
    """
    image_list = {}
    file_list = [data_dir + '/nonrumor_images/', data_dir + '/rumor_images/']
    for path in file_list:
        for i, filename in enumerate(os.listdir(path)):  # assuming gif
            try:
                im = Image.open(path + filename).convert('RGB')
                image_list[filename.split('/')[-1].split(".")[0]] = True
            except:
                print(filename)
    print("image length " + str(len(image_list)))
    return image_list


def clean_str_sst(text):
    """
    Tokenization/string cleaning for the SST dataset
    """
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&nbsp;', '', text)
    text = re.sub(r'&quot', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data():
    """ load data from .txt to dataframe
    """
    pre_path = data_dir + "/tweets/"
    file_list = [pre_path + "test_nonrumor.txt", pre_path + "test_rumor.txt",
                 pre_path + "train_nonrumor.txt", pre_path + "train_rumor.txt"]

    data = []
    column = ['post_id', 'image_id', 'original_post', 'label']
    for k, f in enumerate(file_list):
        f = open(f, 'rb')
        if (k + 1) % 2 == 1:
            label = 0
        else:
            label = 1

        for i, l in enumerate(f.readlines()):
            if (i + 1) % 3 == 1:
                line_data = []
                post_id = l.decode().split('|')[0]
                line_data.append(post_id)

            if (i + 1) % 3 == 2:
                line_data.append(l)

            if (i + 1) % 3 == 0:
                l_original = clean_str_sst(str(l, "utf-8"), False)
                line_data.append(l_original)
                line_data.append(label)
                data.append(line_data)

        f.close()

    data_text_df = pd.DataFrame(np.array(data, dtype=object), columns=column)
    np.save(processed_dir + "/row_data_df.npy", data_text_df)
    print("row_data_df length:" + str(data_text_df.shape[0]))
    return data_text_df


def paired(image, post):
    """ each tweet matchs one image
        output: a dataframe for text:
    """
    ordered_text = []
    label = []
    post_id = []
    image_id_list = []

    no_image = 0
    for i, id in enumerate(post['post_id']):
        have_image = False
        for image_id in post.iloc[i]['image_id'].decode().split('|'):
            image_name = image_id.split("/")[-1].split(".")[0]
            if image_name in image:
                have_image = True
                image_id_list.append(image_name)
                ordered_text.append(post.iloc[i]['original_post'])
                post_id.append(id)
                label.append(post.iloc[i]['label'])
                print(str(i))
                break
        if not have_image:
            no_image += 1

    print("The number of no images:" + str(no_image))
    label = np.array(label, dtype=np.int32)
    print("Label number is " + str(len(label)))
    print("Rummor number is " + str(sum(label)))
    print("Non rummor is " + str(len(label) - sum(label)))

    data = {"post_id": post_id,
            "image_id": image_id_list,
            "original_post": np.array(ordered_text, dtype=object),
            "label": label}
    data_df = pd.DataFrame(data)
    print("data size is " + str(data_df.shape[0]))
    np.save(processed_dir + '/data_df.npy', data_df)

    return data_df


def split_EANN_simple(df_data):
    """ split data according EANN
        df_data: paired() output
    """
    id_test = pickle.load(open("/home/qiaojiao/Code/Datasets/Weibo16_Full/test_id.pickle", 'rb'))
    id_train = pickle.load(open("/home/qiaojiao/Code/Datasets/Weibo16_Full/train_id.pickle", 'rb'))
    id_valid = pickle.load(open("/home/qiaojiao/Code/Datasets/Weibo16_Full/validate_id.pickle", 'rb'))
    test = pd.DataFrame(None, columns=df_data.columns)
    train = pd.DataFrame(None, columns=df_data.columns)
    valid = pd.DataFrame(None, columns=df_data.columns)
    device = "cuda:1"
    model, _ = clip.load_from_name("ViT-B-16", device=device, download_root='/home/qiaojiao/Code/')
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    for i, row in df_data.iterrows():
        print(i, '/', len(df_data))
        new = pd.DataFrame([{'post_id': row['post_id'], 'original_post':row['original_post'], 'image_id':row['image_id'],
                           'label':row['label']}], columns=df_data.columns)
        if row['post_id'] in id_test:
            test = pd.concat([test, new], axis=0, ignore_index=True, sort=False)
        elif row['post_id'] in id_train:
            train = pd.concat([train, new], axis=0, ignore_index=True, sort=False)
        elif row['post_id'] in id_valid:
            valid = pd.concat([valid, new], axis=0, ignore_index=True, sort=False)
        # if mode == 'clip':
        #     word_feature = process(row['original_post'], model.float(), device)
        # word_features[row['post_id']] = word_feature

    # np.save(big_processed_dir + '/word_clipinputs.npy', word_features)
    all_data = pd.concat([test, valid, train], axis=0, ignore_index=True, sort=False)
    train.to_csv(processed_dir + '/train_EANN_frozen.csv', index=False)
    valid.to_csv(processed_dir + '/valid_EANN_frozen.csv', index=False)
    test.to_csv(processed_dir + '/test_EANN_frozen.csv', index=False)
    np.save(processed_dir + '/all_data_EANN_frozen.npy', all_data)
    np.save(processed_dir + '/train_EANN_frozen.npy', train)
    np.save(processed_dir + '/valid_EANN_frozen.npy', valid)
    np.save(processed_dir + '/test_EANN_frozen.npy', test)
    print("train: ", len(train))
    print("valid: ", len(valid))
    print("test: ", len(test))
    print("total: ", len(train) + len(valid) + len(test))
    

def remove_punctuation(text):
    pattern = r"\w{3}\s\w{3}\s\d{2}\s\d{2}:\d{2}:\d{2}\s\+\d{4}\s\d{4}"
    text = re.sub(pattern, '', text)
    pattern = r'[\n\\]|&quot'
    
    return re.sub(pattern, '', text)


def save_text_input(df):
    n_words_dic = np.load(processed_dir + '/n_words.npy', allow_pickle=True).item()
    n_token_dic = {}
    text_input_dic = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        post_id = row['post_id']
        original_post = remove_punctuation(row['original_post'])
        n_words = n_words_dic[post_id]
        text_input = clip.tokenize(original_post, context_length=200).squeeze(0)
        n_token = clip.tokenize(n_words, context_length=20)
        n_token_dic[post_id] = n_token
        text_input_dic[post_id] = text_input
        
    np.save(processed_dir + '/n_tokens.npy', n_token_dic)
    np.save(processed_dir + '/word_clipinputs.npy', text_input_dic)
    

if __name__ == "__main__":

    image_list = read_image()
    np.save(big_processed_dir + '/image_list_origin.npy', image_list)
    image_list = np.load(big_processed_dir + '/image_list_origin.npy', allow_pickle=True).item()

    data_text_df = load_data()
    data_text_df = np.load(processed_dir + "/data_text_df.npy", allow_pickle=True)
    column = ['post_id', 'image_id', 'original_post', 'sentences', 'label']
    data_text_df = pd.DataFrame(data_text_df, columns=column)

    data_df = paired(image_list, data_text_df)

    all_data = np.load(processed_dir + '/all_data_EANN.npy', allow_pickle=True)
    column = ['original_post', 'clip_tokens', 'image', 'label', 'image_id', 'post_id']
    all_data = pd.DataFrame(all_data, columns=column)
    split_EANN_simple(all_data[['original_post', 'label', 'image_id', 'post_id']])
    
    df_train = np.load(processed_dir + "/train_EANN_frozen.npy", allow_pickle=True)
    df_valid = np.load(processed_dir + "/valid_EANN_frozen.npy", allow_pickle=True)
    df_test = np.load(processed_dir + "/test_EANN_frozen.npy", allow_pickle=True)
    df_columns = ['original_post', 'label', 'image_id', 'post_id']
    df_train = pd.DataFrame(df_train, columns=df_columns)
    df_valid = pd.DataFrame(df_valid, columns=df_columns)
    df_test = pd.DataFrame(df_test, columns=df_columns)
    all_data_df = pd.concat([df_train, df_valid, df_test], axis=0, ignore_index=True, sort=False)
    save_text_input(all_data_df)
    

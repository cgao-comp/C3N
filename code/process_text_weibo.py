import os
import pickle
import re
import pandas as pd
import numpy as np
from PIL import Image
import cn_clip.clip as clip

data_dir = "../data/weibo/row"
processed_dir = "../data/weibo/processed"
big_processed_dir = "It is recommended to use a special directory to store large files."
CLIP_MODEL_NAME = "ViT-B-16"


def read_image():
    """ check image whether to open
    """
    image_list = {}
    file_list = [data_dir + '/nonrumor_images/', data_dir + '/rumor_images/']
    for path in file_list:
        for i, filename in enumerate(os.listdir(path)):
            try:
                im = Image.open(path + filename).convert('RGB')
                if re.findall('\.gif', filename):
                    print(filename)
                    continue
                image_list[filename.split('/')[-1].split(".")[0]] = True
            except:
                print(filename)
    print("image length " + str(len(image_list)))
    return image_list


def clean_str_sst(text, sentences=True):
    """
    Tokenization/string cleaning for the SST dataset
    """
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&nbsp;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if sentences:
        text = re.sub(r'【', '', text)
        text = text.strip('。+||！|\!|……|？|\?|】')
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


def text_preprocess(sentence):
    """ preprocess to return tokens for batch samples
        input: a list of 15 sub-sentences
    """
    clip_inputs = clip.tokenize(sentence)
    return clip_inputs


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


def split_process_data(df_data, process):
    """ spli data according benchmark
        df_data: paired() output
    """
    id_test = pickle.load(open(data_dir + "/test_id.pickle", 'rb'))
    id_train = pickle.load(open(data_dir + "/train_id.pickle", 'rb'))
    id_valid = pickle.load(open(data_dir + "/validate_id.pickle", 'rb'))
    test = pd.DataFrame(None, columns=df_data.columns)
    train = pd.DataFrame(None, columns=df_data.columns)
    valid = pd.DataFrame(None, columns=df_data.columns)
    device = "cuda:1"
    model, _ = clip.load_from_name("ViT-B-16", device=device, download_root='./')
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    word_inputs_dic = {}

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

        word_inputs = process(row['original_post'], model.float(), device)
        word_inputs_dic[row['post_id']] = word_inputs

    np.save(big_processed_dir + '/word_clipinputs.npy', word_inputs_dic)
    np.save(processed_dir + '/train.npy', train)
    np.save(processed_dir + '/valid.npy', valid)
    np.save(processed_dir + '/test.npy', test)
    print("train: ", len(train))
    print("valid: ", len(valid))
    print("test: ", len(test))
    print("total: ", len(train) + len(valid) + len(test))


def get_word_inputs(text, model, device):
    """ clip inputs, text -> [seq_length, context_length]
    """
    tokens = clip.tokenize(text, context_length=200)
    return tokens.squeeze(0)


if __name__ == "__main__":

    image_list = read_image()
    np.save(big_processed_dir + '/image_list.npy', image_list)
    # image_list = np.load(big_processed_dir + '/image_list.npy', allow_pickle=True).item()

    data_text_df = load_data()
    # data_text_df = np.load(processed_dir + "/row_data_df.npy", allow_pickle=True)
    # column = ['post_id', 'image_id', 'original_post', 'label']
    # data_text_df = pd.DataFrame(data_text_df, columns=column)

    data_df = paired(image_list, data_text_df)

    # data_df = np.load(processed_dir + '/data_df.npy', allow_pickle=True)
    # data_df = pd.DataFrame(data_df, columns=column)
    split_process_data(data_df, get_word_inputs)

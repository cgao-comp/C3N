from PIL import Image
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from cn_clip.clip import load_from_name
import re


data_dir = "../data/weibo/row"
processed_dir = "../data/weibo/processed"
big_processed_dir = "It is recommended to use a special directory to store large files."
processed_img_dir = "../data/weibo/processed/crops"
device = "cuda:1"
CROP_NUM = 5


def makedir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def clip_image_preprocess_reserve(df, preprocess):
    """ preprocess crops for CLIP, df: note the label, padding to crop_num
    """
    image_dic = {}
    error_origin_count = 0
    error_crop_count = 0
    for i, path in enumerate(os.listdir(processed_img_dir)):
        print(i)
        crop_list = []
        label = df.loc[df['image_id'] == path]['label']
        if len(label) == 0:
            print("can not find: ", path)
            continue
        if label.iloc[0] == 0:
            imgdir = data_dir + '/nonrumor_images/'
        else:
            imgdir = data_dir + '/rumor_images/'
        try:
            img = preprocess(Image.open(imgdir + path + ".jpg").convert('RGB'))
            crop_list.append(img)
            file_list = os.listdir(os.path.join(processed_img_dir, path))
            i = 0
            for j, filename in enumerate(file_list):
                if re.findall('_object', filename):
                    continue
                try:
                    image = preprocess(Image.open(os.path.join(processed_img_dir, path, filename)).convert('RGB'))
                    crop_list.append(image)
                    i += 1
                except:
                    print("error file: ", filename)
                    error_crop_count += 1
            while i < CROP_NUM:
                image = torch.zeros(3, 224, 224)
                crop_list.append(image)
                i += 1
            if i > CROP_NUM:
                crop_list = crop_list[:CROP_NUM + 1]

            crop_inputs = torch.tensor(np.stack(crop_list))
            image_dic[path] = crop_inputs
            print("crop length ", str(len(crop_list) - 1))
        except:
            print("error origin file: ", imgdir + path + ".jpg")
            error_origin_count += 1

    print("image length", str(len(image_dic)))
    print("error origin image: ", error_origin_count)
    print("error crop count: ", error_crop_count)
    np.save(big_processed_dir + "/clip_crop_preprocess.npy", image_dic)
    return image_dic


def drop_in_text(image_dic, train, test, valid):
    """ drop samples again in text-preprocess results according to image preprocess.
    """
    test_new = pd.DataFrame(None, columns=train.columns)
    train_new = pd.DataFrame(None, columns=train.columns)
    valid_new = pd.DataFrame(None, columns=train.columns)
    new_list = [train_new, test_new, valid_new]
    names = ['train','valid', 'test']
    for name, df in enumerate([train, valid, test]):
        for i, row in df.iterrows():
            new = pd.DataFrame([{'post_id': row['post_id'], 'image_id':row['image_id'], 'original_post':row['original_post'], 'label':row['label']}], columns=df.columns)
            if row['image_id'] in image_dic:
                new_list[name] = pd.concat([new_list[name], new], ignore_index=True, sort=False)
        np.save(processed_dir + '/' + names[name] + '.npy', new_list[name])
        print(names[name], len(new_list[name]))


if __name__ == '__main__':

    data_df = np.load(processed_dir + '/data_df.npy', allow_pickle=True)
    column = ['post_id', 'image_id', 'original_post', 'label']
    data_df = pd.DataFrame(data_df, columns=column)

    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    model.float()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    image_dic = clip_image_preprocess_reserve(data_df[['label', 'image_id']], preprocess)
    
    # image_dic = np.load(big_processed_dir + "/clip_crop_preprocess.npy", allow_pickle=True).item()

    df_train = np.load(processed_dir + "/train.npy", allow_pickle=True)
    df_valid = np.load(processed_dir + "/valid.npy", allow_pickle=True)
    df_test = np.load(processed_dir + "/test.npy", allow_pickle=True)
    df_train = pd.DataFrame(df_train, columns=column)
    df_valid = pd.DataFrame(df_valid, columns=column)
    df_test = pd.DataFrame(df_test, columns=column)
    drop_in_text(image_dic, df_train, df_valid, df_test)

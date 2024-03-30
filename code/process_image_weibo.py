from PIL import Image
import os
import numpy as np
import torchvision
import torchvision.transforms as T
import torch
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from cn_clip.clip import load_from_name, available_models
import re
from tqdm import tqdm
import setproctitle
setproctitle.setproctitle('qiaojiao')

data_dir = "/sda/qiaojiao/code/Weibo16/row"
processed_dir = "/sda/qiaojiao/code/Weibo16/processed"
big_processed_dir = "/sda/qiaojiao/code/Weibo16/processed"
processed_img_dir = "/sda/qiaojiao/code/Weibo16/processed/crops"


device = "cuda:1"
CROP_NUM = 5

def makedir(path):
    """ 创建新文件夹, imwrite无法写入未创建的文件夹
    """
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def clip_image_preprocess(df, preprocess):
    """ preprocess crops for CLIP, df: note the label, padding to crop_num
    """
    image_dic = {}
    error_origin_count = 0
    error_crop_count = 0
    for i, path in enumerate(os.listdir(processed_img_dir)):
        print(i)
        crop_list = []
        # first is origin image
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
        except Exception as e:
            print("error origin file: ", imgdir + path + ".jpg")
            error_origin_count += 1
            print(e)

    print("image length", str(len(image_dic)))
    print("error origin image: ", error_origin_count)
    print("error crop count: ", error_crop_count)
    np.save(big_processed_dir + "/clip_image_preprocess.npy", image_dic)  # clip_image_preprocess

def cut_image(image, num):
    """ cut image for patches
    """
    width, height = image.size
    item_width = int(width / num)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, num):
        for j in range(0, num):
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)
    image_list = [np.array(image.crop(box)) for box in box_list]
    return image_list


def image_patch_preprocess(df, preprocess, transform):
    """ save image patch inputs
    """
    image_inputs_dic = {}
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # print(index, '/', len(df))
        image_id = row['image_id']
        path = os.path.join("/sda/qiaojiao/code/Weibo16/processed/Weibo16_images_EANNSplit", image_id + '.jpg')
        image = Image.open(path).convert('RGB')
        ori_image = preprocess(image)
        image = image.resize((640, 640), Image.ANTIALIAS)
        image_list = cut_image(image, 4)  # crop_num
        images = torch.tensor(image_list)  # [patch_num, h, w, 3]
        images = images.permute(0, 3, 1, 2).float()
        images = transform(images)
        images = torch.cat([ori_image.unsqueeze(0), images], dim=0)
        image_inputs_dic[image_id] = images
    np.save(big_processed_dir + "/clip_patch_preprocess_16.npy", image_inputs_dic)


if __name__ == '__main__':
    
    df_train = np.load(processed_dir + "/train_EANN_frozen.npy", allow_pickle=True)
    df_valid = np.load(processed_dir + "/valid_EANN_frozen.npy", allow_pickle=True)
    df_test = np.load(processed_dir + "/test_EANN_frozen.npy", allow_pickle=True)
    df_columns = ['original_post', 'label', 'image_id', 'post_id']
    df_train = pd.DataFrame(df_train, columns=df_columns)
    df_valid = pd.DataFrame(df_valid, columns=df_columns)
    df_test = pd.DataFrame(df_test, columns=df_columns)

    all_data_df = pd.concat([df_train, df_valid, df_test], axis=0, ignore_index=True, sort=False)

    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='/sda/qiaojiao/pretrained_models/cn-clip/')
    model.float()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    clip_image_preprocess(all_data_df[['label', 'image_id']], preprocess)
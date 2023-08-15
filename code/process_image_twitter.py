from PIL import Image
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from cn_clip.clip import load_from_name
import re

processed_dir = "./twitter/processed"
big_processed_dir = "It is recommended to use a special directory to store large files."
processed_img_dir = "./twitter/processed/crops"

device = "cuda:0"
CROP_NUM = 5   


def imagecrop_preprocess(df, preprocess):
    image_dic = {}
    error_origin_count = 0
    error_crop_count = 0
    for i, path in enumerate(os.listdir(processed_img_dir)):
        print(i)
        crop_list = []
        image_path = df.loc[df['image_id'] == path]['imagepath'].iloc[0]
        try:
            img = preprocess(Image.open(image_path).convert('RGB'))
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
            print("error origin file: ", image_path)
            error_origin_count += 1

    print("image length", str(len(image_dic)))
    print("error origin image: ", error_origin_count)
    print("error crop count: ", error_crop_count)
    np.save(big_processed_dir + "/clip_crop_preprocess.npy", image_dic)


if __name__ == '__main__':

    all_data_df = np.load(processed_dir + '/all.npy', allow_pickle=True)
    column = ['post_id', 'original_post', 'image_id', 'label', 'event', 'imagepath']
    all_data_df = pd.DataFrame(all_data_df, columns=column)

    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
    model = model.float()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    imagecrop_preprocess(all_data_df[['label', 'image_id', 'imagepath']], preprocess)


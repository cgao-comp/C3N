import os
import re
import pandas as pd
import numpy as np
from PIL import Image
import csv
import clip

data_dir = "./twitter/row/image-verification-corpus"
dev_pth = "./twitter/row/image-verification-corpus/mediaeval2016/devset/posts.txt"
test_pth = "./twitter/row/image-verification-corpus/mediaeval2016/testset/posts_groundtruth.txt"
processed_dir = "./twitter/processed"
tweet_img_pth = "./twitter/row/image-verification-corpus/tweets_images.txt"
MAX_LENGTH = 4


def clean_str_sst(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'https\S+', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&nbsp;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_image_id(image_id):
    image_id = re.sub(r'\s+', ' ', image_id).strip().strip('"').split(",")[0]
    return image_id


def load_data(file, df):
    """ load data from .txt to dataframe
    """
    column = ['post_id', 'original_post', 'image_id', 'label', 'event']
    data = []
    non_en_count = 0
    not_find_count = 0
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)
        for row in reader:
            image_id = df.loc[df['post_id'] == row[0]]['image_id']
            event = df.loc[df['post_id'] == row[0]]['event']
            if len(image_id) == 0 or len(event) == 0:
                not_find_count += 1
                print("can not find", row[0])
                image_id = row[4]  # test's columns are different from dev
                # image_id = row[3] # dev
                event = None
            else:
                image_id = image_id.iloc[0]
                event = event.iloc[0]
            line_data = []
            line_data.append(row[0])
            line_data.append(clean_str_sst(row[1]))
            line_data.append(clean_image_id(image_id))
            line_data.append(1 if row[6] == 'fake' else 0)
            line_data.append(event)
            data.append(line_data)

    data_text_df = pd.DataFrame(np.array(data, dtype=object), columns=column)
    np.save(processed_dir + "/row_test_df.npy", data_text_df)
    print("data_text_df length:" + str(data_text_df.shape[0]))
    print("non en sample: ", non_en_count)
    print("not find: ", not_find_count)
    return data_text_df


def load_tweet_img(file):
    column = ['post_id', 'image_id', 'label', 'event']
    data = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            line_data = []
            line_data.append(row[0])
            line_data.append(row[1])
            line_data.append(1 if row[-2] == 'fake' else 0)
            line_data.append(row[-1])
            data.append(line_data)

    data_text_df = pd.DataFrame(np.array(data, dtype=object), columns=column)
    np.save(processed_dir + "/tweet_image.npy", data_text_df)
    print("tweet_image length:" + str(data_text_df.shape[0]))
    return data_text_df


def splitdev_checkimage(df):
    """ split dev according to 2015devset/2015testset file structure, and check if the image available
    """
    file_list = [data_dir + '/mediaeval2015/devset/Medieval2015_DevSet_Images/', data_dir + '/mediaeval2015/testset/TestSetImages/']
    new_colunmns = df.columns.values.tolist() + ['imagepath']
    split_list = ['train', 'valid']
    train = pd.DataFrame(None, columns=new_colunmns)
    valid = pd.DataFrame(None, columns=new_colunmns)
    form_list = ['.jpg', '.png', '.jpeg']

    for i, row in df.iterrows():
        imgpth = False
        for j, path0 in enumerate(file_list):
            for path1 in os.listdir(path0):
                path2_list = os.listdir(os.path.join(path0, path1))
                if os.path.isdir(os.path.join(path0, path1, path2_list[0])):
                    for filename in path2_list:
                        pth = os.path.join(path0, path1, filename, row['image_id'])
                        imgpth_ = False
                        for form in form_list:
                            p = pth + form
                            if os.path.exists(p):
                                imgpth_ = p
                                break
                            p = pth
                        if not imgpth_:
                            continue
                        try:
                            Image.open(imgpth_).convert('RGB')
                            imgpth = imgpth_
                            split = split_list[j]
                            break
                        except:
                            print('can not open: ', row['image_id'])

                else:
                    pth = os.path.join(path0, path1, row['image_id'])
                    imgpth_ = False
                    for form in form_list:
                        p = pth + form
                        if os.path.exists(p):
                            imgpth_ = p
                            break
                        p = pth
                    if not imgpth_:
                        continue
                    try:
                        Image.open(imgpth_).convert('RGB')
                        imgpth = imgpth_
                        split = split_list[j]
                        break
                    except:
                        print('can not open: ', row['image_id'])
                if imgpth:
                    break
            if imgpth:
                break
        if imgpth:
            new = pd.DataFrame([{'post_id': row['post_id'], 'original_post':row['original_post'], 'image_id':row['image_id'],
                                 'label':row['label'], 'event':row['event'], 'imagepath':imgpth}], columns=new_colunmns)
            if split == 'train':
                train = pd.concat([train, new], axis=0, ignore_index=True, sort=False)
            elif split == 'valid':
                valid = pd.concat([valid, new], axis=0, ignore_index=True, sort=False)
        else:
            print('can not find: ', row['post_id'], ' ', row['image_id'])

    dev = pd.concat([train, valid], axis=0, ignore_index=True, sort=False)
    
    np.save(processed_dir + '/dev.npy', dev)
    print("devset total: ", len(dev))


def splittest_checkimage(df):
    """ split test according to 2016testset file structure, and check if the image available
    """
    file_list = [data_dir + '/mediaeval2016/testset/Mediaeval2016_TestSet_Images/']
    new_colunmns = df.columns.values.tolist() + ['imagepath']
    form_list = ['.jpg', '.png', '.jpeg']
    test = pd.DataFrame(None, columns=new_colunmns)

    for i, row in df.iterrows():
        imgpth = False
        for j, path0 in enumerate(file_list):
            pth = os.path.join(path0, row['image_id'])
            imgpth_ = False
            for form in form_list:
                p = pth + form
                if os.path.exists(p):
                    imgpth_ = p
                    break
                p = pth
            if not imgpth_:
                print('can not find: ', row['post_id'], ' ', row['image_id'])
            else:
                try:
                    Image.open(imgpth_).convert('RGB')
                    imgpth = imgpth_
                    new = pd.DataFrame([{'post_id': row['post_id'], 'original_post':row['original_post'], 'image_id':row['image_id'],
                                        'label':row['label'], 'event':row['event'], 'imagepath':imgpth}], columns=new_colunmns)
                    test = pd.concat([test, new], axis=0, ignore_index=True, sort=False)
                    break
                except:
                    print('can not open: ', row['post_id'], row['image_id'])

    np.save(processed_dir + '/test.npy', test)
    print("test: ", len(test))


def text_preprocess(sentence):
    """ preprocess to return tokens for batch samples
        input: a list of MAX_LENGTH sub-sentences
    """
    clip_inputs = clip.tokenize(sentence, truncate=True)  # default: max_length=77
    return clip_inputs


def get_wordfeatures(train, test, mode='feature'):
    """ get clip word features dictionary, [seq, dim]
    """
    device = 'cuda:1'
    model, preprocess = clip.load('ViT-B/16', device=device)
    model = model.float()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    word_clipfeatures = {}
    total = len(train)  + len(test)
    for i, df in enumerate([train, test]):
        print(i, '/', total)
        for index, row in df.iterrows():
            if mode == 'feature':
                word_clipfeatures[row['post_id']] = model.my_encode_text(clip.tokenize(row['original_post'], truncate=True).to(device)).squeeze(0)
            else:
                word_clipfeatures[row['post_id']] = clip.tokenize(row['original_post'], truncate=True).to(device).squeeze(0)

    if mode == 'feature':
        np.save(processed_dir + '/word_clipfeatures', word_clipfeatures)
    else:
        np.save(processed_dir + '/word_clipinputs', word_clipfeatures)
    print(len(train))
    print(len(test))
    print(len(train) +  len(test))
    

if __name__ == '__main__':
    tweet_img_df = load_tweet_img(tweet_img_pth)
    load_data(dev_pth, tweet_img_df)
    load_data(test_pth, tweet_img_df)

    column = ['post_id', 'original_post', 'image_id', 'label', 'event']
    df_dev = np.load(processed_dir + '/row_dev_df_dev.npy', allow_pickle=True)
    df_dev = pd.DataFrame(df_dev, columns=column)
    df_test = np.load(processed_dir + '/row_test_df.npy', allow_pickle=True)
    df_test = pd.DataFrame(df_test, columns=column)

    splitdev_checkimage(df_dev)
    splittest_checkimage(df_test)

    train = np.load(processed_dir + "/dev.npy", allow_pickle=True) 
    test = np.load(processed_dir + "/test.npy", allow_pickle=True)  
    column = ['post_id', 'original_post', 'image_id', 'label', 'event', 'imagepath']
    train = pd.DataFrame(train, columns=column)
    test = pd.DataFrame(test, columns=column)
    get_wordfeatures(train, test, mode='feature')
    all = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
    np.save(processed_dir + '/all.npy', all)

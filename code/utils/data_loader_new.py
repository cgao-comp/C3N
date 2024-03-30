import torch
from torch.utils.data import Dataset

data_path_1 = '/sda/qiaojiao/code/Weibo16/row/'
data_path_2 = '/sda/qiaojiao/code/Mediaeval_original/image-verification-corpus/'
twitter_crop_dir = "/sda/qiaojiao/code/Mediaeval2016/processed/crops"
weibo_crop_dir = "/sda/qiaojiao/code/Weibo16/processed/crops"


class FakeNewsDataset(Dataset):
    def __init__(self, data_df, crop_num, st_num, dataset, n_words, crop_input, text_input):
        self.data_df = data_df
        self.crop_num = crop_num
        self.st_num = st_num
        self.dataset = dataset
        if dataset == 'weibo':
            self.n_words = n_words
            self.crop_input = crop_input
            self.text_input = text_input
        else:
            self.n_words = n_words
            self.crop_input = crop_input
            self.text_input = text_input

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()      
        post_id = self.data_df['post_id'][idx]
        image_id = self.data_df['image_id'][idx]
        label = self.data_df['label'][idx]        
        label = torch.tensor(label)
        n_words = self.n_words[post_id]
        crop_input = self.crop_input[image_id]
        text_input = self.text_input[post_id]
        
        sample = {
            'post_id': post_id,
            'label': label,
            'crop_input': crop_input,
            'n_word_input': n_words,
            'text_input': text_input,
        }
        return sample
    
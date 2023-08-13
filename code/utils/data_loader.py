import torch
from torch.utils.data import Dataset


class FakeNewsDataset(Dataset):
    def __init__(self, data_df, crop_num, crop_features, word_features, word_num):
        self.data_df = data_df
        self.crop_num = crop_num
        self.word_num = word_num
        self.crop_features = crop_features
        self.word_features = word_features

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()      
        post_id = self.data_df['post_id'][idx]
        image_id = self.data_df['image_id'][idx]
        crop_features = self.crop_features[image_id]
        word_features = self.word_features[post_id]
        label = self.data_df['label'][idx]
        label = torch.tensor(label)

        sample = {
            'crop_features': crop_features,
            'word_features': word_features,
            'label': label,
            'post_id': post_id
        }
        return sample

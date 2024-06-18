import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import clip

class Cifar(Dataset):
    def __init__(self, data_dir_root, label_list, preprocess, is_train=False, is_test=False, is_val=False, device='cpu'):
        
        self.label_list = np.array(label_list)
        self.preprocess = preprocess
        self.device = device
        self.data = list()

        if is_train:
            data_dir_root = os.path.join(data_dir_root, 'train')
            print(f'Cifar Train')
        elif is_val:
            data_dir_root = os.path.join(data_dir_root, 'val')
            print(f'Cifar Val')
        elif is_test:
            data_dir_root = os.path.join(data_dir_root, 'test')
            print(f'Cifar test')
        
        for label_name in os.listdir(data_dir_root):
            if label_name not in label_list:
                continue

            label_root = os.path.join(data_dir_root, label_name)
            print(f'label {label_name} has {len(os.listdir(label_root))} imgs.')
            for img in os.listdir(label_root):
                self.data.append(os.path.join(label_root, img))

            

    def __getitem__(self, idx):
        data_path = self.data[idx]
        sample = self.preprocess(Image.open(data_path))
        text = data_path.split('/')[-2]
        # text_tokens = clip.tokenize([text])
        # label = np.where(data_path.split('/')[-2] == self.label_list)[0]
        # label = np.zeros_like(self.label_list, dtype=int)
        # label[label_pos] = 1

        return sample, text

    def __len__(self):
        return len(self.data)

import os
import sys
import logging

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from densebody_poc.utils.constants import JOB_CONF_KEY, DATA_DIR_KEY, PHASE_KEY, \
MAX_DATASET_SIZE_KEY, PREDICT, IMAGE_EXTENSIONS, IM_NAMES


class DenseBodyDataset(Dataset):
    '''
    Dataset class to contain relevant data use for various modes of operations
    '''
    def __init__(self, conf):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.transform = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.data_dir = conf[JOB_CONF_KEY][DATA_DIR_KEY]
        try:
            self.im_names = [
                im_name for im_name in self._get_image_names(self.data_dir) \
                if im_name.split('.')[-1].lower() in IMAGE_EXTENSIONS
            ]
            cur_length = len(self.im_names)
            if cur_length == 0:
                raise FileNotFoundError
        except FileNotFoundError:
            logging.error('No input images found in provided data directory "{}",'\
                .format(self.data_dir) + ' exiting program')
            sys.exit()
        
        if conf[JOB_CONF_KEY][PHASE_KEY] == PREDICT:
            max_dataset_size = conf[JOB_CONF_KEY][MAX_DATASET_SIZE_KEY]
            if 0 < max_dataset_size < cur_length:
                self.im_names = self.im_names[:max_dataset_size]
            self.length = len(self.im_names)
            self.itemtypes = [IM_NAMES]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        out_dict = {}
        for itemtype in self.itemtypes:
            item = getattr(self, itemtype)[idx]
            if itemtype.endswith('names'):
                img_tensor = self.transform(cv2.imread(item))
                out_dict[itemtype.replace('names','data')] = img_tensor.to(self.device)
            else:
                out_dict[itemtype] = torch.from_numpy(item)
        return out_dict

    @staticmethod
    def _get_image_names(root):
        image_names = [root + '/' + name for name in os.listdir(root) 
            if name.split('.')[-1].lower() in IMAGE_EXTENSIONS]
        subs = [sub for sub in os.listdir(root) if os.path.isdir(sub)]
        for sub in subs:
            full_sub = root + '/' + sub
            image_names += [
                full_sub + '/' + name for name in os.listdir(full_sub)
                if name.split('.')[-1].lower() in IMAGE_EXTENSIONS
            ]
        return image_names

import os

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
            # Image.fromarray,
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
            raise FileNotFoundError('No input images found in provided data directory "{}",' \
                                    .format(self.data_dir))

        if conf[JOB_CONF_KEY][PHASE_KEY] == PREDICT:
            max_dataset_size = conf[JOB_CONF_KEY][MAX_DATASET_SIZE_KEY]
            if 0 < max_dataset_size < cur_length:
                self.im_names = self.im_names[:max_dataset_size]
            self.length = len(self.im_names)
            self.itemtypes = [IM_NAMES]

    def __len__(self):
        '''
        Mandatory override of Dataset base class method, to get length of dataset
        '''
        return self.length

    def __getitem__(self, idx):
        '''
        Mandatory override of Dataset base class method, to get item in dataset
        TODO: item should contain a batch of images to stack
        '''
        if isinstance(idx, slice):
            start_idx, stop_idx, step_idx = idx.indices(len(self))
        elif isinstance(idx, int):
            start_idx, stop_idx, step_idx = idx, idx + 1, None
        else:
            raise TypeError('Invalid index passed: {}'.format(idx))
        out_dict = {}
        for itemtype in self.itemtypes:
            items = getattr(self, itemtype)[start_idx:stop_idx:step_idx]
            if itemtype.endswith('names'):
                try:
                    img_tensors = [self.transform(cv2.imread(item)) for item in items]
                except TypeError as e:
                    raise TypeError(str(e) + '\nFilename: {}'.format(items))
                out_dict[itemtype.replace('names', 'data')] = torch.stack(img_tensors).to(self.device)
            else:
                out_dict[itemtype.replace('names', 'data')] = torch.from_numpy(items).to(self.device)
        return out_dict

    @staticmethod
    def _get_image_names(root):
        '''
        Internal method to get names of all images within a root directory, 
        including sub-directories
        '''
        # TODO: Take into account potential dup images by .ipynb_checkpoints
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

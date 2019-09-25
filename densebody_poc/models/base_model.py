import os
import logging
import torch

from densebody_poc.utils.constants import TRAIN, IM_DATA, PREDICT


class BaseModel:
    """
    Base class which further classes can extend to implement specific details
    """
    def initialize(self, conf):
        self.conf = conf
        self.isTrain = conf.PHASE == TRAIN
        self.device = torch.device(conf.DEVICE)
        self.save_dir = os.path.join(conf.CHECKPOINT_DIR, conf.NAME)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def setup(self, conf):
        if self.isTrain:
            pass
        if not self.isTrain or conf.CONTINUE_TRAIN:
            self.load_networks(conf.LOAD_EPOCH)
        self.print_networks(conf.VERBOSE)

    def set_input(self, data):
        logging.info('Setting input data')
        self.real_input = data[IM_DATA]
        if not self.conf.PHASE == PREDICT:
            pass

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '{}_net_{}.pth'.format(epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                state_dict = torch.load(load_path)
                net = getattr(self, name)
                net.load_state_dict(state_dict)
        print('Checkpoints loaded from epoch {}'.format(epoch))

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

import os
import logging
import torch

from densebody_poc.utils.constants import JOB_CONF_KEY, PHASE_KEY, TRAIN, CHECKPOINT_DIR_KEY, \
    NAME_KEY, OPS_CONF_KEY, CONTINUE_TRAIN_KEY, LOAD_EPOCH_KEY, VERBOSE_KEY, IM_DATA, PREDICT


class BaseModel:
    """
    Base class which further classes can extend to implement specific details
    """

    def initialize(self, conf):
        self.conf = conf
        self.isTrain = conf[JOB_CONF_KEY][PHASE_KEY] == TRAIN
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_dir = os.path.join(
            conf[JOB_CONF_KEY][CHECKPOINT_DIR_KEY],
            conf[JOB_CONF_KEY][NAME_KEY]
        )
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def setup(self, conf):

        if self.isTrain:
            pass

        if not self.isTrain or conf[OPS_CONF_KEY][CONTINUE_TRAIN_KEY]:
            self.load_networks(conf[OPS_CONF_KEY][LOAD_EPOCH_KEY])

        self.print_networks(conf[OPS_CONF_KEY][VERBOSE_KEY])

    def set_input(self, data):
        logging.info('Setting input data')
        self.real_input = data[IM_DATA]
        if not self.conf[JOB_CONF_KEY][PHASE_KEY] == PREDICT:
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

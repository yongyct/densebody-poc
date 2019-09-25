from .base_model import BaseModel
from . import networks

from densebody_poc.utils.constants import TRAIN, PREDICT


class ResNetModel(BaseModel):
    """
    Model class extending the BaseModel to implement specifics of ResNet
    """
    def initialize(self, conf):
        BaseModel.initialize(self, conf)
        self.loss_names = ['L1', 'TV']
        self.model_names = ['encoder', 'decoder']
        
        self.encoder = networks.define_encoder(
            im_size=conf.IM_SIZE, 
            nz=conf.NZ, 
            nef=conf.NCHANNELS, 
            netE=conf.NET_E,
            ndown=conf.NDOWN,
            norm=conf.NORM, 
            nl=conf.NL_E,
            init_type=conf.INIT_TYPE,
            device=self.device
        )
        
        self.decoder = networks.define_decoder(
            im_size=conf.IM_SIZE, 
            nz=conf.NZ, 
            ndf=conf.NCHANNELS, 
            netD=conf.NET_D,
            nup=conf.NUP,
            norm=conf.NORM, 
            nl=conf.NL_D,
            init_type=conf.INIT_TYPE,
            device=self.device
        )
        
        if conf.PHASE == TRAIN:
            pass
        else:
            self.encoder.eval()
            self.decoder.eval()

    def get_current_visuals(self):
        visuals = {
            'real_image': self.real_input[0],
            'fake_UV': self.fake_UV[0]
        }
        if not self.conf.PHASE == PREDICT:
            visuals['real_UV'] = self.real_UV[0]
        return visuals

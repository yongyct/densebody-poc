import logging
from .base_model import BaseModel
from . import networks

from densebody_poc.utils.constants import MODEL_CONF_KEY, MODEL_KEY, JOB_CONF_KEY,\
PHASE_KEY, TRAIN, IM_SIZE_KEY, NZ_KEY, NCHANNELS_KEY, NET_E_KEY, NET_D_KEY, NDOWN_KEY,\
NUP_KEY, NORM_KEY, NL_E_KEY, NL_D_KEY, INIT_TYPE_KEY, PREDICT


class ResNetModel(BaseModel):
    '''
    Model class extending the BaseModel to implement specifics of ResNet
    '''
    def initialize(self, conf):
        BaseModel.initialize(self, conf)
        model_conf = conf[MODEL_CONF_KEY]
        self.loss_names = ['L1', 'TV']
        self.model_names = ['encoder', 'decoder']
        
        self.encoder = networks.define_encoder(
            im_size=model_conf[IM_SIZE_KEY], 
            nz=model_conf[NZ_KEY], 
            nef=model_conf[NCHANNELS_KEY], 
            netE=model_conf[NET_E_KEY],
            ndown=model_conf[NDOWN_KEY],
            norm=model_conf[NORM_KEY], 
            nl=model_conf[NL_E_KEY],
            init_type=model_conf[INIT_TYPE_KEY],
            device=self.device
        )
        
        self.decoder = networks.define_decoder(
            im_size=model_conf[IM_SIZE_KEY], 
            nz=model_conf[NZ_KEY], 
            ndf=model_conf[NCHANNELS_KEY], 
            netD=model_conf[NET_D_KEY],
            nup=model_conf[NUP_KEY],
            norm=model_conf[NORM_KEY], 
            nl=model_conf[NL_D_KEY],
            init_type=model_conf[INIT_TYPE_KEY],
            device=self.device
        )
        
        if conf[JOB_CONF_KEY][PHASE_KEY] == TRAIN:
            pass
        else:
            self.encoder.eval()
            self.decoder.eval()

    
    def get_current_visuals(self):
        visuals = {
            'real_image': self.real_input[0],
            'fake_UV': self.fake_UV[0]
        }
        if not self.conf[JOB_CONF_KEY][PHASE_KEY] == PREDICT:
            visuals['real_UV'] = self.real_UV[0]
        return visuals

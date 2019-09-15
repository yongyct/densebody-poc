# Top level config keys
JOB_CONF_KEY = 'job_conf'
MODEL_CONF_KEY = 'model_conf'
OPS_CONF_KEY = 'ops_conf'

# 2nd level config keys - job
NAME_KEY = 'name'
DATA_DIR_KEY = 'data_dir'
CHECKPOINT_DIR_KEY = 'checkpoints_dir'
PHASE_KEY = 'phase'
MAX_DATASET_SIZE_KEY = 'max_dataset_size'
BATCH_SIZE_KEY = 'batch_size'

# 2nd level config keys - model
MODEL_KEY = 'model'
IM_SIZE_KEY = 'im_size'
NZ_KEY = 'nz'
NCHANNELS_KEY = 'nchannels'
NET_E_KEY = 'net_e'
NET_D_KEY = 'net_d'
NDOWN_KEY = 'ndown'
NUP_KEY = 'nup'
NORM_KEY = 'norm'
NL_E_KEY = 'nl_e'
NL_D_KEY = 'nl_d'
INIT_TYPE_KEY = 'init_type'

# 2nd level config keys - ops
CONTINUE_TRAIN_KEY = 'continue_train'
LOAD_EPOCH_KEY = 'load_epoch'
VERBOSE_KEY = 'verbose'

# Phases
TRAIN = 'train'
TEST = 'test'
PREDICT = 'in_the_wild'

# Others
IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
IM_NAMES = 'im_names'
IM_DATA = 'im_data'
APP_PROPS_PATH = '../resources/application_properties.json'

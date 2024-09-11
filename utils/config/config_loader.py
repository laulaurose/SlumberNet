import collections
import time
from importlib import import_module
from os import makedirs
from os.path import dirname, join, realpath
import numpy as np
import yaml
import argparse 

def update_dict(d_to_update: dict, update: dict):
    """method to update a dict with the entries of another dict

    `d_to_update` is updated by `update`"""
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            d_to_update[k] = update_dict(d_to_update.get(k, {}), v)
        else:
            d_to_update[k] = v
    return d_to_update


def parse():
    """define and parse arguments for the script"""
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--config_file', '-e', required=True,
                        help='name of experiment to run')

    return parser.parse_args()


class ConfigLoader:
    def __init__(self, path, create_dirs=True):
        
        config = self.load_config(path)

         # general
        self.DEVICE = config['general']['device']
        assert self.DEVICE in ['cpu', 'cuda'], 'DEVICE only support `cpu` or `cuda`'

        # dirs
        self.base_dir = config['dirs']['basedir']
        self.directory = config['dirs']['directory']
        self.output_directory = config['dirs']['output_directory']
        self.input_directory = config['dirs']['input_directory']
        self.DATA_DIR = config['dirs']['data']
        
        makedirs(self.base_dir, exist_ok=True)
        makedirs(self.directory, exist_ok=True)
        makedirs(self.output_directory, exist_ok=True)
        makedirs(self.input_directory, exist_ok=True)

        # data
        self.ORIGINAL_DATASET_SIZE =  config['data']['original_dataset_size']
        self.VALIDATION_SPLIT =  config['data']['validation_split']
        self.STAGES = config['data']['stages']
        self.LABS = config['data']['labs']
        self.number_of_samples_per_second = config['data']['number_of_samples_per_second']
        assert type(self.number_of_samples_per_second) in [int, float]
        self.number_seconds_per_epoch = config['data']['number_seconds_per_epoch']
        assert type(self.number_seconds_per_epoch) is int
        self.ORIGINAL_FS = config['data']['original_fs']
        self.LABS_CHANNELS = config['data']['labs_channels']
        self.SAMPLE_DURATION = config['data']['sample_duration']
        self.DATA_FILE = join(config['dirs']['cache'], config['data']['file'])
        self.DATA_FRACTION =  config['data']['data_fraction']
        # experiment
        self.set_seed = config['experiment']["set_seed"] 
        self.FIXED_N = config['experiment']["fixed_n"] 
        # training
        training_config = config['training']
        self.num_epochs = training_config['num_epochs']
        assert type(self.num_epochs) is int
        self.learning_rate = training_config['learning_rate']
        assert type(self.learning_rate) is float
        self.batch_size_per_gpu = training_config['batch_size_per_gpu']
        assert type(self.batch_size_per_gpu) is int
        
        # model
        model_config = config['model']
        self.nb_classes = model_config['nb_classes']
        assert type(self.nb_classes) is int
        
        self.n_resnet_blocks = model_config['n_resnet_blocks']

        self.n_feature_maps = model_config['n_feature_maps']
        assert type(self.n_feature_maps) is int
        
        self.kernel_expansion_fct = model_config['kernel_expansion_fct']
        assert type(self.kernel_expansion_fct) is int

        self.kernel_y = model_config['kernel_y']
        assert type(self.kernel_y) is int

        self.strides_size = model_config['strides_size']
        assert type(self.strides_size) is int

        self.dropout_rate = model_config['dropout_rate']
        assert type(self.dropout_rate) is int

        self.dropout_rate = model_config['dropout_rate']
        assert type(self.dropout_rate) is int

        self.augment_data = model_config['augment_data']
        self.model_name   = model_config['model_name']
        self.LOSS_TYPE    = model_config['loss_type']
       
    def load_config(self,path):
        """loads config from standard_config.yml and updates it with <experiment>.yml"""
        with open(path, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
        return config
    


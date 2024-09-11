import mne
import numpy as np
import tensorflow as tf
from utils.data.final_preprocessing import preprocess_EEG, preprocess_EMG
import string
import os
import scipy
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt
import h5py 
import tables 
from utils.data.data_table import COLUMN_MOUSE_ID, COLUMN_LABEL, COLUMN_LAB


class SequenceDataset(tf.keras.utils.Sequence):
    
    def __init__(self, data_folder, set ,config,test_lab):

        self.config = config
        self.augment_data = config.augment_data 
        self.set = set
        self.BATCH_SIZE = config.batch_size_per_gpu
        self.data_fraction = config.DATA_FRACTION
        self.data = None
        self.max_idx = 0
        self.test_lab = test_lab
        self.data_folder = data_folder
        self.file = tables.open_file(data_folder)
        self.labs_and_stages = self.get_lab_and_stage_data()
        self.fixed_n = config.FIXED_N
        if self.set == 'train':
            self.train_validation_split()
            self.train_indices, self.train_dist = self.get_indices(self.labs_and_stages_train,self.fixed_n,True)
            self.val_indices, self.val_dist     = self.get_indices(self.labs_and_stages_val,self.fixed_n,False)
            self.loss_weights = self.get_loss_weights()
            self.train_dataloader = TuebingenDataLoaderSet(indices=self.train_indices, config=config, max_idx=self.max_idx, batch_size=self.BATCH_SIZE, loss_weigths=self.loss_weights)
            self.val_dataloader   = TuebingenDataLoaderSet(indices=self.val_indices, config=config, max_idx=self.max_idx,batch_size=self.BATCH_SIZE, loss_weigths=self.loss_weights)
        else:
            self.indices, _ = self.get_indices(self.labs_and_stages)

        self.file.close()
        self.on_epoch_end()  
    
    def __len__(self): # specifies the length of the total number of batches 
        return int(np.floor(len(self.indices) / self.BATCH_SIZE))

    def __getitem__(self, index): 
        if self.data is None:  # open in thread
            self.file = tables.open_file(self.config.DATA_FILE)
            self.data = self.file.root['multiple_labs']

        # Calculate the start and end index for the batch
        start_idx = index * self.BATCH_SIZE
        end_idx = min((index + 1) * self.BATCH_SIZE, len(self.indices))

        batch_features    = []
        batch_labs        = []
        batch_labels      = []

        for idx in range(start_idx, end_idx):
            internal_index = self.indices[idx]
            feature = self.data[internal_index][3]
            label = self.config.STAGES.index(str(self.data[internal_index][COLUMN_LABEL], 'utf-8'))
            lab   = self.config.LABS.index(str(self.data[internal_index][COLUMN_LAB], 'utf-8'))
            w     = self.loss_weights[str(self.data[internal_index][COLUMN_LAB], 'utf-8')][str(self.data[internal_index][COLUMN_LABEL], 'utf-8')]
            
            if self.augment_data:
                feature = np.array(feature).transpose()
                self.num_points_to_shift = feature.shape[0]
                amplitude_eeg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.7, maxval=1.3, dtype=tf.float32)
                amplitude_emg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.95, maxval=1.05, dtype=tf.float32)
                translation_amount_eeg = tf.random.uniform(shape=(), minval=-self.num_points_to_shift, maxval=self.num_points_to_shift, dtype=tf.int32)
                translation_amount_emg = translation_amount_eeg  # No translation augmentation for EMG (yoked to EEG)
                x_eeg = tf.roll(feature[:,0] * amplitude_eeg, shift=translation_amount_eeg, axis=0)
                x_emg = tf.roll(feature[:,1] * amplitude_emg, shift=translation_amount_emg, axis=0)
                feature = tf.stack([x_eeg, x_emg], axis=-1)

            batch_features.append(feature)
            batch_labels.append(label)
            batch_labs.append(w)

        # Convert lists to numpy arrays
        batch_features = np.array(batch_features)
        batch_labels = np.array(batch_labels)
        batch_labs = np.array(batch_labs)
        batch_features = batch_features.reshape(-1, 256, 2, 1)

        return batch_features, batch_labels, batch_labs

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch"""
        self.train_indices = np.random.permutation(self.train_indices)

    def augment(self, x, y):
        amplitude_eeg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.7, maxval=1.3, dtype=tf.float32)
        amplitude_emg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.95, maxval=1.05, dtype=tf.float32)
        translation_amount_eeg = tf.random.uniform(shape=(), minval=-self.num_points_to_shift, maxval=self.num_points_to_shift, dtype=tf.int32)
        translation_amount_emg = translation_amount_eeg  # No translation augmentation for EMG (yoked to EEG)
        x_eeg = tf.roll(x[:, 0] * amplitude_eeg, shift=translation_amount_eeg, axis=0)
        x_emg = tf.roll(x[:, 1] * amplitude_emg, shift=translation_amount_emg, axis=0)
        x = tf.stack([x_eeg, x_emg], axis=-1)
    
    def get_lab_and_stage_data(self):
            """ load indices of samples in the pytables table for each lab

            if data_fraction is set, load only a random fraction of the indices

            Returns:
                list: list with entries for each lab containing lists with indices of samples in that lab
            """
            lab_and_stage_data = {}
            table = self.file.root['multiple_labs']
            self.total_N = 0 
            if self.set != 'test':
                for lab in self.config.LABS:
                    if lab != self.test_lab:
                        lab_and_stage_data[lab] = {}
                        for stage in self.config.STAGES:
                            lab_and_stage_data[lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, lab, COLUMN_LABEL, stage))
                            self.total_N += len(lab_and_stage_data[lab][stage] )

                            if lab_and_stage_data[lab][stage].size > 0:
                                if max(lab_and_stage_data[lab][stage]) > self.max_idx:
                                    self.max_idx = max(lab_and_stage_data[lab][stage])
            else:
                lab_and_stage_data[self.test_lab] = {}
                for stage in self.config.STAGES:
                    lab_and_stage_data[self.test_lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, self.test_lab, COLUMN_LABEL, stage))
                    if lab_and_stage_data[self.test_lab][stage].size > 0:
                        if max(lab_and_stage_data[self.test_lab][stage]) > self.max_idx:
                            self.max_idx = max(lab_and_stage_data[self.test_lab][stage])

            return lab_and_stage_data

    def format_dictionary(dictionary: dict):
        """short method to format dicts into a semi-tabular structure"""
        formatted_str = ''
        for x in dictionary:
            formatted_str += '\t{:20s}: {}\n'.format(x, dictionary[x])
        return formatted_str[:-1]


    def get_indices(self, labs_and_stages_set,fixed_n,train):
            """ loads indices of samples in the pytables table the dataloader returns

            if flag `balanced` is set, rebalancing is done here by randomly drawing samples from all samples in a stage
            until nitems * BALANCING_WEIGHTS[stage] is reached

            drawing of the samples is done with replacement, so samples can occur more than once in the dataloader """
            indices  = np.empty(0)

            data_dist = {}
            for lab in labs_and_stages_set:
                data_dist[lab] = {}
                for stage in labs_and_stages_set[lab]:
                    data_dist[lab][stage] = labs_and_stages_set[lab][stage].size
            
            for lab in labs_and_stages_set:
                for stage in labs_and_stages_set[lab]:
                    indices = np.r_[indices, labs_and_stages_set[lab][stage]].astype('int')
                #indices = np.sort(indices)  # the samples are sorted by index for the creation of a transformation matrix
            print(len(indices))
            return indices, data_dist

            # model self.train_n 508422 self.val_n 127104

    def train_validation_split(self):
            self.labs_and_stages_train = {}
            self.labs_and_stages_val = {}
            self.val_n   = 0
            self.train_n = 0 
            if self.fixed_n: 
                num_samples_per_lab_train = 635526 / len(self.labs_and_stages)
                num_samples_per_lab_val = int(num_samples_per_lab_train * self.config.VALIDATION_SPLIT)
                num_samples_train       = int(num_samples_per_lab_train * (1-self.config.VALIDATION_SPLIT))
 
                for lab in self.labs_and_stages:
                    self.labs_and_stages_train[lab] = {}
                    self.labs_and_stages_val[lab] = {}
                    l_size = sum([s.size for s in self.labs_and_stages[lab].values()])
        
                    for stage in self.labs_and_stages[lab]:
                        lab_stage_size = self.labs_and_stages[lab][stage].size
                        stage_ratio = lab_stage_size / l_size
                        shuffled_indexes = np.random.permutation(self.labs_and_stages[lab][stage])
                        self.labs_and_stages_train[lab][stage] = np.sort(shuffled_indexes[:-int(num_samples_per_lab_val*stage_ratio)])
                        self.labs_and_stages_val[lab][stage] = np.sort(shuffled_indexes[-int(num_samples_per_lab_val*stage_ratio):])
        
                        train_lab_stage_size = self.labs_and_stages_train[lab][stage].size
 
                        if train_lab_stage_size > num_samples_train*stage_ratio:
                            l_downsampled = np.random.choice(self.labs_and_stages_train[lab][stage], size=int(num_samples_train*stage_ratio), replace=False)
                            self.labs_and_stages_train[lab][stage] = l_downsampled
                        else:
                            l_upsampled = np.random.choice(self.labs_and_stages_train[lab][stage], size=int(num_samples_train*stage_ratio), replace=True)
                            self.labs_and_stages_train[lab][stage] = l_upsampled
                  
            else:
                for lab in self.labs_and_stages:
                    self.labs_and_stages_train[lab] = {}
                    self.labs_and_stages_val[lab] = {}

                    for stage in list(self.labs_and_stages[lab]):
                        lab_stage_size = self.labs_and_stages[lab][stage].size
                        shuffled_indexes = np.random.permutation(self.labs_and_stages[lab][stage])
                        self.labs_and_stages_train[lab][stage] = np.sort(shuffled_indexes[:-int(lab_stage_size * self.config.VALIDATION_SPLIT)])
                        self.labs_and_stages_val[lab][stage] = np.sort(shuffled_indexes[-int(lab_stage_size * self.config.VALIDATION_SPLIT):])
                        self.train_n += len(self.labs_and_stages_train[lab][stage])
                        self.val_n   += len(self.labs_and_stages_train[lab][stage])

    
    def get_loss_weights(self):
            loss_weights = {}
            
            lab_sizes = [sum(self.train_dist[lab].values()) for lab in self.train_dist]
            print("N_training data")
            print(sum(lab_sizes))
        
            for lab in self.train_dist:
                loss_weights[lab] = {}

                for stage in self.config.STAGES:
                    # c = len(self.train_dist)
                    # s = 3
                    loss_weights[lab][stage] = sum(lab_sizes) / len(self.train_dist) / 3 / self.train_dist[lab][stage]
                    print(lab)
                    print(stage)
                    print( self.train_dist[lab][stage])
                    print("####")
            print(loss_weights)                        
            return loss_weights
    
class TuebingenDataLoaderSet(SequenceDataset):
    def __init__(self, indices, config, max_idx, batch_size, loss_weigths=None):
        self.indices = np.random.permutation(indices)
        self.config = config
        self.BATCH_SIZE = config.batch_size_per_gpu
        self.augment_data = config.augment_data 
        self.file = tables.open_file(self.config.DATA_FILE)
        self.file.close()
        self.data = None
        self.max_idx = max_idx
        self.loss_weights = loss_weigths



# class SequenceDataset(tf.keras.utils.Sequence):
    
#     def __init__(self, data_folder, set ,config,test_lab):

#         self.config = config
#         self.augment_data = config.augment_data 
#         self.set = set
#         self.BATCH_SIZE = config.batch_size_per_gpu
#         self.data_fraction = config.DATA_FRACTION
#         self.data = None
#         self.max_idx = 0
#         self.test_lab = test_lab
#         self.data_folder = data_folder
#         self.file = tables.open_file(data_folder)
#         self.labs_and_stages = self.get_lab_and_stage_data()
#         self.fixed_n = config.FIXED_N
#         if self.set == 'train':
#             self.train_validation_split()
#             self.train_indices, self.train_dist = self.get_indices(self.labs_and_stages_train,self.fixed_n,True)
#             self.val_indices, self.val_dist     = self.get_indices(self.labs_and_stages_val,self.fixed_n,False)
#             self.loss_weights = self.get_loss_weights()
#             self.train_dataloader = TuebingenDataLoaderSet(indices=self.train_indices, config=config, max_idx=self.max_idx, batch_size=self.BATCH_SIZE, loss_weigths=self.loss_weights)
#             self.val_dataloader   = TuebingenDataLoaderSet(indices=self.val_indices, config=config, max_idx=self.max_idx,batch_size=self.BATCH_SIZE, loss_weigths=self.loss_weights)
#         else:
#             self.indices, _ = self.get_indices(self.labs_and_stages)

#         self.file.close()
#         self.on_epoch_end()  # Shuffle data at the start
    
#     def __len__(self): # specifies the length of the total number of batches 
#         return int(np.floor(len(self.indices) / self.BATCH_SIZE))

#     def __getitem__(self, index): 
#         if self.data is None:  # open in thread
#             self.file = tables.open_file(self.config.DATA_FILE)
#             self.data = self.file.root['multiple_labs']

#         # Calculate the start and end index for the batch
#         start_idx = index * self.BATCH_SIZE
#         end_idx = min((index + 1) * self.BATCH_SIZE, len(self.indices))

#         batch_features    = []
#         batch_labs        = []
#         batch_labels      = []

#         for idx in range(start_idx, end_idx):
#             internal_index = self.indices[idx]
#             feature = self.data[internal_index][3]
#             label = self.config.STAGES.index(str(self.data[internal_index][COLUMN_LABEL], 'utf-8'))
#             lab   = self.config.LABS.index(str(self.data[internal_index][COLUMN_LAB], 'utf-8'))
#             w     = self.loss_weights[str(self.data[internal_index][COLUMN_LAB], 'utf-8')][str(self.data[internal_index][COLUMN_LABEL], 'utf-8')]
            
#             if self.augment_data:
#                 feature = np.array(feature).reshape((256, 2))
#                 self.num_points_to_shift = feature.shape[0]
#                 amplitude_eeg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.7, maxval=1.3, dtype=tf.float32)
#                 amplitude_emg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.95, maxval=1.05, dtype=tf.float32)
#                 translation_amount_eeg = tf.random.uniform(shape=(), minval=-self.num_points_to_shift, maxval=self.num_points_to_shift, dtype=tf.int32)
#                 translation_amount_emg = translation_amount_eeg  # No translation augmentation for EMG (yoked to EEG)
#                 x_eeg = tf.roll(feature[:,0] * amplitude_eeg, shift=translation_amount_eeg, axis=0)
#                 x_emg = tf.roll(feature[:,1] * amplitude_emg, shift=translation_amount_emg, axis=0)
#                 feature = tf.stack([x_eeg, x_emg], axis=-1)

#             batch_features.append(feature)
#             batch_labels.append(label)
#             batch_labs.append(w)

#         # Convert lists to numpy arrays
#         batch_features = np.array(batch_features)
#         batch_labels = np.array(batch_labels)
#         batch_labs = np.array(batch_labs)

#         batch_features = batch_features.reshape(-1, 256, 2, 1)

#         return batch_features, batch_labels, batch_labs

#     def on_epoch_end(self):
#         """Shuffle data at the end of each epoch"""
#         self.train_indices = np.random.permutation(self.train_indices)

#     def augment(self, x, y):
#         amplitude_eeg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.7, maxval=1.3, dtype=tf.float32)
#         amplitude_emg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.95, maxval=1.05, dtype=tf.float32)
#         translation_amount_eeg = tf.random.uniform(shape=(), minval=-self.num_points_to_shift, maxval=self.num_points_to_shift, dtype=tf.int32)
#         translation_amount_emg = translation_amount_eeg  # No translation augmentation for EMG (yoked to EEG)
#         x_eeg = tf.roll(x[:, 0] * amplitude_eeg, shift=translation_amount_eeg, axis=0)
#         x_emg = tf.roll(x[:, 1] * amplitude_emg, shift=translation_amount_emg, axis=0)
#         x = tf.stack([x_eeg, x_emg], axis=-1)
    
#     def get_lab_and_stage_data(self):
#             """ load indices of samples in the pytables table for each lab

#             if data_fraction is set, load only a random fraction of the indices

#             Returns:
#                 list: list with entries for each lab containing lists with indices of samples in that lab
#             """
#             lab_and_stage_data = {}
#             table = self.file.root['multiple_labs']
#             self.total_N = 0 
#             if self.set != 'test':
#                 for lab in self.config.LABS:
#                     if lab != self.test_lab:
#                         lab_and_stage_data[lab] = {}
#                         for stage in self.config.STAGES:
#                             lab_and_stage_data[lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, lab, COLUMN_LABEL, stage))
#                             self.total_N += len(lab_and_stage_data[lab][stage] )

#                             if lab_and_stage_data[lab][stage].size > 0:
#                                 if max(lab_and_stage_data[lab][stage]) > self.max_idx:
#                                     self.max_idx = max(lab_and_stage_data[lab][stage])
#             else:
#                 lab_and_stage_data[self.test_lab] = {}
#                 for stage in self.config.STAGES:
#                     lab_and_stage_data[self.test_lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, self.test_lab, COLUMN_LABEL, stage))
#                     if lab_and_stage_data[self.test_lab][stage].size > 0:
#                         if max(lab_and_stage_data[self.test_lab][stage]) > self.max_idx:
#                             self.max_idx = max(lab_and_stage_data[self.test_lab][stage])

#             return lab_and_stage_data

#     def format_dictionary(dictionary: dict):
#         """short method to format dicts into a semi-tabular structure"""
#         formatted_str = ''
#         for x in dictionary:
#             formatted_str += '\t{:20s}: {}\n'.format(x, dictionary[x])
#         return formatted_str[:-1]


#     def get_indices(self, labs_and_stages_set,fixed_n,train):
#             """ loads indices of samples in the pytables table the dataloader returns

#             if flag `balanced` is set, rebalancing is done here by randomly drawing samples from all samples in a stage
#             until nitems * BALANCING_WEIGHTS[stage] is reached

#             drawing of the samples is done with replacement, so samples can occur more than once in the dataloader """
#             indices  = np.empty(0)
#             indices_ = np.empty(0)
#             random_samples = np.empty(0)
#             # model self.train_n 508422 self.val_n 127104
#             if fixed_n==True & train==True:  
#                 n  = np.round(508422/len(self.labs_and_stages))
#                 print("number of labs:")
#                 print(len(self.labs_and_stages))
#                 print(self.train_n)

#                 for lab in labs_and_stages_set:
#                     indices_ = np.empty(0)
#                     for stage in labs_and_stages_set[lab]:
#                         indices_ = np.r_[indices_, labs_and_stages_set[lab][stage]].astype('int')
#                     if len(indices_) < n:
#                         print(f"Upsampling required for lab {lab}. Number of samples available: {len(indices_)}, required: {int(n)}")
#                         random_samples = np.r_[random_samples, np.random.choice(indices_, int(n), replace=True)].astype('int')
#                     elif len(indices_) == n:
#                         random_samples = np.r_[random_samples, indices_].astype('int')
#                     else:
#                         random_samples = np.r_[random_samples, np.random.choice(indices_, int(n), replace=False)].astype('int')
#                 indices = random_samples

#                 data_dist = {}
#                 for lab in labs_and_stages_set:
#                     data_dist[lab] = {}
#                     for stage in labs_and_stages_set[lab]:
#                         count = np.intersect1d(indices, labs_and_stages_set[lab][stage]).size
#                         data_dist[lab][stage] = count

#             else:
#                 data_dist = {}
#                 for lab in labs_and_stages_set:
#                     data_dist[lab] = {}
#                     for stage in labs_and_stages_set[lab]:
#                         data_dist[lab][stage] = labs_and_stages_set[lab][stage].size
            
#                 for lab in labs_and_stages_set:
#                     for stage in labs_and_stages_set[lab]:
#                         indices = np.r_[indices, labs_and_stages_set[lab][stage]].astype('int')
#                 #indices = np.sort(indices)  # the samples are sorted by index for the creation of a transformation matrix
#             print(len(indices))
#             return indices, data_dist

#     def train_validation_split(self):
#             self.labs_and_stages_train = {}
#             self.labs_and_stages_val = {}
#             self.val_n   = 0
#             self.train_n = 0 
#             if self.config.DATA_FRACTION == True:
#                 num_samples_per_lab_train = int(self.total_N / len(self.labs_and_stages)) 
#                 num_samples_per_lab_val = int(num_samples_per_lab_train * self.config.VALIDATION_SPLIT)

#                 for lab in self.labs_and_stages:
#                     self.labs_and_stages_train[lab] = {}
#                     self.labs_and_stages_val[lab] = {}
#                     l_size = sum([s.size for s in self.labs_and_stages[lab].values()])

#                     for stage in list(self.labs_and_stages[lab]):
#                         lab_stage_size = self.labs_and_stages[lab][stage].size
#                         stage_ratio = lab_stage_size / l_size
                        
#                         shuffled_indexes = np.random.permutation(self.labs_and_stages[lab][stage])

#                         if len(self.labs_and_stages[lab][stage]) > int(num_samples_per_lab_val*stage_ratio): 
#                             self.labs_and_stages_train[lab][stage] = np.sort(shuffled_indexes[:-int(num_samples_per_lab_val*stage_ratio)])
#                             self.labs_and_stages_val[lab][stage]   = np.sort(shuffled_indexes[-int(num_samples_per_lab_val*stage_ratio):])
#                         else:
#                             num_val_samples = int(l_size * stage_ratio)
#                             num_train_samples = l_size - num_val_samples
#                             self.labs_and_stages_train[lab][stage] = np.sort(shuffled_indexes[:num_train_samples])
#                             self.labs_and_stages_val[lab][stage] = np.sort(shuffled_indexes[num_train_samples:num_train_samples + num_val_samples])
                                                
#                         self.train_n += len(self.labs_and_stages_train[lab][stage])
#                         self.val_n   += len(self.labs_and_stages_val[lab][stage])
#             else:

#                 for lab in self.labs_and_stages:
#                     self.labs_and_stages_train[lab] = {}
#                     self.labs_and_stages_val[lab] = {}

#                     for stage in list(self.labs_and_stages[lab]):
#                         lab_stage_size = self.labs_and_stages[lab][stage].size
                        
#                         shuffled_indexes = np.random.permutation(self.labs_and_stages[lab][stage])
#                         self.labs_and_stages_train[lab][stage] = np.sort(shuffled_indexes[:-int(lab_stage_size * self.config.VALIDATION_SPLIT)])
#                         self.labs_and_stages_val[lab][stage] = np.sort(shuffled_indexes[-int(lab_stage_size * self.config.VALIDATION_SPLIT):])
#                         self.train_n += len(self.labs_and_stages_train[lab][stage])
#                         self.val_n   += len(self.labs_and_stages_train[lab][stage])

    
#     def get_loss_weights(self):
#             loss_weights = {}
            
#             lab_sizes = [sum(self.train_dist[lab].values()) for lab in self.train_dist]
#             print("N_training data")
#             print(sum(lab_sizes))
        
#             for lab in self.train_dist:
#                 loss_weights[lab] = {}

#                 for stage in self.config.STAGES:
#                     # c = len(self.train_dist)
#                     # s = 3
#                     loss_weights[lab][stage] = sum(lab_sizes) / len(self.train_dist) / 3 / self.train_dist[lab][stage]
#                     print(lab)
#                     print(stage)
#                     print( self.train_dist[lab][stage])
#                     print("####")
#             print(loss_weights)                        
#             return loss_weights
    
# class TuebingenDataLoaderSet(SequenceDataset):
#     def __init__(self, indices, config, max_idx, batch_size, loss_weigths=None):
#         self.indices = np.random.permutation(indices)
#         self.config = config
#         self.BATCH_SIZE = config.batch_size_per_gpu
#         self.augment_data = config.augment_data 
#         self.file = tables.open_file(self.config.DATA_FILE)
#         self.file.close()
#         self.data = None
#         self.max_idx = max_idx
#         self.loss_weights = loss_weigths



class SequenceDataset_test(tf.keras.utils.Sequence):
    
    def __init__(self, data_folder, set ,config,test_lab):
        self.config = config
        self.set = set
        self.BATCH_SIZE = config.batch_size_per_gpu
        self.data_fraction = config.DATA_FRACTION
        self.data = None
        self.max_idx = 0
        self.test_lab = test_lab
        self.data_folder = data_folder
        self.file = tables.open_file(data_folder)
        self.labs_and_stages = self.get_lab_and_stage_data()
        self.indices, _ = self.get_indices(self.labs_and_stages)
        self.file.close()
    
    def __len__(self): # specifies the length of the total number of batches 
        return int(np.floor(len(self.indices) / self.BATCH_SIZE))

    def __getitem__(self, index): 
        if self.data is None:  # open in thread
            self.file = tables.open_file(self.config.DATA_FILE)
            self.data = self.file.root['multiple_labs']

        # Calculate the start and end index for the batch
        start_idx = index * self.BATCH_SIZE
        end_idx = min((index + 1) * self.BATCH_SIZE, len(self.indices))

        batch_features    = []
        batch_labs        = []
        batch_labels      = []

        for idx in range(start_idx, end_idx):
            internal_index = self.indices[idx]
            feature = self.data[internal_index][3]
            feature = feature.transpose().reshape(1,256,2).astype('float32')
            #feature = feature.reshape(-1,256,2).astype('float32') 
            label = self.config.STAGES.index(str(self.data[internal_index][COLUMN_LABEL], 'utf-8'))
            lab = self.config.LABS.index(str(self.data[internal_index][COLUMN_LAB], 'utf-8'))

            batch_features.append(feature)
            batch_labels.append(label)
            batch_labs.append(lab)

        # Convert lists to numpy arrays
        batch_features = np.vstack(batch_features)
        #batch_features = batch_features.reshape(-1, 256, 2, 1)
        batch_labels = np.array(batch_labels)
        batch_labs = np.array(batch_labs)

        return batch_features, batch_labels, batch_labs


    def get_lab_and_stage_data(self):
            """ load indices of samples in the pytables table for each lab

            if data_fraction is set, load only a random fraction of the indices

            Returns:
                list: list with entries for each lab containing lists with indices of samples in that lab
            """
            lab_and_stage_data = {}
            table = self.file.root['multiple_labs']

            if self.set != 'test':
                for lab in self.config.LABS:
                    if lab != self.test_lab:
                        lab_and_stage_data[lab] = {}
                        for stage in self.config.STAGES:
                            lab_and_stage_data[lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, lab, COLUMN_LABEL, stage))
                            if lab_and_stage_data[lab][stage].size > 0:
                                if max(lab_and_stage_data[lab][stage]) > self.max_idx:
                                    self.max_idx = max(lab_and_stage_data[lab][stage])
            else:
                lab_and_stage_data[self.test_lab] = {}
                for stage in list(self.config.STAGES):
                    lab_and_stage_data[self.test_lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, self.test_lab, COLUMN_LABEL, stage))
                    if lab_and_stage_data[self.test_lab][stage].size > 0:
                        if max(lab_and_stage_data[self.test_lab][stage]) > self.max_idx:
                            self.max_idx = max(lab_and_stage_data[self.test_lab][stage])

            return lab_and_stage_data

    def format_dictionary(dictionary: dict):
        """short method to format dicts into a semi-tabular structure"""
        formatted_str = ''
        for x in dictionary:
            formatted_str += '\t{:20s}: {}\n'.format(x, dictionary[x])
        return formatted_str[:-1]


    def get_indices(self, labs_and_stages_set):
            """ loads indices of samples in the pytables table the dataloader returns

            if flag `balanced` is set, rebalancing is done here by randomly drawing samples from all samples in a stage
            until nitems * BALANCING_WEIGHTS[stage] is reached

            drawing of the samples is done with replacement, so samples can occur more than once in the dataloader """
            indices = np.empty(0)
            
            data_dist = {}
            for lab in labs_and_stages_set:
                data_dist[lab] = {}
                for stage in labs_and_stages_set[lab]:
                    data_dist[lab][stage] = labs_and_stages_set[lab][stage].size
        
            for lab in labs_and_stages_set:
                for stage in labs_and_stages_set[lab]:
                    indices = np.r_[indices, labs_and_stages_set[lab][stage]].astype('int')
            indices = np.sort(indices)  # the samples are sorted by index for the creation of a transformation matrix

        
            return indices, data_dist

    
# AugmentDataGenerator class definition remains the same
class AugmentDataGenerator():
    def __init__(self, x_set, y_set, batch_size, is_training=True):
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_points_to_shift = x_set.shape[1]

    def augment(self, x, y):
        amplitude_eeg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.7, maxval=1.3, dtype=tf.float32)
        amplitude_emg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.95, maxval=1.05, dtype=tf.float32)
        translation_amount_eeg = tf.random.uniform(shape=(), minval=-self.num_points_to_shift, maxval=self.num_points_to_shift, dtype=tf.int32)
        translation_amount_emg = translation_amount_eeg  # No translation augmentation for EMG (yoked to EEG)
        x_eeg = tf.roll(x[:, 0] * amplitude_eeg, shift=translation_amount_eeg, axis=0)
        x_emg = tf.roll(x[:, 1] * amplitude_emg, shift=translation_amount_emg, axis=0)
        x = tf.stack([x_eeg, x_emg], axis=-1)
        return x, y
    
    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_set, self.y_set))
        if self.is_training:
            dataset = dataset.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
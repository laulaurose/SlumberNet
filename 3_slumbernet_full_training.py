#########################################################################
# SlumberNet: Sleep Stage Classification using Residual Neural Networks #
#########################################################################

# This Python script implements SlumberNet, a deep learning model for classifying
# sleep stages using residual neural networks.

# The script takes in preprcessed EEG and EMG data and then performs training on all
# available data.

# The model was developed by Pawan K. Jha, Utham K. Valekunja, and Akhilesh B. Reddy,
# and the work was conducted at the Department of Systems Pharmacology & Translational
# Therapeutics, the Institute for Translational Medicine and Therapeutics, and the
# Chronobiology and Sleep Institute (CSI), Perelman School of Medicine, University
# of Pennsylvania.

# Date: May 5th, 2023

# NOTE: This script is designed to run on an Nvidia GPU-enabled machine. It may require
# modification to run on a CPU-only machine, or on your particular Nvidia GPU. Also,
# tensorflow-gpu can run into machine-specific CUDA driver issues, which you may need
# to troubleshoot prior to running this script. We recommend using the Anaconda/miniconda
# environments to install tensorflow-gpu and its dependencies.


### ---- Load libraries ---- ###
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger


# Input directory
input_directory =  '/work3/laurose/SlumberNet/preprocessed_data/'

# Set the random seed
seed = 154727
np.random.seed(seed)

# Use multiple GPUs if available - this will use all available GPUs
strategy = tf.distribute.MirroredStrategy()
number_of_gpus_available = strategy.num_replicas_in_sync

# Use mixed precision compute on GPU (float16 and float32) for higher speed training on compute 6.0+ Nvidia GPUs
# https://www.tensorflow.org/guide/mixed_precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# Model traiing parameters
num_epochs = 50
learning_rate = 1e-06
batch_size_per_gpu = 128
batch_size = batch_size_per_gpu * number_of_gpus_available
optimizer_name = keras.optimizers.legacy.Adam(learning_rate)                # This is to pull out the correct name label for the optimizer
optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer_name)        # Wrapping the optimizer avoid crashes on multiple GPUs

# Define ResNet2D model function and parameters
input_shape = (256,2,1)
nb_classes = 3              # Number of classes (W, N, R)   

n_resnet_blocks = 7
n_feature_maps = 8 
kernel_expansion_fct = 1
kernel_y = 2
strides = (1,1)
dropout_rate = 0
dropout_str = str(dropout_rate)     # Convert dropout rate to string for output data

# Data augmentation?
augment_data = True
if augment_data:
    augmented = 'yes'
else:    
    augmented = 'no'

# Output directory (make new output directory if it doesn't exist)
output_directory = f'{input_directory}final_model/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to create ResNet2D model
def resnet_blocks(input_tensor, n_feature_maps, kernel_y, kernel_expansion_fct, strides, n_blocks, dropout_rate=0.0):
    output_tensor = input_tensor

    for i in range(n_blocks-1):
        # Repeating Resnet blocks
        conv_x = keras.layers.Conv2D(filters=n_feature_maps * (2 ** i), kernel_size=(kernel_y,8*kernel_expansion_fct), strides=strides, padding='same')(output_tensor)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Dropout(dropout_rate)(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=n_feature_maps * (2 ** i), kernel_size=(kernel_y,5*kernel_expansion_fct), strides=strides, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Dropout(dropout_rate)(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=n_feature_maps * (2 ** i), kernel_size=(kernel_y,3*kernel_expansion_fct), strides=strides, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv2D(filters=n_feature_maps * (2 ** i), kernel_size=(kernel_y,1), padding='same')(output_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_tensor = keras.layers.add([shortcut_y, conv_z])
        output_tensor = keras.layers.Dropout(dropout_rate)(output_tensor)
        output_tensor = keras.layers.Activation('relu')(output_tensor)
    
    # Final block
    conv_x = keras.layers.Conv2D(filters=n_feature_maps * (2 ** i), kernel_size=(kernel_y,8*kernel_expansion_fct), strides=strides, padding='same')(output_tensor)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv2D(filters=n_feature_maps * (2 ** i), kernel_size=(kernel_y,8*kernel_expansion_fct), strides=strides, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv2D(filters=n_feature_maps * (2 ** i), kernel_size=(kernel_y,8*kernel_expansion_fct), strides=strides, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # No need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_tensor)

    output_tensor = keras.layers.add([shortcut_y, conv_z])
    output_tensor = keras.layers.Dropout(dropout_rate)(output_tensor)
    output_tensor = keras.layers.Activation('relu')(output_tensor)

    return output_tensor

# Save parameters (num_folds, num_epochs, batch_size_per_gpu, n_feature_maps, kernel_expansion_fct) to csv
parameters = {'Number of Resnet blocks': n_resnet_blocks, 'Optimizer': optimizer_name.__class__.__name__, 
                'Number of epochs': num_epochs, 'Batch size per GPU': batch_size_per_gpu, 'Learning rate (initial)': learning_rate, 
                'Number of feature maps in top layer': n_feature_maps, 'Kernel y dimension (1 or 2)': kernel_y, 
                'Kernel factor size multiple': kernel_expansion_fct, 'Training data augmented': augmented, 'Dropout rate': dropout_rate}
parameters_df = pd.DataFrame(parameters, index=[0])
parameters_df.to_csv(output_directory + 'run_parameters.csv', index=False)

# Data and labels
X = np.load(input_directory + "eeg_input_array.npy")
y = np.load(input_directory + "epoch_input_array.npy")

# Reshape array for Conv2D shape(256,2) 
X = X.reshape(-1,256,2)
X = X.astype('float32')         # Make float32 for tensorflow data augmentation calculations (default is float32)

# Data generator with augmentation if specified
class AugmentDataGenerator():
    def __init__(self, x_set, y_set, batch_size, is_training=True):
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_points_to_shift = x_set.shape[1]

    # Tensorflow only data augmentation (faster)
    def augment(self, x, y):
        amplitude_eeg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.7, maxval=1.3, dtype=tf.float32)
        amplitude_emg = tf.random.uniform(shape=(self.num_points_to_shift,), minval=0.95, maxval=1.05, dtype=tf.float32)
        translation_amount_eeg = tf.random.uniform(shape=(), minval=-self.num_points_to_shift, maxval=self.num_points_to_shift, dtype=tf.int32)
        translation_amount_emg = translation_amount_eeg     # No translation augmentation for EMG (yoked to EEG)
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

# # For testing
# X = X[:10000]
# y = y[:10000]

# Shuffle the data for training
X, y = shuffle(X, y, random_state=seed)

# Print the number of samples
print(f'Training data: {X.shape}, Training labels: {y.shape}')

# Build model for multiple GPUs
with strategy.scope():
    input_layer = keras.layers.Input(input_shape)

    output_tensor= resnet_blocks(input_tensor=input_layer, n_feature_maps=n_feature_maps, kernel_y=kernel_y, 
                                kernel_expansion_fct=kernel_expansion_fct, strides=strides, 
                                n_blocks=n_resnet_blocks, dropout_rate=dropout_rate)
    
    # Final layer
    gap_layer = keras.layers.GlobalAveragePooling2D()(output_tensor)
    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model on the strategy scope (multi-GPU)
with strategy.scope():
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer,
                metrics=['accuracy'])

# Wrap data in generators
train_generator = AugmentDataGenerator(X, y, batch_size, is_training=augment_data)
train_dataset = train_generator.create_dataset()


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="SlumberNet",
    # track hyperparameters and run metadata
    config={
    "architecture": "SlumberNET",
    "dataset": "SlumberNET",
    "epochs": num_epochs,
    }
)

# Create a callback function to save the best model and set up learning rate reduction
best_model_filepath = output_directory + "final_model_best.h5"
checkpoint = ModelCheckpoint(best_model_filepath, monitor='accuracy', save_best_only=True, mode='max', save_weights_only=False)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-07)
callbacks_list = [reduce_lr, checkpoint,WandbMetricsLogger()]


# Train the model on the training data (using multi-GPU)
with strategy.scope():
    history = model.fit(train_dataset,
                        callbacks=callbacks_list,
                        epochs=num_epochs)                        

# Save last model
model.save(output_directory + "final_model_last.h5")

# Save training and validation accuracy and losses for each epoch
history_df = pd.DataFrame(history.history)
history_df.to_csv(output_directory + 'history.csv', index=False)

# Plot accuracy and losses over epochs
plt.plot(history.history['accuracy'], color='blue')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend('Train', loc='upper left')
plt.savefig(output_directory + 'accuracy_over_epoch.png')
plt.savefig(output_directory + 'accuracy_over_epoch.pdf')
plt.show()

plt.clf()

plt.plot(history.history['loss'], color='blue')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend('Train', loc='upper left')
plt.savefig(output_directory + 'loss_over_epoch.png')
plt.savefig(output_directory + 'loss_over_epoch.pdf')
plt.show()

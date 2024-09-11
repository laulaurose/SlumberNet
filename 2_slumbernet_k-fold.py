#########################################################################
# SlumberNet: Sleep Stage Classification using Residual Neural Networks #
#########################################################################

# This Python script implements SlumberNet, a deep learning model for classifying
# sleep stages using residual neural networks.

# The script takes in preprcessed EEG and EMG data and then performs training and
# testing using k-fold cross-validation. 

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score, explained_variance_score, log_loss
import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

# Input directory
input_directory =  '/work3/laurose/SlumberNet/preprocessed_data/'

# Set the random seed for reproducible results
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

# Number of folds for k-fold cross-validation
num_folds = 5

# Model traiing parameters
num_epochs = 50
learning_rate = 1e-06
batch_size_per_gpu = 128
batch_size = batch_size_per_gpu * number_of_gpus_available

# This is to pull out the correct name label for the optimizer (if we need for metadata)
optimizer_name = keras.optimizers.legacy.Adam(learning_rate)
# Wrapping the optimizer avoid crashes on multiple GPUs:               
optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer_name)        

# Define ResNet2D model function and parameters
input_shape = (256,2,1)
nb_classes = 3              # Number of classes (W, N, R)   

n_resnet_blocks = 7
n_feature_maps = 8 
kernel_expansion_fct = 1
kernel_y = 2
strides = (1,1)
dropout_rate = 0
dropout_str = str(dropout_rate)     # Convert dropout rate to string for metadata

# Data augmentation?
augment_data = True
if augment_data:
    augmented = 'yes'
else:    
    augmented = 'no'

# Output directory (make new output directory if it doesn't exist)
output_directory = f'{input_directory}k-fold_models/'
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

        # Expand channels for the sum
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
parameters = {'Number of folds for k-fold cross-validation': num_folds, 'Number of Resnet blocks': n_resnet_blocks, 'Optimizer': optimizer_name.__class__.__name__, 
                'Number of epochs': num_epochs, 'Batch size per GPU': batch_size_per_gpu, 'Learning rate (initial)': learning_rate, 
                'Number of feature maps in top layer': n_feature_maps, 'Kernel y dimension (1 or 2)': kernel_y, 
                'Kernel factor size multiple': kernel_expansion_fct, 'Training data augmented': augmented, 'Dropout rate': dropout_rate}
parameters_df = pd.DataFrame(parameters, index=[0])
parameters_df.to_csv(output_directory + 'run_parameters.csv', index=False)

# Initialize metrics
precisions, recalls, f1_scores, supports = [], [], [], []
accuracies, kappas, loss, explained_variances, confusion_matrices = [], [], [], [], []

# Data and labels
X = np.load(input_directory + "eeg_input_array.npy")
y = np.load(input_directory + "epoch_input_array.npy")

# Reshape array for Conv2D shape(256,2) 
X = X.reshape(-1,256,2)
X = X.astype('float32')         # Make float32 for tensorflow data augmentation calculations (default is float32)

# Data generator class with augmentation if specified
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

# Define the k-fold cross validation - maintains proportions of classes in each fold
sss = StratifiedShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=seed)

# Print the number of samples in each fold
for train_index, test_index in sss.split(X,y):
    print(f'Train set: {train_index.shape[0]}, Test set: {test_index.shape[0]},\
        Train:test = {train_index.shape[0]*100/(train_index.shape[0]+test_index.shape[0])}:\
                    {test_index.shape[0]*100/(train_index.shape[0]+test_index.shape[0])}')

# Train the model in k-folds
fold_num = 1
for train_index, test_index in sss.split(X,y):

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

    # Split the data into train and test sets using the indexes from the k-fold split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(f'Fold {fold_num}, X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}')

    # Wrap data in generators
    train_generator = AugmentDataGenerator(X_train, y_train, batch_size, is_training=augment_data)
    test_generator = AugmentDataGenerator(X_test, y_test, batch_size, is_training=False)    # Don't augment test data ever
    train_dataset = train_generator.create_dataset()
    test_dataset = test_generator.create_dataset()

    # Create a callback function to save the best model and set up learning rate reduction
    best_model_filepath = output_directory + "fold_" + str(fold_num) + "_model_best.h5"
    checkpoint = ModelCheckpoint(best_model_filepath, monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-07)
    callbacks_list = [reduce_lr, checkpoint]

    # Train the model on the training data (using multi-GPU)
    with strategy.scope():
        history = model.fit(train_dataset,
                            validation_data=test_dataset,
                            callbacks=callbacks_list,
                            epochs=num_epochs)                        

    # Save last model
    model.save(output_directory + "fold_" + str(fold_num) + "_model_last.h5")

    # Save training and validation accuracy and losses for each epoch
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_directory + 'fold_{}_history.csv'.format(fold_num), index=False)

    # Load the best model for predictions       
    model = keras.models.load_model(best_model_filepath)

    # Make predictions on the test set (using multi-GPU)
    with strategy.scope():
        y_pred = model.predict(X_test)
    
    # Store raw predictions
    y_test_raw = y_test.copy()
    y_pred_raw = y_pred.copy()

    # Convert y_test and y_pred to integer coding for sklearn metrics
    y_test = np.argmax(y_test_raw, axis=1)
    y_pred = np.argmax(y_pred_raw, axis=1)

    # Convert y_test and y_pred to one-hot encoded format for some metrics
    y_test_one_hot = np.eye(y_test_raw.shape[1])[np.argmax(y_test_raw, axis=1)].copy()
    y_pred_one_hot = np.eye(y_pred_raw.shape[1])[np.argmax(y_pred_raw, axis=1)].copy()

    # Calculate various metrics
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)
    supports.append(support)
    accuracies.append(accuracy_score(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(conf_matrix)
    kappas.append(cohen_kappa_score(y_test, y_pred))
    loss.append(log_loss(y_test_one_hot, y_pred_one_hot))
    explained_variances.append(explained_variance_score(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix,annot=True,cmap="YlGnBu",fmt='g')
    plt.title('Confusion Matrix for Fold {}'.format(fold_num))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(output_directory + 'fold_{}_confusion_matrix.png'.format(fold_num))
    plt.savefig(output_directory + 'fold_{}_confusion_matrix.pdf'.format(fold_num))
    plt.show()
    
    plt.clf()

    # Plot accuracy and losses over epochs
    plt.plot(history.history['accuracy'], color='blue')
    plt.plot(history.history['val_accuracy'], color='orange')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(output_directory + 'fold_{}_accuracy_over_epoch.png'.format(fold_num))
    plt.savefig(output_directory + 'fold_{}_accuracy_over_epoch.pdf'.format(fold_num))
    plt.show()

    plt.clf()
    
    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='orange')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(output_directory + 'fold_{}_loss_over_epoch.png'.format(fold_num))
    plt.savefig(output_directory + 'fold_{}_loss_over_epoch.pdf'.format(fold_num))
    plt.show()

    fold_num += 1

# Create a dataframe to hold the metric values for each all folds
metrics = {
    'Precision': precisions,
    'Recall': recalls,
    'F1-Score': f1_scores,
    'Support': supports,
    'Accuracy': accuracies,
    'Cohen_Kappa': kappas,
    'Log_Loss': loss,
    'Explained_Variance': explained_variances
}

metrics_results_df = pd.DataFrame(metrics)
metrics_results_df.to_csv(output_directory + 'kfold_metrics.csv', index=False)

# Save the confusion matrices to a csv file
for i, matrix in enumerate(confusion_matrices):
    with open(output_directory + 'confusion_matrix_fold_{}.csv'.format(i+1), mode='w') as csv_file:
        writer = csv.writer(csv_file)
        for row in matrix:
            writer.writerow(row)


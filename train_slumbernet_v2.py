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
from utils.config.config_loader import ConfigLoader
import argparse
from model import resnet_blocks 
from data_loading import SequenceDataset
import keras as keras 
from utils.metrics.metrics import *

def parse():
    """define and parse arguments for the script"""
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--basedir', '-e', required=True,
                        help='name of experiment to run')
    parser.add_argument('--test_lab', '-t', required=True,
                        help='name of experiment to run')
    parser.add_argument('--out_dir', '-s', required=True,
                        help='name of experiment to run')
    return parser.parse_args()


def customloss(y_true, y_pred,y_lab,n_classes,config):
    
    y_true = tf.one_hot(y_true, depth=n_classes)
    if config.LOSS_TYPE=='normal_ce':
        ce = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
    else: 
        ce = tf.keras.metrics.categorical_crossentropy(y_true, y_pred)
        ce = ce*y_lab

    return tf.reduce_mean(ce)

args       = parse()

config_file = args.basedir 
config      = ConfigLoader(config_file,create_dirs=True)  # load config from experiment
save_path   = args.out_dir
model_name  = config.model_name 

# ------------------------------------------------------ WEIGHTS AND BIASES -------------------------------------------------------------------

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="SlumberNet",
    # track hyperparameters and run metadata
    config={
    "architecture": "SlumberNET",
    "dataset": "SlumberNET",
    "epochs": config.num_epochs,
    }
)

# ------------------------------------------------------ MODEL SPECIFICATIONS -------------------------------------------------------------------



# Output directory (make new output directory if it doesn't exist)
if not os.path.exists(save_path):
    os.makedirs(save_path)


optimizer = tf.keras.optimizers.Adam(config.learning_rate)

input_shape = (256,2,1)

input_layer = keras.layers.Input(input_shape)

output_tensor= resnet_blocks(input_tensor=input_layer, 
                            n_feature_maps=config.n_feature_maps, 
                            kernel_y=config.kernel_y, 
                            kernel_expansion_fct=config.kernel_expansion_fct, 
                            strides=(config.strides_size,config.strides_size), 
                            n_blocks=config.n_resnet_blocks, 
                            dropout_rate=config.dropout_rate)
    
# Final layer
gap_layer    = keras.layers.GlobalAveragePooling2D()(output_tensor)
output_layer = keras.layers.Dense(config.nb_classes, activation='softmax')(gap_layer)
model        = keras.models.Model(inputs=input_layer, outputs=output_layer)
loss_fn_SS   = customloss

metrics_list_SS    = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                        MulticlassF1Score(n_classes=config.nb_classes,name='f1_score'),
                        MulticlassBalancedAccuracy(n_classes=config.nb_classes,name='balance_accuracy')]

model.compile(optimizer=optimizer, loss=loss_fn_SS, metrics=metrics_list_SS)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy() 
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy() 

# ------------------------------------------------------ LOAD DATA -------------------------------------------------------------------

dl_train = SequenceDataset(data_folder=config.DATA_FILE,
                                  set='train',
                                  config=config,
                                  test_lab = args.test_lab)


# ------------------------------------------------------ TRAINING LOOP -------------------------------------------------------------------

epochs                 = config.num_epochs
best_val_f1            = -np.inf
early_stopping_counter = 0
patience               = 10
f1_metric              = MulticlassF1Score(n_classes=3, name='f1_score')
acc_all                = []
categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()
f1_all = []
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")

    dl_train.on_epoch_end()
    for step, (x_batch_train, y_batch_train, y_batch_labs) in enumerate(dl_train.train_dataloader):
        with tf.GradientTape() as tape:

            logits_SS  = model(x_batch_train, training=True)  
            loss_value = loss_fn_SS(y_batch_train, logits_SS, y_batch_labs, config.nb_classes, config)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(y_batch_train, logits_SS)

            for metric in metrics_list_SS:
                metric.update_state(tf.one_hot(y_batch_train, depth=3), tf.one_hot(np.argmax(logits_SS,axis=1),depth=3))
            
            if step % 20 == 0: 
                metrics_log = {metric.name: metric.result().numpy() for metric in metrics_list_SS}
                metrics_log["loss"] = loss_value.numpy()
                wandb.log(metrics_log)

    # Evaluate on the validation dataset at the end of each epoch
    f1_metric.reset_states()
    categorical_accuracy.reset_states()

    for x_batch_val, y_batch_val, y_batch_labs in dl_train.val_dataloader:
        val_logits = model(x_batch_val, training=False)
        mask = y_batch_val != 3    
        f1_metric.update_state(tf.one_hot(y_batch_val[mask], depth=3), tf.one_hot(np.argmax(val_logits[mask],axis=1),depth=3))
        categorical_accuracy.update_state(tf.one_hot(y_batch_val[mask], depth=3), tf.one_hot(np.argmax(val_logits[mask],axis=1),depth=3))

    val_f1  = f1_metric.result().numpy()
    val_acc = categorical_accuracy.result().numpy()
    acc_all.append(val_acc)
    f1_all.append(val_f1)
    model.save_weights(save_path+"epoch"+str(epoch)+".h5")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_weights(save_path+"epoch"+str(epoch)+".h5")

    # Log metrics at the end of the epoch
    metrics_log = {metric.name: metric.result().numpy() for metric in metrics_list_SS}
    metrics_log["epoch"] = epoch
    metrics_log["val_f1"] = val_f1
    wandb.log(metrics_log)
    print(f"Epoch {epoch} metrics: {metrics_log}")

np.save(save_path+'acc_array.npy', np.array(acc_all))
np.save(save_path+'f1_array.npy', np.array(f1_all))

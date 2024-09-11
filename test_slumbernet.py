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
from data_loading import SequenceDataset,  SequenceDataset_test
import keras as keras 
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from utils.metrics.metrics import *

def cal_metrics(labels, pred):
    assert len(pred) == len(labels)

    metrics_dict = {
        "Recall_W": [],
        "Precision_W": [],
        "F1 Score_W": [],
        "Accuracy_W": [],
        "Balanced Accuracy_W": [],
        "Recall_N": [],
        "Precision_N": [],
        "F1 Score_N": [],
        "Accuracy_N": [],
        "Balanced Accuracy_N": [],
        "Recall_R": [],
        "Precision_R": [],
        "F1 Score_R": [],
        "Accuracy_R": [],
        "Balanced Accuracy_R": []
    }

    n_classes = 3

    mask = np.where(np.logical_and(
            np.greater_equal(labels, 0),
            np.less(labels, n_classes)
        ), np.ones_like(labels), np.zeros_like(labels)).astype(bool)
    pred = pred[mask]
    labels = labels[mask]

    for i in np.unique(pred):  # loop across class
        TP = np.sum((labels == i) & (pred == i))
        FP = np.sum((labels != i) & (pred == i))  # when pred says positive but it was another class
        FN = np.sum((labels == i) & (pred != i))  # when we have a positive we did not predict as positive
        TN = np.sum((labels != i) & (pred != i))

        recall_i      = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision_i   = TP / (TP + FP) if (TP + FP) > 0 else 0
        specificity_i = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1score_i     = 2 * (precision_i * recall_i) / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0
        accuracy_i    = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        b_accuracy_i  = (recall_i + specificity_i) / 2

        if i == 0:
            metrics_dict["Recall_W"].append(recall_i)
            metrics_dict["Precision_W"].append(precision_i)
            metrics_dict["F1 Score_W"].append(f1score_i)
            metrics_dict["Accuracy_W"].append(accuracy_i)
            metrics_dict["Balanced Accuracy_W"].append(b_accuracy_i)
        elif i == 1:
            metrics_dict["Recall_N"].append(recall_i)
            metrics_dict["Precision_N"].append(precision_i)
            metrics_dict["F1 Score_N"].append(f1score_i)
            metrics_dict["Accuracy_N"].append(accuracy_i)
            metrics_dict["Balanced Accuracy_N"].append(b_accuracy_i)
        elif i == 2:
            metrics_dict["Recall_R"].append(recall_i)
            metrics_dict["Precision_R"].append(precision_i)
            metrics_dict["F1 Score_R"].append(f1score_i)
            metrics_dict["Accuracy_R"].append(accuracy_i)
            metrics_dict["Balanced Accuracy_R"].append(b_accuracy_i)
        
        
    return metrics_dict



def plot_cm(true,pred,normalize=False,
            cmap="Blues",title=None,out_path=None,precision=False,odds=False):

    n_classes = 3

    mask = np.where(np.logical_and(
            np.greater_equal(true, 0),
            np.less(true, n_classes)
        ), np.ones_like(true), np.zeros_like(true)).astype(bool)
    pred = pred[mask]
    true = true[mask]

    cm = confusion_matrix(true, pred,labels=[0,1,2])

    if precision:
        total_samples = np.sum(cm, axis=0)
        normalized_cm = (cm / total_samples[np.newaxis,:]) * 100
        normalized_cm = np.round(normalized_cm, 2) 
    else: 
        total_samples = np.sum(cm, axis=1)
        normalized_cm = (cm / total_samples[:, np.newaxis]) * 100
        normalized_cm = np.round(normalized_cm, 2)

    # Determine the colormap
    if isinstance(cmap, str) or isinstance(cmap, LinearSegmentedColormap):
        cmap2 = cmap
    else:
        n_shades = 100  # Adjust the number of shades as needed
        colors = [np.linspace(1, c, n_shades) for c in cmap]
        shades = np.column_stack(colors)
        cmap2 = LinearSegmentedColormap.from_list("custom_cmap", shades)

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["Wake","NREM","REM"] 
    norm = None    
    im = ax.imshow(normalized_cm, interpolation='nearest', cmap=cmap2, norm=norm)

    # Adjust the colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)  # Shrink to 80% of the axes height, aspect ratio is thinner
    im.set_clim(vmin=0, vmax=100)  # Normalize color bar for percentage values

    # Set labels and ticks
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label', fontsize=10)
    ax.set(xticks=np.arange(normalized_cm.shape[1]), yticks=np.arange(normalized_cm.shape[0]),
           xticklabels=labels, yticklabels=labels, ylabel='Manual label', xlabel='Predicted label')
    ax.tick_params(axis='both', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    thresh = normalized_cm.max() / 2.
    for i in range(normalized_cm.shape[0]):
        for j in range(normalized_cm.shape[1]):
            ax.text(j, i, f'{normalized_cm[i, j]:.1f}%',
                    ha="center", va="center",
                    color="black" if normalized_cm[i, j] > thresh else "black", fontsize=10)
    
    if precision==True:
        fig.savefig(out_path+title+"prec_EEG_cm.png", dpi=600)
        plt.close(fig)
    else: 
        fig.savefig(out_path+title+"recall_EEG_cm.png", dpi=600)
        plt.close(fig)


# ------------------------------------------------------ LOAD CONFIG FILE ------------------------------------------------------
def parse():
    """define and parse arguments for the script"""
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--test_lab', '-t', required=True,
                        help="lab to test the model on: 'Antoine', 'Kornum', 'Alessandro', 'Sebastian' or 'Maiken'"),
    parser.add_argument('--model_dir', '-m', required=True,
                        help="path to save model'")
    parser.add_argument('--color', '-c', required=True,
                        help="List of integers representing color")

    return parser.parse_args()

args       = parse()
args.color = [eval(x) for x in args.color.strip('[]').split(',')]
config_file = "/zhome/dd/4/109414/Validationstudy/slumbernet/config/slumbernet_labs.yml"
config      = ConfigLoader(config_file,create_dirs=True)  # load config from experiment
save_path  = config.output_directory+args.test_lab+"/"
model_name = args.model_dir 

if not os.path.exists(save_path):
    os.makedirs(save_path)

# ------------------------------------------------------ CREATE SPINDLE MODEL ------------------------------------------------------


input_shape   = (256,2,1)
input_layer   = keras.layers.Input(input_shape)
output_tensor = resnet_blocks(input_tensor=input_layer, 
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


# LOAD MODEL WEIGHTS 
model.load_weights(model_name)

# LOAD DATA 
dataset = SequenceDataset_test(data_folder=config.DATA_FILE,
                                  set='test',
                                  config=config,
                                  test_lab = args.test_lab)

preds_all      = [] 
true_all       = []
batch_labs_all = []

# 128 2 256 => (128, 256, 2, 1)

for batch_features, batch_labels, batch_labs in dataset:
    predictions = model.predict(batch_features)
    preds_all.append(predictions.argmax(axis=1))
    true_all.append(batch_labels)
    batch_labs_all.append(batch_labs)

print(batch_labels)
print(predictions)
print(batch_labs_all)
unique_labs    = np.unique(batch_labs_all)
true_all       = np.concatenate(true_all)
preds_all      = np.concatenate(preds_all)
batch_labs_all = np.concatenate(batch_labs_all)

# Colors = [[213/255,194/255,220/255],
# 		 [218/255,229/255,196/255], 
# 		 [187/255,212/255,233/255],
#  		 [247/255,234/255,197/255], 
#          [237/255,205/255,191/255]]

print(unique_labs)
print("after")
print(len(true_all))

for j in unique_labs:
    print(len(true_all[np.where(batch_labs_all==j)]))

    plot_cm(true_all[np.where(batch_labs_all==j)],preds_all[np.where(batch_labs_all==j)],normalize=False,cmap=args.color,
            title="lab_"+args.test_lab,out_path=save_path,precision=False,odds=False)
   
    plot_cm(true_all[np.where(batch_labs_all==j)],preds_all[np.where(batch_labs_all==j)],normalize=False,cmap=args.color,
            title="lab_"+args.test_lab,out_path=save_path,precision=True,odds=False)
    
    metrics    = cal_metrics(true_all[np.where(batch_labs_all==j)], preds_all[np.where(batch_labs_all==j)])
    metrics_df = pd.DataFrame(metrics)

    # Save the DataFrame to an Excel file
    metrics_df.to_csv(save_path+"lab_"+args.test_lab+"_metrics.csv", index=False)


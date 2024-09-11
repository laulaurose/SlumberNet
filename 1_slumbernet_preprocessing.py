#########################################################################
# SlumberNet: Sleep Stage Classification using Residual Neural Networks #
#########################################################################

# This Python script implements SlumberNet, a deep learning model for classifying
# sleep stages using residual neural networks.

# The script takes in raw sleep signals in the form of EEG and EMG data, and
# preprocesses the data for into into training and testing the model.

# The model was developed by Pawan K. Jha, Utham K. Valekunja, and Akhilesh B. Reddy,
# and the work was conducted at the Department of Systems Pharmacology & Translational
# Therapeutics, the Institute for Translational Medicine and Therapeutics, and the
# Chronobiology and Sleep Institute (CSI), Perelman School of Medicine, University
# of Pennsylvania.

# Date: May 5th, 2023


### ---- Load libraries ---- ###
import pandas as pd
import numpy as np
import os
import re
from scipy.signal import decimate, resample_poly
from scipy import signal
from datetime import datetime

### ---- Pre-process multiple files ---- ###
"""
Files are in pairs. One file (-256Hz_sco.txt) contains the epoch labels, which were
scored manually. These are either wake (W), non-REM sleep (N), or REM sleep (R), while
other labels are ignored by the model. The second file (-256Hz.txt) contains the raw
voltage data for EEG and EMG. The data is sampled at 256Hz, and each epoch is 4 seconds
"""

# Set the directory where the EEG data files are located
basedir = "/home/projects/eeg_deep_learning"
directory = f"{basedir}/eeg_training_data"
output_directory = f"{basedir}/eeg_data_preprocessed"

# Create two empty lists to store the filenames
voltage_file_list = []
epoch_file_list = []

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith("256Hz.txt"):
        # Add the filename to list1
        voltage_file_list.append(filename)
    elif filename.endswith("sco.txt"):
        # Add the filename to list2
        epoch_file_list.append(filename)

# Create a Pandas dataframe with two columns
file_list = pd.DataFrame(columns=["voltage_file", "epoch_file"])

# Loop through the values in list1
for voltage_file in voltage_file_list:
    # Use regular expressions to match the beginning of the filename
    match = re.match(r"^(.*)-256Hz.txt", voltage_file)
    if match:
        # Get the matching part of the filename
        stem = match.group(1)
        # Loop through the values in list2
        for epoch_file in epoch_file_list:
            # Check if the filename also matches the beginning of a value in list2
            if epoch_file.startswith(stem):
                # Add the values to the dataframe
                file_list = pd.concat([file_list, 
                                    pd.DataFrame({"voltage_file": voltage_file, "epoch_file": epoch_file}, index=[0])], 
                                    ignore_index=True)
                # Break out of the inner loop
                break

# Read first few relevant lines of voltage files and check that they are in the correct format
# i.e. both voltage columns are float64. Exclude file pair from list if that isn't the case (i.e. don't process later)
# Takes about 5 mins for 80 files
indexes_to_delete = []

for file_number in range(0,len(file_list)):
    current_voltage_file_name = file_list.loc[file_number, "voltage_file"]
    voltage_file_name = f'{directory}/{current_voltage_file_name}'
    voltages = pd.read_table(voltage_file_name)
    if voltages.loc[0].str.contains(",").any():      # This is to check that it is a voltage file; if not add to deletes
        indexes_to_delete.append(file_number)
    else:    
        voltages = pd.read_table(voltage_file_name, header=None, delimiter='\s+', skiprows=0, 
                                low_memory=False)    # Needed if a column is a string and cannot convert easily
        voltages.columns = ["eeg_voltage", "emg_voltage"]                        
        if voltages["eeg_voltage"].dtype != 'float64' or voltages["emg_voltage"].dtype != 'float64':
            indexes_to_delete.append(file_number)

file_list = file_list.drop(file_list.index[indexes_to_delete])
file_list.reset_index(drop=True, inplace=True)       # To make sure indexes are sequential in final data frame

### ---- Pre-process the data for input into Keras/Tensorflow model ---- ###
# Takes about 1.5 mins per file pair

number_of_samples_per_second = 256          # 256Hz sampling rate - change if different
number_seconds_per_epoch = 4                # 4 second epochs - change if different
samples_per_epoch = number_of_samples_per_second * number_seconds_per_epoch

for file_index in range(0,len(file_list)):

    # Output which file is being worked on and datetime at start
    current_file_number = file_index + 1
    last_file_number = len(file_list)
    
    print(f'Processing file set {current_file_number} of {last_file_number}.')
    current_voltage_file_name = file_list.loc[file_index, "voltage_file"]
    print(f'File index: {file_index}, Filename: {current_voltage_file_name}.')
    current_dateTime = datetime.now()
    print(f'Started at {current_dateTime}.') 

    # Load data from voltages file into a dataframe:
    voltage_file_name = f'{directory}/{current_voltage_file_name}'
    voltages = pd.read_table(voltage_file_name, header=None, delimiter='\s+', skiprows=0)
    voltages.columns = ["eeg_voltage", "emg_voltage"]

    # Load data from epochs file with wake (W), REM(R), NREM(N). Anything else (!= WRN) is Artefact (A):
    epoch_file_name = f'{directory}/{file_list.loc[file_index, "epoch_file"]}'
    epochs = pd.read_table(epoch_file_name, header=None, delimiter=',', skiprows=19)
    if epochs.loc[0].str.contains("\t").any():
        epochs = pd.read_table(epoch_file_name, header=None, delimiter='\t', skiprows=19)
    epochs.columns = ["datetime", "epoch_num", "sleep_stage", "n", "blank"]
    epochs = epochs.drop(columns = ["n", "blank"])

    # Extract voltage data for each epoch and matching sleep_stage:
    voltage_array = np.zeros((1,2,samples_per_epoch))    # Reset arrays for each file iteration
    epoch_array = np.zeros((1,3))                        # Set initial epoch to [0,0,0] as this label is not used

    # Pull out and label sleep_stage if W, N, R; anything else label with A (artefact): 
    for pointer_index_epoch in range(0,len(epochs)):
        current_sleep_stage = epochs.loc[pointer_index_epoch, "sleep_stage"]
        if ~epochs.loc[pointer_index_epoch].str.contains("W|N|R").any():       # if not WNR, call it A
            current_sleep_stage = "A"
        start_index_voltage = pointer_index_epoch * samples_per_epoch
        end_index_voltage = start_index_voltage + samples_per_epoch - 1

        # Grab voltages for current epoch and add to np array
        current_eeg_voltage_array = voltages.loc[start_index_voltage:end_index_voltage, "eeg_voltage"].to_numpy()
        current_emg_voltage_array = voltages.loc[start_index_voltage:end_index_voltage, "emg_voltage"].to_numpy()
        current_voltage_array = np.stack((current_eeg_voltage_array, current_emg_voltage_array), axis = 0)
        if np.all(voltage_array == np.zeros((samples_per_epoch))):
            voltage_array[0] = current_voltage_array.copy()
        else:
            current_voltage_array = current_voltage_array.reshape((1,2,samples_per_epoch))
            voltage_array = np.vstack((voltage_array, current_voltage_array))

        # Make np array for epochs in matching order. 
        # One-hot encoding: W is [1,0,0], N is [0,1,0], R is [0,0,1]. A is [1,1,1] (removed later)
        if current_sleep_stage == "W":
            current_epoch_array = np.array([1,0,0], dtype = "float64")
        elif current_sleep_stage == "N":     
            current_epoch_array = np.array([0,1,0], dtype = "float64")
        elif current_sleep_stage == "R":     
            current_epoch_array = np.array([0,0,1], dtype = "float64")
        elif current_sleep_stage == "A":     
            current_epoch_array = np.array([1,1,1], dtype = "float64")

        if np.all(epoch_array == np.zeros((1,3))):              # if initial array
            epoch_array[0] = current_epoch_array.copy()
        else:
            current_epoch_array = current_epoch_array.reshape((1,3))
            epoch_array = np.vstack((epoch_array, current_epoch_array))                    

    # Update master voltage array with data from latest file        
    if file_index == 0:
        master_voltage_array = voltage_array.copy()
    else:
        voltage_array = voltage_array.reshape((-1,2,samples_per_epoch))
        master_voltage_array = np.vstack((master_voltage_array, voltage_array))

    # Update master epoch array with data from latest file   
    if file_index == 0:
        master_epoch_array = epoch_array.copy()
    else:
        epoch_array = epoch_array.reshape((-1,3))
        master_epoch_array = np.vstack((master_epoch_array, epoch_array))        

    # Output datetime when file has finished being processed
    current_dateTime = datetime.now()
    print(f'Finished at {current_dateTime}.')
    number_of_epochs_added = master_epoch_array.shape[0]
    number_of_voltages_added = master_voltage_array.shape[0]
    print(f'Epochs count: {number_of_epochs_added}, Voltage count: {number_of_voltages_added}.\n')

# Save all data
np.save(f"{output_directory}/eeg_voltage_data_no_scaling_all.npy", master_voltage_array)
np.save(f"{output_directory}/eeg_epoch_data_no_scaling_all.npy", master_epoch_array)

# Delete "A": [1,1,1] labelled epochs and corresponding voltage data
A_list = []
for i in range(master_epoch_array.shape[0]):
    if np.all(master_epoch_array[i] == np.array([1,1,1])):
        A_list.append(i)    
master_epoch_array = np.delete(master_epoch_array, A_list, axis = 0)
master_voltage_array = np.delete(master_voltage_array, A_list, axis = 0)

# Save data trimmed of "A" epochs
np.save(f"{output_directory}/eeg_voltage_data_no_scaling.npy", master_voltage_array)
np.save(f"{output_directory}/eeg_epoch_data_no_scaling.npy", master_epoch_array)

### --- Downsample EEG and EMG, and baseline correct each epoch ---- ###

# Apply downsampling to EEG data and baseline subtract (in case of baseline shift; more a problem with EMG)
def pre_process_eeg(sample_index, method="resample"):
    # Downsample data
    down_sample_ratio = 4        # Downsample by 4 (1024 -> 256 datapoints / epoch)
    original_num_samples = master_voltage_array[sample_index,0].shape[0]
    desired_num_samples = master_voltage_array[sample_index,0].shape[0] // down_sample_ratio

    if method == "decimate":
        down_sampled = decimate(master_voltage_array[sample_index,0], q=down_sample_ratio) 
    elif method == "resample":   
        down_sampled = resample_poly(master_voltage_array[sample_index,0], desired_num_samples, original_num_samples)

    # Baseline correct data
    time = np.arange(len(down_sampled))
    fit = np.polyfit(time, down_sampled, 12)
    fit_curve = np.polyval(fit, time)
    baseline_corrected = down_sampled - fit_curve

    return baseline_corrected

input_eeg_array = [pre_process_eeg(i) for i in np.arange(master_voltage_array.shape[0])] 
input_eeg_array = np.array(input_eeg_array)

def pre_process_emg(sample_index, method="resample"):
    orig_data = master_voltage_array[sample_index,1]         # EMG data are in index 1

    # Butterworth low pass filter
    cutoff = 16         # Set the cutoff frequency for the filter (in Hz)
    sample_rate = 256   # Set the sample rate for the data (in Hz)
    order = 4           # Set the order of the Butterworth filter
    b, a = signal.butter(order, cutoff / (sample_rate / 2), 'low')  # Create the Butterworth filter
    filtered_data = signal.lfilter(b, a, orig_data)                 # Apply the filter 

    # Downsample data
    down_sample_ratio = 4        # Downsample by 4 (1024 -> 256 datapoints / epoch)
    original_num_samples = filtered_data.shape[0]
    desired_num_samples = filtered_data.shape[0] // down_sample_ratio

    if method == "decimate":
        down_sampled = decimate(filtered_data, q=down_sample_ratio) 
    elif method == "resample":   
        down_sampled = resample_poly(filtered_data, desired_num_samples, original_num_samples)

    time = np.arange(len(down_sampled))
    fit = np.polyfit(time, down_sampled, 12)
    fit_curve = np.polyval(fit, time)
    baseline_corrected = down_sampled - fit_curve

    return baseline_corrected

input_emg_array = [pre_process_emg(i) for i in np.arange(master_voltage_array.shape[0])]  
input_emg_array = np.array(input_emg_array)

input_array = np.zeros((input_eeg_array.shape[0], 2, input_eeg_array.shape[1]))
for sample_index in np.arange(input_eeg_array.shape[0]):
    current_array = np.vstack((input_eeg_array[sample_index], input_emg_array[sample_index]))
    input_array[sample_index] = current_array

# Save data input_array - these are the data that will be used for training and testing
np.save(f"{output_directory}/eeg_input_array.npy", input_array)
np.save(f"{output_directory}/epoch_input_array.npy", master_epoch_array)

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

lr1 = np.load("/work3/laurose/SlumberNet/models/orig/run1/array.npy") 
lr2 = np.load("/work3/laurose/SlumberNet/models/orig/run2/array.npy") 
lr3 = np.load("/work3/laurose/SlumberNet/models/orig/run3/array.npy") 
lr4 = np.load("/work3/laurose/SlumberNet/models/orig/run4/array.npy") 
lr5 = np.load("/work3/laurose/SlumberNet/models/orig/run5/array.npy") 

lr_stack = np.vstack((lr1, lr2, lr3, lr4, lr5))

# Calculate the mean across the 5 lr arrays (along axis 0)
mean_lr = np.mean(lr_stack, axis=0)

# Calculate the standard error of the mean (SEM)
sem_lr = np.std(lr_stack, axis=0) / np.sqrt(lr_stack.shape[0])

# Generate x values
x = np.arange(0, len(lr1))

# Plot the mean with shaded standard error
plt.figure(figsize=(10, 6))
plt.plot(x, mean_lr, color='blue', label='Mean LR')
plt.fill_between(x, mean_lr - sem_lr, mean_lr + sem_lr, color='blue', alpha=0.2, label='SEM')
plt.xlabel('Epoch')
plt.ylabel('Validation accuracy')
plt.title('Averaged learning curve')
plt.legend()
plt.grid(True)
plt.savefig("/zhome/dd/4/109414/Validationstudy/slumbernet/results/reproduction/avr_lr.png")

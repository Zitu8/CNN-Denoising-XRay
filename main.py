#!/usr/bin/env python3
import os

RNG_SEED = 0 # random seed to be used
USE_GPU = True # whether or not to use a GPU
GPU_ID = 0 # GPU index (if multiple GPUs available)
GPU_MEM = 10000 # GPU memory to use (MB)

os.environ["PYTHONHASHSEED"] = str(RNG_SEED)

if USE_GPU is True:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

import tensorflow as tf
import sys

from models import VDSR, IRUNet
from loss_functions import mae_ssim_ms_loss

import helper_functions as hf

if USE_GPU is True:
    # Enable memory growth for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    

def lr_scheduler(epoch, lr):
    """Example of a learning rate scheduler to be used with the keras.callbacks.LearningRateScheduler method.

    Parameters
    ----------
    epoch : int
        current epoch
    lr : float
        current learning rate

    Returns
    -------
    updated learning rate for the current epoch
    """

    if lr < 100:
        return lr
    else:
        return lr*0.5**(epoch//100)

def main():

    # Reset the random seed
    hf.reset_random_seeds(RNG_SEED)

    # If the data is in the form of HDF5, convert to TIF
    generate_tif_from_hdf5 = True

    # Define some folders
    base_folder = "output" # path to base folder
    train_folder = "data/diffraction_data/training" # path to folder containing training data
    valid_folder = "data/diffraction_data/validation" # path to folder containing validation data

    # Define training parameters
    model_name = "trained_model" # name of the model (will be the folder containing the outputs within the base folder) 
    input_shape = (192,240,1) # input shape of the data frames
    n_epochs = 100 # number of training epochs
    batch_size = 8 # batch size to be used for the training
    simulate_low_count = None # whether to simulate the low-count data; None, 'poisson' or 'gaussian'
    lrs = lr_scheduler # whether or not to use a learning rate scheduler callback; None or function (e.g. lrs = lr_scheduler)

    # Define the model to be trained
    model = IRUNet(input_shape=input_shape)
    
    # Compile the model with a certain loss and optimizer
    model.compile(loss=mae_ssim_ms_loss(max_val_ssim=1., alpha=0.7), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, amsgrad=True), metrics=['mae'])

    # Initialize the dictionary for potentially simulating the low-count data
    sim_lc = {'type': None, 'constant': None}
    sim_lc['type'] = simulate_low_count

    if simulate_low_count is not None:
        if simulate_low_count == 'gaussian':
            # Use either the average mean and standard deviation of all low-count frames
            # The noise is then obtained from a normal distribution using these two parameters
            _, gmean, _, gstd = hf.calculate_global_mean_and_std(train_folder+"/LC") # average mean and standard deviation of all low-count files
            print(f"Global normalized mean value of LC: {gmean}")
            print(f"Global normalized standard deviation value of LC: {gstd}")
            sim_lc['constant'] = [gmean, gstd] # custom values can be used instead of the global mean and standard deviation
        elif simulate_low_count == 'poisson':
            # Define multiplicative factor lambda as the mean or median over the sum of low-count intensity
            # divided by the sum of high-count intensity for all frame pairs
            # The noise is then obtained from the Poisson distribution of lambda*high-count
            lam = hf.calculate_global_lambda(train_folder+"/LC", train_folder+"/HC", type='median')
            print(f"Global lambda value: {lam}")
            sim_lc['constant'] = lam # custom value can be used instead of the global lambda
        else:
            raise ValueError("The type of simulated noise has to be either 'poisson' or 'gaussian'.")

    if generate_tif_from_hdf5:
        training_file = "data/training_data.hdf5" # path to HDF5 file containing the training data
        validation_file = "data/validation_data.hdf5" # path to the HDF5 file containing the validation data
        hf.convert_zenodo_hdf5_to_tif(training_file, save_folder=train_folder)
        hf.convert_zenodo_hdf5_to_tif(validation_file, save_folder=valid_folder)

    # Call to training function (add potential custom objects for the compiling step)
    hf.train_model(base_folder, model_name, model, input_shape, train_folder, valid_folder, n_epochs, batch_size, sim_lc, rng_seed=RNG_SEED, lrs=lrs) 

if __name__ == '__main__':
    hf.reset_random_seeds(RNG_SEED)
    main()

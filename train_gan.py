# Code to do WGAN-GP training
# NOTE: don't attempt to run this in 'dft' or 'dct' configuration without changing the data loading method,
#   unless you have a lot of RAM (something like >= 24GB); as the method used to compute the transforms are
#   rather inefficient memory-wise for the sake of speed

import os, sys, errno
import glob
import numpy as np
import tensorflow as tf
import keras
import models
from features import dataset_utils

# ---------------------------------------------------------------------------------------------------------------------
# Training Configuration
#   DATASET_PATTERN: glob pattern for all .wav files to train on
#   OUT_DIR: output directory, will contain the following files:
#       model_g.h5 - the saved generator model weights (note that only the weights are saved)
#       model_d.h5 - the saved discriminator model weights (note that only the weights are saved)
#       normalization.npy - normalization factors that should be divided element-wise with generator output before
#           synthesizing the output audio
#   DATA_REPR: data representation to use
#       'dft' for DFT images, 'dct' for DCT images, anything else for raw waveform
#   USE_IL: use ILConv in the generator; automatically is true for raw waveform representation
#   BIAS_OUT: use bias at the last layer of the generator
#   G_OPTIM: generator optimizer, defaults to keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
#   D_OPTIM: discriminator optimizer, defaults to keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
#   D_STEPS_PER_G: number of iterations to train the discriminator per iteration of generator training, must be >= 1
# ---------------------------------------------------------------------------------------------------------------------
DATASET_PATTERN = '/vulcan/scratch/bzhi/sc09/train/*.wav'
OUT_DIR = '/vulcan/scratch/bzhi/ildct2_out'
DATA_REPR = 'dct'
USE_IL = True
BIAS_OUT = True
BATCH_SIZE = 64
MODEL_SIZE = 64
NUM_LATENT = 100
NUM_EPOCH_PER_CHECKPOINT = 50
NUM_EPOCH = 400
G_OPTIM = None
D_OPTIM = None
D_STEPS_PER_G = 5

# make output directory
try:
    os.makedirs(OUT_DIR)
except OSError as e:
    if (e.errno != errno.EEXIST):
        raise e
    elif (os.path.isfile(OUT_DIR)):
        print('output directory is an existing file; exiting')
        sys.exit()
    elif (len(os.listdir(OUT_DIR)) > 0):
        print('output directory is non-empty; exiting')
        sys.exit()

# 'labels' used to apply in computing the Wasserstein loss
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32) # need to feed in some labels for gradient penalty loss

# create and build model
if (DATA_REPR == 'dft'):
    gen_func = models.dft_generator if USE_IL else models.dft_generator_tr
    disc_func = models.dft_discriminator
elif (DATA_REPR == 'dct'):
    gen_func = models.dct_generator if USE_IL else models.dct_generator_tr
    disc_func = models.dct_discriminator
else:
    gen_func = models.wave_generator
    disc_func = models.wave_discriminator
generator = gen_func(MODEL_SIZE, NUM_LATENT, BIAS_OUT)
discriminator = disc_func(MODEL_SIZE)
g_m, d_m = models.create_wgan(generator, discriminator)

# read in dataset and transform it appropriately
datalist = glob.glob(DATASET_PATTERN)
data = dataset_utils.read_wav_dataset(datalist)
num_samples = data.shape[0]
if (DATA_REPR == 'dft'):
    data = dataset_utils.dft_transform_forward(data)
elif (DATA_REPR == 'dct'):
    data = dataset_utils.dct_transform_forward(data)
else:
    data = np.expand_dims(data, axis=-1)

# normalize data
if (DATA_REPR == 'dft' or DATA_REPR == 'dct'):
    if (DATA_REPR == 'dft'):
        data_mag = np.abs(data[:,:,:,0:1] + 1j * data[:,:,:,1:2])
    else:
        data_mag = np.abs(data)
    amp_max = np.max(data_mag, axis=(0,-2,-1), keepdims=True)
    norm_f = 0.9/amp_max
else:
    norm_f = 1.0
np.save(os.path.join(OUT_DIR, 'normalization.npy'), norm_f)

# begin training
model_path_g = os.path.join(OUT_DIR, 'model_g_{:d}.h5')
model_path_d = os.path.join(OUT_DIR, 'model_d_{:d}.h5')
idx = np.arange(num_samples)
for epoch_i in range(0, NUM_EPOCH * D_STEPS_PER_G):
    np.random.shuffle(idx)
    for superbatch_i in range(0, num_samples // (BATCH_SIZE * D_STEPS_PER_G)):
        batch_idx = idx[(superbatch_i*BATCH_SIZE*D_STEPS_PER_G):((superbatch_i+1)*BATCH_SIZE*D_STEPS_PER_G)]
        # train discriminator
        for batch_i in range(0, D_STEPS_PER_G):
            real = data[batch_idx[(batch_i*BATCH_SIZE):((batch_i+1)*BATCH_SIZE)],...] * norm_f
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, NUM_LATENT)).astype(np.float32)
            d_loss = d_m.train_on_batch([real, noise], [positive_y, negative_y, dummy_y])
        # train generator
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, NUM_LATENT)).astype(np.float32)
        g_loss = g_m.train_on_batch(noise, positive_y)
    if (epoch_i % D_STEPS_PER_G == 0):
        print('Epoch {:d}:'.format(epoch_i // D_STEPS_PER_G))
        print('\t D loss: {:f}, grad penalty: {:f}'.format(d_loss[1]+d_loss[2], d_loss[3]))
        print('\t G loss: {:f}'.format(g_loss), flush=True)
        if ((epoch_i // D_STEPS_PER_G) % NUM_EPOCH_PER_CHECKPOINT == 0):
            generator.save_weights(model_path_g.format(epoch_i // D_STEPS_PER_G))
            discriminator.save_weights(model_path_d.format(epoch_i // D_STEPS_PER_G))
generator.save_weights(model_path_g.format(NUM_EPOCH))
discriminator.save_weights(model_path_d.format(NUM_EPOCH))

import os, sys, errno
import argparse

parser = argparse.ArgumentParser(description="""Generate a set of wav files using a provided keras model,
                                                which should take uniform noise ~[-1,1] as input.""")
parser.add_argument('-model', dest='model_path', metavar='model_path', type=str, required=True, help='file containing the weights of the model to load')
parser.add_argument('-norm', dest='norm_path', metavar='norm_path', type=str, required=True, help='file containing the normalization factors used')
parser.add_argument('-repr', dest='repr', metavar='representation', type=str, required=True, help='output type: dft, dct, or anything else for raw waveform')
parser.add_argument('-il', dest='use_il', action='store_const', default=False, const=True, help='use ILConv to bulid model (always True for raw waveform)')
parser.add_argument('-bias', dest='use_bias', action='store_const', default=False, const=True, help='generator output layer has a bias')
parser.add_argument('-o', dest='out_dir', metavar='output_dir', type=str, required=True, help='output directory to place wav files')
parser.add_argument('-lsize', dest='latent_size', metavar='latent_size', type=int, required=True, help='number of latent dimensions')
parser.add_argument('-msize', dest='model_size', metavar='model_size', type=int, required=True, help='model size')
parser.add_argument('-n', dest='num_wavs', metavar='num_wavs', type=int, default=50000, help='number of wav files to generate')
parser.add_argument('-b', dest='batch_size', metavar='batch_size', type=int, default=50, help='number of wav files to generate per network run')
parser.add_argument('-r', dest='sample_rate', metavar='sample_rate', type=int, default=16000, help='sample rate of output files')
parser.add_argument('-v', dest='verbose', action='store_const', default=False, const=True, help='output per-batch progress')
args = parser.parse_args()

if (args.num_wavs < 1):
    print('number of files to generate is zero or negative; exiting')
    sys.exit()
if (args.batch_size < 1):
    print('invalid batch size; exiting')
    sys.exit()
if (args.sample_rate < 1):
    print('invalid output sample rate; exiting')
    sys.exit()
if (args.latent_size < 1):
    print('invalid latent size; exiting')
    sys.exit()
if (args.model_size < 1):
    print('invalid model size; exiting')
    sys.exit()
try:
    os.makedirs(args.out_dir)
except OSError as e:
    if (e.errno != errno.EEXIST):
        raise e
    elif (os.path.isfile(args.out_dir)):
        print('output directory is an existing file; exiting')
        sys.exit()
    elif (len(os.listdir(args.out_dir)) > 0):
        print('output directory is non-empty; exiting')
        sys.exit()

import numpy as np
import keras
import models
from features import dataset_utils

# create and build model
norm_f = np.load(args.norm_path)
if (args.repr == 'dft'):
    gen_func = models.dft_generator if args.use_il else models.dft_generator_tr
elif (args.repr == 'dct'):
    gen_func = models.dct_generator if args.use_il else models.dct_generator_tr
else:
    gen_func = models.wave_generator
generator = gen_func(args.model_size, args.latent_size, args.use_bias)
generator.load_weights(args.model_path)

# generate audio
wav_i = 0
z_rs = np.random.RandomState(seed=177013)
while (wav_i < args.num_wavs):
    batch_size = min(args.batch_size, args.num_wavs - wav_i)
    if (args.verbose):
        print('generating files {:d}-{:d} (out of {:d})...'.format(wav_i, wav_i + batch_size - 1, args.num_wavs))
    z_in = z_rs.uniform(low=-1, high=1, size=(batch_size, args.latent_size)).astype(np.float32)
    G_z = generator.predict(z_in, batch_size=batch_size) / norm_f
    if (args.repr == 'dft'):
        G_z = dataset_utils.dft_transform_backward(G_z)
    elif (args.repr == 'dct'):
        G_z = dataset_utils.dct_transform_backward(G_z)
    else:
        G_z = np.squeeze(G_z)
    dataset_utils.write_wav_dataset(G_z, args.out_dir, fname_init=wav_i)
    wav_i += batch_size
if (args.verbose):
    print('done')

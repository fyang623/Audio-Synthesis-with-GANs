import os, sys, errno
import argparse

parser = argparse.ArgumentParser(description="""Generate a set of wav files using a provided tensorflow model,
                                                which should take uniform noise ~[-1,1] as input.""")
parser.add_argument('-meta', dest='meta_path', metavar='meta_path', type=str, required=True, help='file containing the metagraph of the model to load')
parser.add_argument('-ckpt', dest='ckpt_path', metavar='ckpt_path', type=str, required=True, help='checkpoint file prefix of the model to load')
parser.add_argument('-o', dest='out_dir', metavar='output_dir', type=str, required=True, help='output directory to place wav files')
parser.add_argument('-ishape', dest='input_shape', metavar='input_shape', type=int, nargs='+', required=True, help='input shape')
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
if (len(args.input_shape) < 1 or any(input_dim < 1 for input_dim in args.input_shape)):
    print('invalid input shape; exiting')
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
import tensorflow as tf
from scipy.io import wavfile

tf.reset_default_graph()
saver = tf.train.import_meta_graph(args.meta_path)
graph = tf.get_default_graph()
with tf.Session() as sess:
    saver.restore(sess, args.ckpt_path)
    wav_i = 0
    z_rs = np.random.RandomState(seed=177013)
    while (wav_i < args.num_wavs):
        batch_size = min(args.batch_size, args.num_wavs - wav_i)
        if (args.verbose):
            print('generating files {:d}-{:d} (out of {:d})...'.format(wav_i, wav_i + batch_size - 1, args.num_wavs))
        feed_dict = {}
        z_in = z_rs.uniform(low=-1, high=1, size=tuple([batch_size] + args.input_shape)).astype(np.float32)
        z = graph.get_tensor_by_name('z:0')
        feed_dict[z] = z_in
        try:
            ngl = graph.get_tensor_by_name('ngl:0')
            feed_dict[ngl] = 16
        except KeyError:
            pass
        G_z = graph.get_tensor_by_name('G_z_int16:0')
        G_z_out = sess.run(G_z, feed_dict)
        for out_i in range(0, G_z_out.shape[0]):
            try:
                fname_out = os.path.join(args.out_dir, '{:d}.wav'.format(wav_i + out_i))
                wavfile.write(fname_out, args.sample_rate, G_z_out[out_i,:,:])
            except IOError:
                print('error writing out file {:s}; skipped'.format(fname_out))
        wav_i += batch_size
    if (args.verbose):
        print('done')

import os, sys, errno
import argparse

parser = argparse.ArgumentParser(description="""Concatenates multiple .wav files into one, with some silence between them.
                                                All files must be mono, have the same samplerate, and be stored as 16-bit int.
                                                Additionally, all files are roughly normalized to the same peak (I couldn't find
                                                a good LUFS loudness implementation to use; this is a possible improvement)""")
parser.add_argument('-i', dest='input_paths', metavar='input_path', type=str, required=True, nargs='+', help='input .wav filenames')
parser.add_argument('-o', dest='output_path', metavar='output_path', type=str, required=True, help='output .wav filename')
parser.add_argument('-s', dest='silence', metavar='silence', type=float, default=0.5, help='amount of silence to place between files, in seconds')
parser.add_argument('-y', dest='overwrite', action='store_const', default=False, const=True, help='overwrite output file if it exists')
args = parser.parse_args()

if ((not args.overwrite) and os.path.exists(args.output_path)):
    print('output file path exists; exiting')
    sys.exit()

import numpy as np
from scipy.io import wavfile

sample_rate = None
input_signals = []
signal_length = 0
feps = np.finfo(np.float32).eps
for file_path in args.input_paths:
    srate, wavdata = wavfile.read(file_path)
    if (len(wavdata.shape) != 1):
        print('input file {:s} either empty or not mono; exiting'.format(file_path))
        sys.exit()
    if (wavdata.dtype != np.int16):
        print('input file {:s} is not a 16-bit int wav; exiting'.format(file_path))
        sys.exit()
    if (sample_rate is None):
        sample_rate = srate
    elif (srate != sample_rate):
        print('input file {:s} has inconsistent sample rate from previous files; exiting'.format(file_path))
        sys.exit()
    # renormalize data based on peak, with care to prevent clipping or excessive amplification
    wavdata_f = wavdata.astype(np.float32) / 32767
    wavdata_f *= np.clip(0.9 / (np.max(np.abs(wavdata_f)) + feps), a_min=1, a_max=8)
    input_signals.append(np.clip(wavdata_f * 32767, -32767, 32767).astype(np.int16))
    signal_length += wavdata.shape[0] + int(sample_rate * args.silence)
signal_length -= int(sample_rate * args.silence)

output_signal = np.zeros((signal_length,), dtype=np.int16)
sig_start = 0
for input_signal in input_signals:
    sig_end = sig_start + input_signal.shape[0]
    output_signal[sig_start:sig_end] = input_signal
    sig_start = sig_end + int(sample_rate * args.silence)
wavfile.write(args.output_path, sample_rate, output_signal)

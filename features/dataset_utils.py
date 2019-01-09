import os
import numpy as np
from scipy import fftpack, signal
from scipy.io import wavfile

# Constants
# with the current configuration:
#   dft_transform - outputs images of shape (128,128,2), excluding the batch dimension
#       nyquist frequency is trimmed, 1/2 overlap w/ Hann window, last 128 samples taper off
#   dct_transform - outputs images of shape (256,256,1), excluding the batch dimension
#       3/4 overlap w/ Hann window, last 192 samples taper off
WAV_LENGTH = 16384 # length of each audio sample (in time-domain)
WAV_SAMPLE_RATE = 16000 # sample rate of each audio sample
DFT_FRAME_SIZE = 256
DFT_FRAME_OVER = 128
DFT_WINDOW = 'hann' # assert signal.check_COLA(DFT_WINDOW, DFT_FRAME_SIZE, DFT_FRAME_OVER)
DFT_LOG_EPS = np.finfo(np.float32).eps # numerical stability for log-scaling near zero
DCT_FRAME_SIZE = 256
DCT_FRAME_OVER = 192
DCT_WINDOW = 'hann' # assert signal.check_COLA(DCT_WINDOW, DCT_FRAME_SIZE, DCT_FRAME_OVER)
DCT_COLA = 2.0 # adding up 3/4-overlapped Hann windows yields double the gain

# reads all .wav files in file_list, reading only the first WAV_LENGTH samples (zero-padding if a file is shorter)
# outputs a numpy array of shape (len(file_list), WAV_LENGTH)
# NOTE: .wav files must be mono, 16-bit int format, and (probably) should all be the same samplerate
def read_wav_dataset(file_list):
    data = np.zeros((len(file_list), WAV_LENGTH), dtype=np.float32)
    for (file_i, file_path) in enumerate(file_list):
        sample_rate, wavdata = wavfile.read(file_path)
        if (sample_rate != WAV_SAMPLE_RATE):
            raise ValueError("The sample rate of file {:d} is inconsistent".format(file_path))
        wavlen = min(WAV_LENGTH, wavdata.shape[0])
        data[file_i,:wavlen] = wavdata[:wavlen].astype(np.float32) / 32767
    return data

# writes all rows in data to .wav files, in 16-bit mono format to folder out_dir
# output filenames will be formatted as "{:d}.wav".format(file_i) for file_i in range(fname_init, fname_init + data.shape[0])
# NOTE: out_dir must already exist
def write_wav_dataset(data, out_dir, fname_init=0):
    data_i16 = np.clip(data * 32767, -32767, 32767).astype(np.int16)
    for file_i in range(0, data_i16.shape[0]):
        fname_out = os.path.join(out_dir, '{:d}.wav'.format(file_i + fname_init))
        wavfile.write(fname_out, WAV_SAMPLE_RATE, data_i16[file_i,:])

# transforms time-domain signals into time-frequency images with real and imaginary channels via the short-time DFT
# optionally also log-scales the amplitudes via f(z) = (z/|z|) * log10(1 + |z|),
# where z are the complex values output by the DFT
def dft_transform_forward(data, logscale=True):
    data_complex = signal.stft(data, window=DFT_WINDOW, nperseg=DFT_FRAME_SIZE, noverlap=DFT_FRAME_OVER, axis=-1)[-1].astype(np.complex64)
    # trim nyquist freq and last time window for (128,128) spatial dimensions
    data_complex = data_complex[:,:-1,:-1]
    if (logscale):
        data_complex *= np.log10(np.abs(data_complex) + 1.0) / np.maximum(np.abs(data_complex), DFT_LOG_EPS)
    return np.stack([data_complex.real, data_complex.imag], axis=-1)

# transforms time-frequency images generated via dft_transform_forward back to time-domain signals
# optionally also undoes the amplitude log-scaling via g(z) = (z/|z|) * (10^(|z|) - 1)
def dft_transform_backward(data, logscale=True):
    data_complex = data[:,:,:,0] + 1j * data[:,:,:,1]
    if (logscale):
        data_complex *= (np.power(10.0, np.abs(data_complex)) - 1.0) / np.maximum(np.abs(data_complex), DFT_LOG_EPS)
    data_complex = np.pad(data_complex, ((0,0), (0,1), (0,1)), 'constant')
    return signal.istft(data_complex, window=DFT_WINDOW, nperseg=DFT_FRAME_SIZE, noverlap=DFT_FRAME_OVER)[-1].astype(np.float32)

# transforms time-domain signals into time-frequency images with real channels via the short-time DCT (type-II)
# optionally also log-scales the amplitudes via f(x) = sgn(x) * log10(1 + |x|),
# where x are the real values output by the DCT
def dct_transform_forward(data, logscale=True):
    frame_step = DCT_FRAME_SIZE - DCT_FRAME_OVER
    window_func = np.sqrt(np.expand_dims(signal.get_window(DCT_WINDOW, DCT_FRAME_SIZE), axis=0))
    data_dct = np.empty((data.shape[0], DCT_FRAME_SIZE, WAV_LENGTH // frame_step, 1), dtype=np.float32)
    for t_i in range(0, data_dct.shape[-2]):
        t_start = t_i * frame_step - DCT_FRAME_OVER
        t_end = (t_i + 1) * frame_step
        if (t_start < 0):
            data_window = np.pad(data[:,:t_end], ((0,0), (-t_start,0)), 'constant')
        else:
            data_window = data[:,t_start:t_end]
        data_dct[:,:,t_i,0] = fftpack.dct(data_window * window_func, type=2, n=DCT_FRAME_SIZE, axis=-1, norm='ortho', overwrite_x=True)
    if (logscale):
        data_dct = np.sign(data_dct) * np.log10(np.abs(data_dct) + 1.0)
    return data_dct

# transforms time-frequency images generated via dct_transform_forward back to time-domain signals
# optionally also undoes the amplitude log-scaling via g(x) = sgn(x) * (10^(|x|) - 1)
def dct_transform_backward(data, logscale=True):
    frame_step = DCT_FRAME_SIZE - DCT_FRAME_OVER
    window_func = np.sqrt(np.expand_dims(signal.get_window(DCT_WINDOW, DCT_FRAME_SIZE), axis=0))
    data_out = np.zeros((data.shape[0], WAV_LENGTH), dtype=np.float32)
    if (logscale):
        data = np.sign(data) * (np.power(10.0, np.abs(data)) - 1.0)
    for t_i in range(0, data.shape[-2]):
        t_start = t_i * frame_step - DCT_FRAME_OVER
        t_end = (t_i + 1) * frame_step
        data_window = fftpack.dct(data[:,:,t_i,0], type=3, n=DCT_FRAME_SIZE, axis=-1, norm='ortho', overwrite_x=True)
        data_window *= window_func
        if (t_start < 0):
            data_window = data_window[:,-t_start:]
            t_start = 0
        data_out[:,t_start:t_end] += data_window / DCT_COLA
    return data_out

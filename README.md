# Audio Syntheis Using Generative Adversarial Networks
A team project for CMSC723 at the University of Maryland College Park, fall 2018. 

In this project our team came up with an approach to unsupervised audio synthesis using GANs on invertible image-like audio representations, and demonstrated that our method was competitive with the state-of-the-art raw waveform baseline while requiring less than half the training time. 

For details of our work, please refer to our project report here.

## Introduction
As an attempt to adapt image-generating GANs to audio, Donahue et al. proposed two different approaches to generating fixed-length audio segments based on the DCGAN architecture: SpecGAN and WaveGAN. SpecGAN operates on image-like 2D magnitude spectrograms of audio, but the process of converting audio into such spectrograms lacks an inverse, as phase information is discarded. WaveGAN instead operates on raw audio waveforms; however, artifacts caused by convolutional operations become more troublesome to mitigate in the audio domain. 

Here we extend the work of Donahue. We only focused on SpecGAN and did not put efforts on WaveGAN as we believed frequency information is more representative of human sound perception. The main improvements we made over SpecGAN are summarized below.
- we use invertible time-frequency representations of audio, so that the generated audio does not suffer the significant distortion introduced by approximate inversion methods.
- we use sub-pixel convolution instead of transposed convolution to upsample audio, as sub-pixel convolution is less prone to checkerboard artifacts than transpose convolution.
- we experimented with discrete cosine transformation (DCT) in addition to discrete Fourier transformation (DFT). DCT’s outputs are purely real and may be better suited for conventional deep learning architectures.

## Methodology
As with SpecGAN, we consider the problem of generating audio clips consisting of 16384 samples at 16 kHz. FOr the purpose we transform the audio signal in an invertible manner. This involves using the short-time DFT or DCT to create a time-frequency representation with phase information. A fixed transformation is applied to all input audio data, and its inverse is applied to the generator output to produce audio signals. GAN training then proceeds entirely in the transformed time-frequency domain, using the Wasserstein loss with gradient penalty. 

#### 1. DATA SCALING AND NORMALIZATION
The magnitudes of the DFT and DCT outputs are scaled roughly logarithmically both to simulate human auditory perception and to better bound the range of values. To do so without affecting phase information, we use an invertible function that maps 0 to itself:

along with its inverse:

Further, the magnitudes of the data are rescaled on a per-frequency basis: individually for each frequency, all values are divided by the maximal magnitude across the entire training dataset. Thus, all values are scaled into the range [−1, 1], matching the range of the tanh activation function used for the generator. These values are then multiplied by 0.9 to further limit this range, thereby allowing the generator to predict all possible values present in the dataset without saturating its output activations.

#### 2. NETWORK ARCHITECTURE
We introduce four model configurations, which we denote as DFT GAN, DCT GAN, DFT-SP GAN, and DCT-SP GAN. The generator architectures for each of these are shown in Table 1, and the discriminator architectures in Table 2.


import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Conv1D, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from .layers import ILConv1D, ILConv2D, RandomConvex
from functools import partial

# Generator for ST DFT images using ILConv2D to upsample
def dft_generator(dim, num_latent, bias_out):
    x = Input(shape=(num_latent,), dtype='float32')
    y = Dense(256*dim, activation='relu', kernel_initializer='he_uniform')(x)
    y = Reshape((4,4,16*dim))(y)
    y = ILConv2D(8*dim, (3,3), 2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv2D(4*dim, (3,3), 2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv2D(2*dim, (3,3), 2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv2D(dim, (3,3), 2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv2D(2, (3,3), 2, padding='same', activation='tanh', kernel_initializer='glorot_uniform', use_bias=bias_out)(y)
    return Model(inputs=x, outputs=y)

# Generator for ST DFT images using Conv2DTranspose to upsample
def dft_generator_tr(dim, num_latent, bias_out):
    x = Input(shape=(num_latent,), dtype='float32')
    y = Dense(256*dim, activation='relu', kernel_initializer='he_uniform')(x)
    y = Reshape((4,4,16*dim))(y)
    y = Conv2DTranspose(8*dim, (5,5), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = Conv2DTranspose(4*dim, (5,5), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = Conv2DTranspose(2*dim, (5,5), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = Conv2DTranspose(dim, (5,5), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = Conv2DTranspose(2, (5,5), strides=(2,2), padding='same', activation='tanh', kernel_initializer='glorot_uniform', use_bias=bias_out)(y)
    return Model(inputs=x, outputs=y)

# Discriminator for ST DFT images
def dft_discriminator(dim):
    x = Input(shape=(128,128,2), dtype='float32')
    y = Conv2D(dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(x)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(2*dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(4*dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(8*dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(16*dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Reshape((256*dim,))(y)
    y = Dense(1, activation=None, kernel_initializer='glorot_uniform')(y)
    return Model(inputs=x, outputs=y)

# Generator for ST DCT images using ILConv2D to upsample
def dct_generator(dim, num_latent, bias_out):
    x = Input(shape=(num_latent,), dtype='float32')
    y = Dense(256*dim, activation='relu', kernel_initializer='he_uniform')(x)
    y = Reshape((4,4,16*dim))(y)
    y = ILConv2D(8*dim, (3,3), 2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv2D(4*dim, (3,3), 2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv2D(2*dim, (3,3), 2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv2D(dim, (3,3), 2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv2D(dim, (3,3), 2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv2D(1, (3,3), 2, padding='same', activation='tanh', kernel_initializer='glorot_uniform', use_bias=bias_out)(y)
    return Model(inputs=x, outputs=y)

# Generator for ST DCT images using Conv2DTranspose to upsample
def dct_generator_tr(dim, num_latent, bias_out):
    x = Input(shape=(num_latent,), dtype='float32')
    y = Dense(256*dim, activation='relu', kernel_initializer='he_uniform')(x)
    y = Reshape((4,4,16*dim))(y)
    y = Conv2DTranspose(8*dim, (5,5), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = Conv2DTranspose(4*dim, (5,5), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = Conv2DTranspose(2*dim, (5,5), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = Conv2DTranspose(dim, (5,5), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = Conv2DTranspose(dim, (5,5), strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', activation='tanh', kernel_initializer='glorot_uniform', use_bias=bias_out)(y)
    return Model(inputs=x, outputs=y)

# Discriminator for ST DCT images
def dct_discriminator(dim):
    x = Input(shape=(256,256,1), dtype='float32')
    y = Conv2D(dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(x)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(2*dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(4*dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(8*dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(16*dim, (5,5), strides=(2,2), padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Reshape((256*dim,))(y)
    y = Dense(1, activation=None, kernel_initializer='glorot_uniform')(y)
    return Model(inputs=x, outputs=y)

# Generator for raw waveforms using ILConv1D to upsample
def wave_generator(dim, num_latent, bias_out):
    x = Input(shape=(num_latent,), dtype='float32')
    y = Dense(256*dim, activation='relu', kernel_initializer='he_uniform')(x)
    y = Reshape((16,16*dim))(y)
    y = ILConv1D(8*dim, 7, 4, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv1D(4*dim, 7, 4, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv1D(2*dim, 7, 4, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv1D(dim, 7, 4, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
    y = ILConv1D(1, 7, 4, padding='same', activation='tanh', kernel_initializer='glorot_uniform', use_bias=bias_out)(y)
    return Model(inputs=x, outputs=y)

# Discriminator for raw waveforms
def wave_discriminator(dim):
    x = Input(shape=(16384,1), dtype='float32')
    y = Conv1D(dim, 25, strides=4, padding='same', activation=None, kernel_initializer='he_uniform')(x)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv1D(2*dim, 25, strides=4, padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv1D(4*dim, 25, strides=4, padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv1D(8*dim, 25, strides=4, padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv1D(16*dim, 25, strides=4, padding='same', activation=None, kernel_initializer='he_uniform')(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Reshape((256*dim,))(y)
    y = Dense(1, activation=None, kernel_initializer='glorot_uniform')(y)
    return Model(inputs=x, outputs=y)

"""
Calculates the Wasserstein loss for a sample batch.
The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
the discriminator wants to make the distance between its output for real and generated samples as large as possible.
The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
Note that the nature of this loss means that it can be (and frequently will be) less than 0.
"""
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

"""
Calculates the gradient penalty loss for a batch of "averaged" samples.
In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
this function at all points in the input space. The compromise used in the paper is to choose random points
on the lines between real and generated samples, and check the gradients at these points. Note that it is the
gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
Then we get the gradients of the discriminator w.r.t. the input averaged samples.
The l2 norm and penalty can then be calculated for this gradient.
Note that this loss function requires the original averaged samples as input, but Keras only supports passing
y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
averaged_samples argument, and use that for model training.
"""
def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

# Creates and compiles generator and discriminator models ready for training using WGAN-GP as per https://arxiv.org/abs/1704.00028
# Inputs:
#   generator - a keras model for the generator that should not be already built
#   discriminator - a keras model fro the discriminator that should not be already built
#   gen_optimizer - a keras optimizer to be used in training the generator model,
#       defaults to keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
#   disc_optimizer - a keras optimizer to be used in training the discriminator model,
#       defaults to keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
#   grad_penalty - weight for the gradient penalty term, defaults to 10
# Returns:
#   a tuple of the compiled generator model and discriminator model, in that order
def create_wgan(generator, discriminator, gen_optimizer=None, disc_optimizer=None, grad_penalty=10):
    # The generator_model is used when we want to train the generator layers.
    # As such, we ensure that the discriminator layers are not trainable.
    # Note that once we compile this model, updating .trainable will have no effect within it. As such, it
    # won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
    # as we compile the generator_model first.
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generator_input = Input(shape=generator.input_shape[1:], dtype='float32')
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
    if gen_optimizer is None:
        gen_optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
    generator_model.compile(optimizer=gen_optimizer, loss=wasserstein_loss)
    # Now that the generator_model is compiled, we can make the discriminator layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False
    # The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
    # The noise seed is run through the generator model to get generated images. Both real and generated images
    # are then run through the discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples = Input(shape=discriminator.input_shape[1:], dtype='float32')
    generator_input_for_discriminator = Input(shape=generator.input_shape[1:], dtype='float32')
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)
    # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
    averaged_samples = RandomConvex()([real_samples, generated_samples_for_discriminator])
    # We then run these samples through the discriminator as well. Note that we never really use the discriminator
    # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)
    # The gradient penalty loss function requires the input averaged samples to get gradients. However,
    # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
    # of the function with the averaged samples here.
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=grad_penalty)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error
    # Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
    # real samples and generated samples before passing them to the discriminator: If we had, it would create an
    # output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
    # would have only BATCH_SIZE samples.
    # If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
    # samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
    discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    # Wasserstein loss for both the real and generated samples, and the gradient penalty loss for the averaged samples.
    if disc_optimizer is None:
        disc_optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
    discriminator_model.compile(optimizer=disc_optimizer,
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    return generator_model, discriminator_model

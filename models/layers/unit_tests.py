if not (__name__ == '__main__'):
    assert False, "tried to import un-importable module: models.layers.unit_tests"

from . import *
import numpy as np
import tensorflow as tf
import keras
from scipy.interpolate import interpn
import functools

# test initialization
print('testing initialization...')

print('\t ILConv1D: ', end="")
x = keras.layers.Input(shape=(5,1))
y = ILConv1D(filters=4, kernel_size=3, upscale_factor=2, padding='same', bias_initializer='random_uniform',
             activation='sigmoid')(x)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
assert (m.layers[1].get_weights()[0][...,0:4] == m.layers[1].get_weights()[0][...,4:8]).all(), "ILConv1D: weight initialization incorrect"
assert (m.layers[1].get_weights()[1][...,0:4] == m.layers[1].get_weights()[1][...,4:8]).all(), "ILConv1D: bias initialization incorrect"
m_out = m.predict(np.random.uniform(low=-1,high=1,size=(32,5,1)).astype(np.float32), batch_size=32)
assert (m_out.shape == (32,10,4)), "ILConv1D: output shape incorrect"
assert (m_out[:,0::2,:] == m_out[:,1::2,:]).all(), "ILConv1D: initial evaluation incorrect"

x = keras.layers.Input(shape=(4096,32))
y = ILConv1D(filters=8, kernel_size=25, upscale_factor=2, padding='same', activation='relu')(x)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
assert (m.layers[1].get_weights()[0][...,0:8] == m.layers[1].get_weights()[0][...,8:16]).all(), "ILConv1D: weight initialization incorrect"
assert (m.layers[1].get_weights()[1][...,0:8] == m.layers[1].get_weights()[1][...,8:16]).all(), "ILConv1D: bias initialization incorrect"
m_out = m.predict(np.random.uniform(low=-1,high=1,size=(64,4096,32)).astype(np.float32), batch_size=64)
assert (m_out.shape == (64,8192,8)), "ILConv1D: output shape incorrect"
assert (m_out[:,0::2,:] == m_out[:,1::2,:]).all(), "ILConv1D: initial evaluation incorrect"

x = keras.layers.Input(shape=(50,1))
y = ILConv1D(filters=1, kernel_size=5, upscale_factor=4, strides=2, padding='same', activation=None)(x)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
m_out = m.predict(np.random.uniform(low=-1,high=1,size=(32,50,1)).astype(np.float32), batch_size=32)
assert (m_out.shape == (32,100,1)), "ILConv1D: output shape incorrect"
assert (    (m_out[:,0::4,:] == m_out[:,1::4,:]).all()
        and (m_out[:,0::4,:] == m_out[:,2::4,:]).all()
        and (m_out[:,0::4,:] == m_out[:,3::4,:]).all()), "ILConv1D: initial evaluation incorrect"

x = keras.layers.Input(shape=(511,127))
y = ILConv1D(filters=31, kernel_size=24, upscale_factor=7, strides=2, padding='valid', activation='relu',
             kernel_initializer='he_uniform', use_bias=False)(x)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('sgd', loss='mean_squared_error')
m_out = m.predict(np.random.uniform(low=-1,high=1,size=(101,511,127)).astype(np.float32), batch_size=101)
assert (m_out.shape == (101,1708,31)), "ILConv1D: output shape incorrect"
assert (    (m_out[:,0::7,:] == m_out[:,1::7,:]).all()
        and (m_out[:,0::7,:] == m_out[:,2::7,:]).all()
        and (m_out[:,0::7,:] == m_out[:,3::7,:]).all()
        and (m_out[:,0::7,:] == m_out[:,4::7,:]).all()
        and (m_out[:,0::7,:] == m_out[:,5::7,:]).all()
        and (m_out[:,0::7,:] == m_out[:,6::7,:]).all()), "ILConv1D: initial evaluation incorrect"
print('pass')

print('\t ILConv2D: ', end="")
x = keras.layers.Input(shape=(5,5,1))
y = ILConv2D(filters=4, kernel_size=(3,3), upscale_factor=2, padding='same', bias_initializer='random_uniform',
             activation='sigmoid')(x)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
assert (    (m.layers[1].get_weights()[0][...,0:4] == m.layers[1].get_weights()[0][...,4:8]).all()
        and (m.layers[1].get_weights()[0][...,0:4] == m.layers[1].get_weights()[0][...,8:12]).all()
        and (m.layers[1].get_weights()[0][...,0:4] == m.layers[1].get_weights()[0][...,12:16]).all()), "ILConv2D: weight initialization incorrect"
assert (    (m.layers[1].get_weights()[1][...,0:4] == m.layers[1].get_weights()[1][...,4:8]).all()
        and (m.layers[1].get_weights()[1][...,0:4] == m.layers[1].get_weights()[1][...,8:12]).all()
        and (m.layers[1].get_weights()[1][...,0:4] == m.layers[1].get_weights()[1][...,12:16]).all()), "ILConv2D: bias initialization incorrect"
m_out = m.predict(np.random.uniform(low=-1,high=1,size=(32,5,5,1)).astype(np.float32), batch_size=32)
assert (m_out.shape == (32,10,10,4)), "ILConv2D: output shape incorrect"
assert (    (m_out[:,0::2,0::2,:] == m_out[:,0::2,1::2,:]).all()
        and (m_out[:,0::2,0::2,:] == m_out[:,1::2,0::2,:]).all()
        and (m_out[:,0::2,0::2,:] == m_out[:,1::2,1::2,:]).all()), "ILConv2D: initial evaluation incorrect"

x = keras.layers.Input(shape=(64,64,16))
y = ILConv2D(filters=2, kernel_size=(5,5), upscale_factor=2, padding='same', activation='relu')(x)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
assert (    (m.layers[1].get_weights()[0][...,0:2] == m.layers[1].get_weights()[0][...,2:4]).all()
        and (m.layers[1].get_weights()[0][...,0:2] == m.layers[1].get_weights()[0][...,4:6]).all()
        and (m.layers[1].get_weights()[0][...,0:2] == m.layers[1].get_weights()[0][...,6:8]).all()), "ILConv2D: weight initialization incorrect"
assert (    (m.layers[1].get_weights()[1][...,0:2] == m.layers[1].get_weights()[1][...,2:4]).all()
        and (m.layers[1].get_weights()[1][...,0:2] == m.layers[1].get_weights()[1][...,4:6]).all()
        and (m.layers[1].get_weights()[1][...,0:2] == m.layers[1].get_weights()[1][...,6:8]).all()), "ILConv2D: bias initialization incorrect"
m_out = m.predict(np.random.uniform(low=-1,high=1,size=(64,64,64,16)).astype(np.float32), batch_size=64)
assert (m_out.shape == (64,128,128,2)), "ILConv2D: output shape incorrect"
assert (    (m_out[:,0::2,0::2,:] == m_out[:,0::2,1::2,:]).all()
        and (m_out[:,0::2,0::2,:] == m_out[:,1::2,0::2,:]).all()
        and (m_out[:,0::2,0::2,:] == m_out[:,1::2,1::2,:]).all()), "ILConv2D: initial evaluation incorrect"

x = keras.layers.Input(shape=(50,50,1))
y = ILConv2D(filters=1, kernel_size=(5,5), upscale_factor=4, strides=2, padding='same', activation=None)(x)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
m_out = m.predict(np.random.uniform(low=-1,high=1,size=(32,50,50,1)).astype(np.float32), batch_size=32)
assert (m_out.shape == (32,100,100,1)), "ILConv2D: output shape incorrect"
assert (    (m_out[:,0::4,0::4,:] == m_out[:,0::4,1::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,0::4,2::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,0::4,3::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,1::4,0::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,1::4,1::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,1::4,2::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,1::4,3::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,2::4,0::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,2::4,1::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,2::4,2::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,2::4,3::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,3::4,0::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,3::4,1::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,3::4,2::4,:]).all()
        and (m_out[:,0::4,0::4,:] == m_out[:,3::4,3::4,:]).all()), "ILConv2D: initial evaluation incorrect"

x = keras.layers.Input(shape=(127,127,31))
y = ILConv2D(filters=18, kernel_size=(7,7), upscale_factor=5, strides=11, padding='valid', activation='relu',
             kernel_initializer='he_uniform', use_bias=False)(x)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
m_out = m.predict(np.random.uniform(low=-1,high=1,size=(63,127,127,31)).astype(np.float32), batch_size=63)
assert (m_out.shape == (63,55,55,18)), "ILConv2D: output shape incorrect"
assert (    (m_out[:,0::5,0::5,:] == m_out[:,0::5,1::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,0::5,2::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,0::5,3::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,0::5,4::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,1::5,0::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,1::5,1::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,1::5,2::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,1::5,3::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,1::5,4::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,2::5,0::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,2::5,1::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,2::5,2::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,2::5,3::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,2::5,4::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,3::5,0::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,3::5,1::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,3::5,2::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,3::5,3::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,3::5,4::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,4::5,0::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,4::5,1::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,4::5,2::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,4::5,3::5,:]).all()
        and (m_out[:,0::5,0::5,:] == m_out[:,4::5,4::5,:]).all()), "ILConv2D: initial evaluation incorrect"
print('pass')

# test training
print('testing training (note that due to randomness, there is a miniscule chance of test failure; try re-running if anything fails)...')

# generator to create a toy linear interpolation problem
class LinearSequence(keras.utils.Sequence):
    def __init__(self, num_batch, batch_size, in_shape):
        self.in_shape = tuple([batch_size] + list(in_shape))
        self.out_shape = tuple([batch_size] + list(map(lambda x: 2*x-1, in_shape[:-1])) + [in_shape[-1]])
        self.num_batch = num_batch
        self.random = np.random.RandomState(seed=177013)
    def __len__(self):
        return self.num_batch
    def __getitem__(self, key):
        data = self.random.uniform(low=0,high=1,size=self.in_shape).astype(np.float32)
        targets = np.empty(self.out_shape, dtype=np.float32)
        if (len(self.in_shape) == 3):
            t_in = 2 * np.arange(self.in_shape[1]).astype(np.float32)
            t_out = np.arange(self.out_shape[1]).astype(np.float32)
            interp_func = functools.partial(interpn, points=(t_in,), xi=t_out)
        if (len(self.in_shape) == 4):
            x_in = 2 * np.arange(self.in_shape[1]).astype(np.float32)
            y_in = 2 * np.arange(self.in_shape[2]).astype(np.float32)
            x_out = np.arange(self.out_shape[1]).astype(np.float32)
            y_out = np.arange(self.out_shape[2]).astype(np.float32)
            x_out, y_out = np.meshgrid(x_out, y_out)
            xy_out = np.stack([y_out.flatten(),x_out.flatten()], axis=-1)
            interp_func = functools.partial(interpn, points=(x_in,y_in), xi=xy_out)
        for batch_i in range(self.out_shape[0]):
            targets[batch_i,...] = np.reshape(interp_func(values=data[batch_i,...]), self.out_shape[1:])
        return data, targets
    def on_epoch_end(self):
        pass

print('\t ILConv1D: ', end="")
x = keras.layers.Input(shape=(5,1))
y = ILConv1D(filters=1, kernel_size=3, upscale_factor=2, padding='same', use_bias=False, activation=None)(x)
y = keras.layers.Lambda(lambda x: x[:,:-1,:])(y)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
gen = LinearSequence(128,32,(5,1))
h = m.fit_generator(gen, epochs=50, verbose=0, shuffle=False)
assert (h.history['loss'][-1] / h.history['loss'][0] <= 0.001), "ILConv1D: not learning (single layer)"

x = keras.layers.Input(shape=(5,3))
y = ILConv1D(filters=3, kernel_size=3, upscale_factor=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
y = ILConv1D(filters=3, kernel_size=3, upscale_factor=2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
y = keras.layers.Conv1D(filters=3, kernel_size=3, strides=2, padding='valid')(y)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
gen = LinearSequence(128,32,(5,3))
h = m.fit_generator(gen, epochs=50, verbose=0, shuffle=False)
assert (h.history['loss'][-1] / h.history['loss'][0] <= 0.001), "ILConv1D: not learning (multiple layers)"

x = keras.layers.Input(shape=(5,3))
y = ILConv1D(filters=3, kernel_size=3, upscale_factor=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
y = ILConv1D(filters=3, kernel_size=3, upscale_factor=2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
y = keras.layers.Conv1D(filters=3, kernel_size=3, strides=2, padding='valid')(y)
m = keras.models.Model(inputs=x, outputs=y)
m.layers[2].trainable = False
m0 = keras.models.Model(inputs=x, outputs=m(x))
m0.compile('adam', loss='mean_squared_error')
m.layers[1].trainable = False
m.layers[2].trainable = True
m1 = keras.models.Model(inputs=x, outputs=m(x))
m1.compile('adam', loss='mean_squared_error')
weights_1_0 = m.layers[1].get_weights()
weights_2_0 = m.layers[2].get_weights()
m0.fit_generator(gen, epochs=2, verbose=0, shuffle=False)
assert (    (m.layers[1].get_weights()[0] != weights_1_0[0]).any()
        and (m.layers[1].get_weights()[1] != weights_1_0[1]).any()), "ILConv1D: weight un-freeze failure"
assert (    (m.layers[2].get_weights()[0] == weights_2_0[0]).all()
        and (m.layers[2].get_weights()[1] == weights_2_0[1]).all()), "ILConv1D: weight freeze failure"
weights_1_0 = m.layers[1].get_weights()
weights_2_0 = m.layers[2].get_weights()
m1.fit_generator(gen, epochs=2, verbose=0, shuffle=False)
assert (    (m.layers[1].get_weights()[0] == weights_1_0[0]).all()
        and (m.layers[1].get_weights()[1] == weights_1_0[1]).all()), "ILConv1D: weight freeze failure"
assert (    (m.layers[2].get_weights()[0] != weights_2_0[0]).any()
        and (m.layers[2].get_weights()[1] != weights_2_0[1]).any()), "ILConv1D: weight un-freeze failure"
print('pass')

print('\t ILConv2D: ', end="")
x = keras.layers.Input(shape=(5,5,1))
y = ILConv2D(filters=1, kernel_size=(3,3), upscale_factor=2, padding='same', use_bias=False, activation=None)(x)
y = keras.layers.Lambda(lambda x: x[:,:-1,:-1,:])(y)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
gen = LinearSequence(128,32,(5,5,1))
h = m.fit_generator(gen, epochs=50, verbose=0, shuffle=False)
assert (h.history['loss'][-1] / h.history['loss'][0] <= 0.001), "ILConv2D: not learning (single layer)"

x = keras.layers.Input(shape=(5,5,3))
y = ILConv2D(filters=3, kernel_size=(3,3), upscale_factor=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
y = ILConv2D(filters=3, kernel_size=(3,3), upscale_factor=2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
y = keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=2, padding='valid')(y)
m = keras.models.Model(inputs=x, outputs=y)
m.compile('adam', loss='mean_squared_error')
gen = LinearSequence(128,32,(5,5,3))
h = m.fit_generator(gen, epochs=50, verbose=0, shuffle=False)
assert (h.history['loss'][-1] / h.history['loss'][0] <= 0.001), "ILConv2D: not learning (multiple layers)"

x = keras.layers.Input(shape=(5,5,3))
y = ILConv2D(filters=3, kernel_size=(3,3), upscale_factor=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
y = ILConv2D(filters=3, kernel_size=(3,3), upscale_factor=2, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
y = keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=2, padding='valid')(y)
m = keras.models.Model(inputs=x, outputs=y)
m.layers[2].trainable = False
m0 = keras.models.Model(inputs=x, outputs=m(x))
m0.compile('adam', loss='mean_squared_error')
m.layers[1].trainable = False
m.layers[2].trainable = True
m1 = keras.models.Model(inputs=x, outputs=m(x))
m1.compile('adam', loss='mean_squared_error')
weights_1_0 = m.layers[1].get_weights()
weights_2_0 = m.layers[2].get_weights()
m0.fit_generator(gen, epochs=2, verbose=0, shuffle=False)
assert (    (m.layers[1].get_weights()[0] != weights_1_0[0]).any()
        and (m.layers[1].get_weights()[1] != weights_1_0[1]).any()), "ILConv2D: weight un-freeze failure"
assert (    (m.layers[2].get_weights()[0] == weights_2_0[0]).all()
        and (m.layers[2].get_weights()[1] == weights_2_0[1]).all()), "ILConv2D: weight freeze failure"
weights_1_0 = m.layers[1].get_weights()
weights_2_0 = m.layers[2].get_weights()
m1.fit_generator(gen, epochs=2, verbose=0, shuffle=False)
assert (    (m.layers[1].get_weights()[0] == weights_1_0[0]).all()
        and (m.layers[1].get_weights()[1] == weights_1_0[1]).all()), "ILConv2D: weight freeze failure"
assert (    (m.layers[2].get_weights()[0] != weights_2_0[0]).any()
        and (m.layers[2].get_weights()[1] != weights_2_0[1]).any()), "ILConv2D: weight un-freeze failure"
print('pass')

print('done.')

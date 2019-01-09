import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils import conv_utils

# Base class implementing ICNR "sub-pixel" (i.e. interleaved) convolutions
# a modified version of the vanilla convolution base class from keras
# NOTE: currently only works with tensorflow backend
class _ILConv(Layer):
    """
    Abstract nD upscaling sub-sample interleave convolution layer
    (private, used as implementation base).

    This layer creates a list of convolution kernels that are convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    # Arguments
        rank: An integer, the rank of the convolution,
            e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        upscale_factor: An integer, representing the upscaling factor along
            all convolved dimensions.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """
    def __init__(self, rank,
            filters,
            kernel_size,
            upscale_factor,
            strides=1,
            padding='valid',
            data_format=None,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs):
        super(_ILConv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.upscale_factor = upscale_factor
        self.num_subsamples = self.upscale_factor ** self.rank
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters
                                                      * self.num_subsamples)
        subkernel_shape = self.kernel_size + (input_dim, self.filters)
        k_init = self.__icnr(self.kernel_initializer, subkernel_shape,
                             self.num_subsamples)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=k_init,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            bias_shape = (self.filters * self.num_subsamples,)
            subbias_shape = (self.filters,)
            b_init = self.__icnr(self.bias_initializer, subbias_shape,
                                 self.num_subsamples)
            self.bias = self.add_weight(shape=bias_shape,
                                        initializer=b_init,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # perform convolutions with each set of kernels
        if self.rank == 1:
            outputs = K.conv1d(inputs,
                               self.kernel,
                               strides=self.strides[0],
                               padding=self.padding,
                               data_format=self.data_format,
                               dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(inputs,
                               self.kernel,
                               strides=self.strides,
                               padding=self.padding,
                               data_format=self.data_format,
                               dilation_rate=self.dilation_rate)
        if self.rank == 3:
            # we don't need this for our models
            raise NotImplementedError

        if self.use_bias:
            outputs = K.bias_add(outputs,
                                 self.bias,
                                 data_format=self.data_format)

        # interleave the kernels
        if self.rank == 1:
            if self.data_format == 'channels_first':
                # rather unelegant and probably inefficient, but using
                # 'channels_first' w/ tf backend is strange anyways
                outputs = K.permute_dimensions(outputs, (0,2,1))
            out_shape = K.concatenate([K.shape(inputs)[0:1], (-1, self.filters)])
            outputs = K.reshape(outputs, out_shape)
            if self.data_format == 'channels_first':
                # rather unelegant and probably inefficient, but using
                # 'channels_first' w/ tf backend is strange anyways
                outputs = K.permute_dimensions(outputs, (0,2,1))
        if self.rank == 2:
            if self.data_format == 'channels_first':
                tf_data_format = 'NCHW'
            else:
                tf_data_format = 'NHWC'
            outputs = tf.depth_to_space(outputs, self.upscale_factor,
                        data_format=tf_data_format)
        if self.rank == 3:
            # we don't need this for our models
            raise NotImplementedError

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(max(0, self.upscale_factor * new_dim))
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(max(0, self.upscale_factor * new_dim))
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'upscale_factor': self.upscale_factor,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_ILConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # initializer wrapper for ICNR initialization
    @staticmethod
    def __icnr(base_initializer, sub_shape, num_subsamples):
        def icnr_init(shape, dtype=None):
            values = base_initializer(sub_shape, dtype)
            if (len(sub_shape) < 2):
                values = K.expand_dims(values, axis=0)
            return K.reshape(K.repeat_elements(values, num_subsamples,
                                               axis=-2), shape)
        return icnr_init

class ILConv1D(_ILConv):
    """
    1D interleave-upscaling convolution layer.

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs which are then interleaved to expand the
    spatial (or temporal) dimension. If `use_bias` is True, a bias vector is
    created and added to the outputs. Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide an `input_shape` argument (tuple of integers or `None`, does not
    include the batch axis), e.g. `input_shape=(10, 128)` for time series
    sequences of 10 time steps with 128 features per step in
    `data_format="channels_last"`, or `(None, 128)` for variable-length
    sequences with 128 features per step.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        upscale_factor: An integer representing the upscaling factor along
            the temporal or spatial dimension.
        strides: An integer or tuple/list of a single integer,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
            `"valid"` means "no padding".
            `"same"` results in padding the input such that
            the output has the same length as the original input.
            `"causal"` results in causal (dilated) convolutions,
            e.g. `output[t]` does not depend on `input[t + 1:]`.
            A zero padding is used such that
            the output has the same length as the original input.
            Useful when modeling temporal data where the model
            should not violate the temporal order. See
            [WaveNet: A Generative Model for Raw Audio, section 2.1](
            https://arxiv.org/abs/1609.03499).
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, steps, channels)`
            (default format for temporal data in Keras)
            while `"channels_first"` corresponds to inputs
            with shape `(batch, channels, steps)`.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        3D tensor with shape: `(batch, steps, channels)`
    # Output shape
        3D tensor with shape: `(batch, new_steps * upscale_factor, filters)`
        where `new_steps` is the same value as presented for the output of
        [Conv1D](https://keras.io/layers/convolutional/#conv1d).
    """
    def __init__(self, filters,
            kernel_size,
            upscale_factor,
            strides=1,
            padding='valid',
            data_format=None,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs):
        if padding == 'causal':
            if data_format != 'channels_last':
                raise ValueError('When using causal padding in `ILConv1D`, '
                                 '`data_format` must be "channels_last" '
                                 '(temporal data).')
        super(ILConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            upscale_factor=upscale_factor,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
    def get_config(self):
        config = super(ILConv1D, self).get_config()
        config.pop('rank')
        return config

class ILConv2D(_ILConv):
    """
    2D interleave-upscaling convolution layer.

    This layer creates a convolution kernel that is convolved with the layer
    input over two spatial dimensions to produce a tensor of outputs which
    are then interleaved into square 2D blocks to expand the spatial dimension.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`, it is applied to the outputs as
    well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        upscale_factor: An integer representing the upscaling factor along
            the spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
            Note that `"same"` is slightly inconsistent across backends with
            `strides` != 1, as described
            [here](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(batch, rows, cols, channels)`
        if `data_format` is `"channels_last"`.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows * upscale_factor, new_cols * upscale_factor)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(batch, new_rows * upscale_factor, new_cols * upscale_factor, filters)`
        if `data_format` is `"channels_last"`.
        Here, `new_rows` and `new_cols` are the same values as presented for
        the output of [Conv2D](https://keras.io/layers/convolutional/#conv2d).
    """
    def __init__(self, filters,
            kernel_size,
            upscale_factor,
            strides=1,
            padding='valid',
            data_format=None,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs):
        super(ILConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            upscale_factor=upscale_factor,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
    def get_config(self):
        config = super(ILConv2D, self).get_config()
        config.pop('rank')
        return config

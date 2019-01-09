import keras.backend as K
from keras.layers.merge import _Merge

"""
Takes a random convex combination of two tensors.
Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
Improvements appreciated.
"""
class RandomConvex(_Merge):
    def _merge_function(self, inputs):
        input_shape = K.shape(inputs[0])
        weights = K.random_uniform(shape=tuple([input_shape[0]] + [1] * (input_shape.shape[0].value - 1)))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

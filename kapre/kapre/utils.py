# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from keras.engine import Layer
from keras import backend as K
from . import backend
from . import backend_keras
import tensorflow as tf


class AmplitudeToDB(Layer):
    '''

    ### `AmplitudeToDB`

    ```python
    kapre.utils.AmplitudeToDB(ref_power=1.0, amin=1e-10, top_db=80.0, **kargs)
    ```

    A layer that converts amplitude to decibel

    #### Parameters

    * ref_power: float [scalar]
        - reference power. Default: 1.0

    * amin: float [scalar]
        - Noise floor. Default: 1e-10

    * top_db: float [scalar]
        - Dynamic range of output. Default: 80.0

    #### Example
    Adding ``AmplitudeToDB`` after a spectrogram:
    ```python
        model.add(Spectrogram(return_decibel=False))
        model.add(AmplitudeToDB())
    ```
    , which is the same as:
    ```python
        model.add(Spectrogram(return_decibel=True))
    ```

    '''

    def __init__(self, ref_power=1.0, amin=1e-10, top_db=80.0, **kwargs):
        assert isinstance(ref_power, float) or ref_power == 'max'
        self.ref_power = ref_power
        self.amin = amin
        self.top_db = top_db
        super(AmplitudeToDB, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return backend_keras.amplitude_to_decibel(x, self.ref_power, self.amin, self.top_db)

    def get_config(self):
        config = {'ref_power': self.ref_power,
                  'amin': self.amin,
                  'top_db': self.top_db}
        base_config = super(AmplitudeToDB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Normalization2D(Layer):
    '''

    ### `Normalization2D`

    `kapre.utils.Normalization2D`

    A layer that normalises input data in ``axis`` axis.

    #### Parameters

    * input_shape: tuple of ints
        - E.g., ``(None, n_ch, n_row, n_col)`` if theano.

    * str_axis: str
        - used ONLY IF ``int_axis`` is ``None``.
        - ``'batch'``, ``'data_sample'``, ``'channel'``, ``'freq'``, ``'time')``
        - Even though it is optional, actually it is recommended to use
        - ``str_axis`` over ``int_axis`` because it provides more meaningful
        - and image data format-robust interface.

    * int_axis: int
        - axis index that along which mean/std is computed.
        - `0` for per data sample, `-1` for per batch.
        - `1`, `2`, `3` for channel, row, col (if channels_first)
        - if `int_axis is None`, ``str_axis`` SHOULD BE set.

    #### Example

    A frequency-axis normalization after a spectrogram::
        ```python
        model.add(Spectrogram())
        model.add(Normalization2D(stf_axis='freq'))
        ```
    '''

    def __init__(self, str_axis=None, int_axis=None, image_data_format='default',
                 eps=1e-10, **kwargs):
        assert not (int_axis is None and str_axis is None), \
            'In Normalization2D, int_axis or str_axis should be specified.'

        assert image_data_format in ('channels_first', 'channels_last', 'default'), \
            'Incorrect image_data_format: {}'.format(image_data_format)

        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format

        self.str_axis = str_axis
        if self.str_axis is None:  # use int_axis
            self.int_axis = int_axis
        else:  # use str_axis
            # warning
            if int_axis is not None:
                print(
                    'int_axis={} passed but is ignored, str_axis is used instead.'.format(int_axis))
            # do the work
            assert str_axis in ('batch', 'data_sample', 'channel', 'freq', 'time'), \
                'Incorrect str_axis: {}'.format(str_axis)
            if str_axis == 'batch':
                int_axis = -1
            else:
                if self.image_data_format == 'channels_first':
                    int_axis = ['data_sample', 'channel',
                                'freq', 'time'].index(str_axis)
                else:
                    int_axis = ['data_sample', 'freq',
                                'time', 'channel'].index(str_axis)

        assert int_axis in (-1, 0, 1, 2,
                            3), 'invalid int_axis: ' + str(int_axis)
        self.axis = int_axis
        self.eps = eps
        super(Normalization2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.axis == -1:
            mean = K.mean(x, axis=[3, 2, 1, 0], keepdims=True)
            std = K.std(x, axis=[3, 2, 1, 0], keepdims=True)
        elif self.axis in (0, 1, 2, 3):
            all_dims = [0, 1, 2, 3]
            del all_dims[self.axis]
            mean = K.mean(x, axis=all_dims, keepdims=True)
            std = K.std(x, axis=all_dims, keepdims=True)
        return (x - mean) / (std + self.eps)

    def get_config(self):
        config = {'int_axis': self.axis,
                  'str_axis': self.str_axis,
                  'image_data_format': self.image_data_format}
        base_config = super(Normalization2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeltaDelta(Layer):
    '''
    Layer that appends deltas as an extra channel
    '''

    def __init__(self, n=2, **kwargs):
        self.n = n
        super(DeltaDelta, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        # if self.image_data_format == 'channels_first':
        #     return input_shape[0], input_shape[1] + 2, input_shape[2], input_shape[3]
        # else:
        return input_shape[0], input_shape[1], input_shape[2], input_shape[3] + 2

    def build(self, input_shape):

        delta_kernel = np.arange(-self.n, self.n + 1
                                 ).reshape((1, 2 * self.n + 1, 1, 1))
        delta_kernel = delta_kernel/(2*sum(np.arange(self.n+1)**2))

        self.delta_kernel = K.variable(delta_kernel, dtype=K.floatx())

        self.non_trainable_weights.append(self.delta_kernel)
        self.paddings = K.constant([[0,0], [0, 0], [self.n, self.n], [0,0]], dtype="int32")
        super(DeltaDelta, self).build(input_shape)
        # self.built = True

    def call(self, x, mask=None):

        x_pad = tf.pad(x, self.paddings)
        delta = K.conv2d(x_pad, self.delta_kernel, data_format="channels_last")
        delta_pad = tf.pad(delta, self.paddings)
        delta_delta = K.conv2d(delta_pad, self.delta_kernel,
                               data_format="channels_last")

        return K.concatenate((x, delta, delta_delta), axis=-1)

    def get_config(self):
        config = {'n': self.n}
        base_config = super(DeltaDelta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

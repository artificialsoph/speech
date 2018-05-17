# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from keras import backend as K


# def amplitude_to_decibel(x, amin=1e-10, dynamic_range=80.0):
#     """[K] Convert (linear) amplitude to decibel (log10(x)).

#     x: Keras tensor or variable.

#     amin: minimum amplitude. amplitude smaller than `amin` is set to this.

#     dynamic_range: dynamic_range in decibel
#     """
#     log_spec = 10 * K.log(K.maximum(x, amin)) / np.log(10).astype(K.floatx())
#     log_spec = log_spec - K.max(log_spec)  # [-?, 0]
#     log_spec = K.maximum(log_spec, -1 * dynamic_range)  # [-80, 0]
#     return log_spec

def log10(x):
    """ [K] change of log base for convenience
    """
    return K.log(x) / np.log(10).astype(K.floatx())


def amplitude_to_decibel(x, amin=1e-10, ref=1.0, dynamic_range=80.0):
    """[K] Convert (linear) amplitude to decibel (log10(x)).

    x: Keras tensor or variable.

    amin: minimum amplitude. amplitude smaller than `amin` is set to this.

    dynamic_range: dynamic_range in decibel
    """
    assert amin > 0
    assert dynamic_range >= 0

    magnitude = K.abs(x)
    ref_value = np.abs(ref)

    log_spec = 10.0 * log10(K.maximum(amin, magnitude))
    log_spec -= 10.0 * log10(np.maximum(amin, ref_value).astype(K.floatx()))
    log_spec = K.maximum(log_spec, K.max(log_spec) - dynamic_range)
    return log_spec

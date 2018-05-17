import random
import pandas as pd
import scipy.io.wavfile as wav
import scipy.signal
from scipy.ndimage.interpolation import shift
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
import samplerate
import tensorflow as tf

ex_df = pd.read_pickle("data/ex_df.pkl")

def soph_scaler(wave):
    return wave
#     mean = np.mean(wave)
#     std = np.std(wave)
#     return (wave - mean) / std


bg_waves = [soph_scaler(wav.read(fn)[1])
            for fn in ex_df[ex_df.state == "bg"].fn]
maxlen = 16000


def center_wave(wav_fn, vol_range=0, displacement=0, shift=0, p_transform=0):

    if wav_fn == "silence":
        bg_wave = random.choice(bg_waves)
        bg_start = np.random.randint(len(bg_wave) - maxlen)
        wave = bg_wave[bg_start:bg_start + maxlen]

    else:

        wave = soph_scaler(wav.read(wav_fn)[1])

        if np.random.rand() < shift:
            wave = samplerate.resample(
                wave, np.random.randint(14400, 17600) / 16000, "sinc_fastest")
# wave = librosa.core.resample(wave,
#                              orig_sr=16000,
#                              target_sr=np.random.randint(14400,17600),
#                              res_type='kaiser_fast')

        if len(wave) > 16000:
            wave_start = np.random.randint(len(wave) - maxlen)
            wave = wave[wave_start:wave_start + maxlen]
        elif len(wave) < 16000:
            left_pad = (maxlen - wave.shape[0]) // 2
            right_pad = maxlen - wave.shape[0] - left_pad

            wave = np.pad(
                wave, (left_pad, right_pad), 'constant', constant_values=0)

        if vol_range > 0:
            if np.random.rand() < p_transform:

                bg_vol = vol_range * np.random.rand(1)

                wave = (1 - bg_vol) * wave + bg_vol * \
                    center_wave("silence", maxlen)
                wave = np.clip(wave, -1, 1)

    return wave


def ex_generator(batch_size=32,
                 shuffle=True,
                 raw_label=False,
                 state="train",
                 vol_range=0,
                 displacement=0,
                 shift=0,
                 p_transform=0,
                 num_seq=None):

    epoch_df = ex_df[ex_df.state.isin(state)]
    num_ex = len(epoch_df)
    indices = np.arange(num_ex)

    # epoch loop runs
    while True:
        
        y_col = "raw_label_i" if raw_label else "label_i"
        
        # shuffle anew every epoch
        if shuffle:
            epoch_df = epoch_df.sample(frac=1)

        # batch loop
        for i in np.arange(0, num_ex, batch_size):
            
            if (i + batch_size) > num_ex:
                continue

            batch_df = epoch_df.iloc[i:i + batch_size:, :]
            
            x = np.zeros((len(batch_df),16000))
            y = np.zeros(len(batch_df))

            # example loop
            for i in range(len(batch_df)):

                # get the processed file
                x[i,...] = center_wave(
                    epoch_df.fn.values[i], 
                    vol_range=vol_range, 
                    displacement=displacement, 
                    p_transform=p_transform, 
                    shift=shift)
                y[i] = epoch_df[y_col].values[i]
            
            cw = batch_df.label_weight.values
            yield x, y, cw

            
class MFCC(Layer):
    """
    """

    def __init__(self,
                 frame_length=1024,
                 frame_step=512,
                 fft_length=1024,
                 lower_edge_hertz=None,
                 upper_edge_hertz=None,
                 num_mel_bins=64,
                 sr=16000,
                 n_mfcc=13,
                 **kwargs):

        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.n_bins = fft_length // 2 + 1
        self.log_offset = 1e-6
        if lower_edge_hertz:
            self.lower_edge_hertz = lower_edge_hertz
        else:
            self.lower_edge_hertz = 0
        if upper_edge_hertz:
            self.upper_edge_hertz = upper_edge_hertz
        else:
            self.upper_edge_hertz = sr//2
        self.num_mel_bins = num_mel_bins
        self.sr = sr
        self.n_mfcc = n_mfcc
        super(MFCC, self).__init__(**kwargs)

    def build(self, input_shape):
        self.linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, self.n_bins, self.sr, self.lower_edge_hertz,
            self.upper_edge_hertz)
        
        self.non_trainable_weights.append(self.linear_to_mel_weight_matrix)
        super(MFCC, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        n_samples = input_shape[1]
        n_seq = n_samples//self.frame_step

        return batch, n_seq, self.n_mfcc

    def call(self, x):

        stfts = tf.contrib.signal.stft(
            x,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            pad_end=True,
        )
        magnitude_spectrograms = tf.abs(stfts)

        mel_spectrograms = K.dot(magnitude_spectrograms, self.linear_to_mel_weight_matrix)

        log_mel_spectrograms = tf.log(mel_spectrograms + self.log_offset)
        
        mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
                  log_mel_spectrograms
        )[..., :self.n_mfcc]
        
        return mfccs

    def get_config(self):
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.num_mel_bins = num_mel_bins

        config = {
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'fft_length': self.fft_length,
            'log_offset': self.log_offset,
            'n_bins': self.n_bins,
            'lower_edge_hertz': self.lower_edge_hertz,
            'upper_edge_hertz': self.upper_edge_hertz,
            'num_mel_bins': self.num_mel_bins,
            'sr': self.sr,
            'n_mfcc': n_mfcc
        }
        base_config = super(MFCC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Delta(Layer):
    '''
    Layer that appends deltas as an extra channel
    '''

    def __init__(self, n=2, order=2, **kwargs):
        assert order==1 or order==2
        self.n = n
        self.order = order
        super(Delta, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        time = input_shape[1]
        features = input_shape[2]

        return batch, time, features, self.order+1

    def build(self, input_shape):

        delta_kernel = np.arange(-self.n, self.n + 1
                                 ).reshape((1, 2 * self.n + 1, 1, 1))
        delta_kernel = delta_kernel/(2*sum(np.arange(self.n+1)**2))

        self.delta_kernel = K.variable(delta_kernel, dtype=K.floatx())

        self.non_trainable_weights.append(self.delta_kernel)
        self.paddings = K.constant([[0,0], [0, 0], [self.n, self.n], [0,0]], dtype="int32")
        super(Delta, self).build(input_shape)

    def call(self, x, mask=None):
        
        x_orig = tf.expand_dims(x, -1)
        deltas = [x_orig]
        
        to_delta = x_orig
        for i in range(self.order):
            x_pad = tf.pad(to_delta, self.paddings)
            delta = K.conv2d(x_pad, self.delta_kernel, data_format="channels_last")
            deltas.append(delta)
            to_delta = delta

        return K.concatenate(deltas, axis=-1)

    def get_config(self):
        config = {'n': self.n, 'order': self.order}
        base_config = super(Delta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
def acc_12(y_true, y_pred):
    y_pred = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    y_true = K.max(y_true, axis=-1)
    y_true = K.minimum(y_true, 11.)
    y_pred = K.minimum(y_pred, 11.)
    return K.mean(K.equal(y_true, y_pred))


def dct(n_filters, n_input):
    """
    """

    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2 * n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_input)

    return basis


def log10(x):
    """ keras backend change of base for convenience
    """
    return K.log(x) / K.log(10)


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    assert amin > 0
    assert top_db >= 0

    magnitude = K.abs(S)

    ref_value = np.abs(ref).astype(K.floatx())

    log_spec = log10(K.maximum(amin, magnitude)) * 10.0
    log_spec -= 10.0 * log10(K.maximum(amin, ref_value))

    log_spec = K.maximum(log_spec, K.max(log_spec) - top_db)

    return log_spec


def amplitude_to_decibel(x, amin=1e-10, dynamic_range=80.0):
    """[K] Convert (linear) amplitude to decibel (log10(x)).
    x: Keras tensor or variable.
    amin: minimum amplitude. amplitude smaller than `amin` is set to this.
    dynamic_range: dynamic_range in decibel
    """
    log_spec = 10 * K.log(K.maximum(x, amin)) / np.log(10).astype(K.floatx())
    log_spec = log_spec - K.max(log_spec)  # [-?, 0]
    log_spec = K.maximum(log_spec, -1 * dynamic_range)  # [-80, 0]
    return log_spec
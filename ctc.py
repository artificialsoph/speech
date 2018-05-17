import keras
import keras.backend as K
import kapre
import pandas as pd
import scipy.io.wavfile as wav
import numpy as np
import random
import arrow
import threading
import pprint
import nltk
from lsuv_init import LSUVInit
from soph import ex_generator, center_wave

# https://github.com/keunwoochoi/kapre

ex_df = pd.read_pickle("data/ex_df.pkl")

# DEFINE PARAMS

nltk.download('cmudict')
arpabet = nltk.corpus.cmudict.dict()

words = list(ex_df.raw_label.unique())
words.remove(np.nan)
words.remove("silence")

phone_dict = dict()
phone_set = set()
maxlen = 0
for w in words:
    phones = arpabet[w][0]
    phone_dict[w] = phones
    phone_set |= set(phones)
    if len(phones) > maxlen:
        maxlen = len(phones)
phone_dict["silence"] = ["-"]
alphabet = sorted(list(phone_set)) + ["-"] + ["<b>"]

def text_to_labels(text):
    phones = phone_dict[text]
    ret = [alphabet.index(p) for p in phones]
    return ret


SR = 16000
N_BATCH = 512
N_SEQ = 10
SEQ_SIZE = 1
N_HOP = SR//(N_SEQ*SEQ_SIZE)
N_DFT = max(2**(int(np.log2(N_HOP))+1),512)
N_MELS = 40
DROP = .5
INIT = "he_normal"
ACT = "elu"
N_CAT = len(alphabet)
VAL = ["val", "test"]
TRAIN = ["train"]
# REG = keras.regularizers.l2(0.005)
REG=None
SHIFT = .5

start_time = arrow.now()
current_time = start_time.to('US/Eastern').format('YYYY-MM-DD-HH-mm')




def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
#     y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


## INPUT BLOCK

# these inputs are for the CTC loss
labels = keras.layers.Input(name='labels', shape=[maxlen], dtype='float32')
input_length = keras.layers.Input(name='input_len', shape=[1], dtype='int64')
label_length = keras.layers.Input(name='label_len', shape=[1], dtype='int64')

# input the wav file
input_layer = keras.layers.Input(shape=(1, SR), name='wav_input')
input_block = kapre.time_frequency.MFCC(
    n_mfcc=N_MELS,
    n_mels=N_MELS,
    n_dft=N_DFT,
    n_hop=N_HOP,
    power_melgram=2.0,
    return_decibel_melgram=True,
#     trainable_fb=True,
)(input_layer)
input_block = kapre.utils.DeltaDelta(n=2)(input_block)
input_block = keras.layers.Permute((1, 3, 2))(input_block)
input_block = keras.layers.Reshape((N_MELS*3*SEQ_SIZE,N_SEQ))(input_block)
input_block = keras.layers.Permute((2, 1))(input_block)
input_block = keras.layers.BatchNormalization()(input_block)



## DENSE BLOCK

time = lambda x, y:  keras.layers.TimeDistributed(x)(y)

rnn_block = input_block
for _ in range(4):
    rnn_block = time(keras.layers.Dense(
        100, 
        activation=ACT, 
        kernel_initializer=INIT, 
        kernel_regularizer=REG,
        bias_regularizer=REG
    ), rnn_block)
    rnn_block = time(keras.layers.BatchNormalization(), rnn_block)
    rnn_block = time(keras.layers.Dropout(DROP), rnn_block)

## RNN BLOCK

for _ in range(2):
    rnn_block = keras.layers.Bidirectional(
        keras.layers.GRU(
            100,
            activation=ACT,
            kernel_initializer=INIT,
            dropout=DROP,
            recurrent_dropout=DROP,
            return_sequences=True,
            kernel_regularizer=REG,
            bias_regularizer=REG,
            recurrent_regularizer=REG,
        ))(rnn_block)
    rnn_block = keras.layers.BatchNormalization()(rnn_block)

## OUT BLOCK
y_pred = keras.layers.Dense(N_CAT, kernel_initializer=INIT, activation="softmax")(rnn_block)

## LOSS
loss_out = keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

## MODEL
ctc_model = keras.Model(inputs=[input_layer, labels, input_length, label_length], outputs=loss_out)
ctc_model.summary()

ctc_loss = lambda y_true, y_pred: y_pred

ctc_model.compile(
    loss={'ctc': ctc_loss},
    optimizer="nadam",
    metrics=['accuracy'])

def ex_generator(batch_size=32,
                 shuffle=True,
                 state="train",
                 num_seq=None,
                 input_len=10,
                 shift=0):

    epoch_df = ex_df[ex_df.state.isin(state)]
    num_ex = len(epoch_df)
    indices = np.arange(num_ex)

    # epoch loop runs
    while True:

        # shuffle anew every epoch
        if shuffle:
            epoch_df = epoch_df.sample(frac=1)

        # batch loop
        for i in np.arange(0, num_ex, batch_size):

            batch_df = epoch_df.iloc[i:i + batch_size, :]

            x = np.zeros((len(batch_df), 1, 16000))
            labels = np.zeros((len(batch_df), maxlen))
            label_len = np.zeros(len(batch_df))

            # example loop
            for i in range(len(batch_df)):

                x[i, ...] = center_wave(epoch_df.fn.values[i], shift=SHIFT)

#                 if np.random.rand() < 0.01:
#                     label_is = [N_CAT]
#                 else:
                label_is = text_to_labels(epoch_df.raw_label.values[i])

                labels[i, 0:len(label_is)] = label_is
                label_len[i] = len(label_is)

            inputs = {
                'wav_input': x,
                'labels': labels,
                'input_len': np.full(len(batch_df), input_len),
                'label_len': label_len
            }
            outputs = {
                'ctc': np.zeros([len(batch_df)])
            }  # dummy data for dummy loss function
            yield (inputs, outputs)

log_base = "logs/ctc/{}/".format(current_time)
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=log_base + 'tb',
        batch_size=N_BATCH,
        histogram_freq=0,
        write_grads=False,
        write_images=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=log_base + 'model-checkpoint.hdf5',
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1),
    keras.callbacks.CSVLogger(log_base + 'training.log'),
    keras.callbacks.EarlyStopping(
        patience=4, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        factor=.5, patience=1, verbose=1, min_lr=1e-7)
]


def launchTensorBoard():
    import os
    os.system('pkill tensorboard')
    os.system('tensorboard --logdir=' + log_base + 'tb')
    return


t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

val_data = next(
    ex_generator(
        batch_size=sum(ex_df.state.isin(VAL)),
        shuffle=False,
        state=VAL,
        input_len=N_SEQ))

traing_gen = ex_generator(batch_size=N_BATCH, state=TRAIN, input_len=N_SEQ, shift=SHIFT)

history = ctc_model.fit_generator(
    generator=traing_gen,
    steps_per_epoch=sum(ex_df.state.isin(TRAIN)) / N_BATCH,
    epochs=200,
    verbose=1,
    max_queue_size=100,
    callbacks=callbacks,
    validation_data=val_data)

# out_dict = src_args
# out_dict["train_loss"] = np.min(history.history["loss"])
# out_dict["train_acc"] = np.max(history.history["acc"])
# out_dict["val_loss"] = np.min(history.history["val_loss"])
# out_dict["val_acc"] = np.max(history.history["val_acc"])
# out_dict["current_time"] = current_time
# elapsed = (arrow.now() - start_time).seconds
# out_dict["elapsed"] = elapsed
# out_dict["epochs"] = len(history.history["loss"])
# out_dict["epoch_dur"] = elapsed / len(history.history["loss"])
# out_dict["end_time"] = arrow.now().to('US/Eastern').format('YYYY-MM-DD-HH-mm')

# with open("logs/ctc-log", "a") as f:
#     f.writelines(str(out_dict) + "\n")

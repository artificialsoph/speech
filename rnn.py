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
from soph import ex_generator, center_wave

ex_df = pd.read_pickle("data/ex_df.pkl")

src_args = {
    "dropout_prob": .1,
    "activation": "elu",
    "batch_size": 512,
    "regularize": True,
    "l2_reg": 0.0005,
    "init": "glorot_normal",                   # glorot, tnormal, lsuv
    "init_param": 0.05,
    "filters_start": 50,
    "filters_step": 25,
    "kernel_size": 10,
    "n_mfcc": 40,
    "n_mels": 80,
    "n_dft": 512,
    "n_hop": 160,
    "pool": "max",
    "pool_pad": "same",
    "batch_normalize": True,
    "train_state": ["train"],
    "val_state": ["val", "test"],
    "p_transform": 1,
    "vol_range": .1,
    "shift": 1,
    "delta_delta": True,
    "lr_step": .5,
    "lr_patience": 1,
    "power_melgram": 2.0,
    "return_decibel_melgram": True,
    "trainable_fb": False,
    "trainable_kernel": False,
    "early_patience": 4
}
pprint.pprint(src_args)

num_cat = 12
start_time = arrow.now()
current_time = start_time.to('US/Eastern').format('YYYY-MM-DD-HH-mm')



drop = src_args["dropout_prob"]

init = src_args["init"]

if init == None:
    init = 'glorot_uniform'
elif init == "tnormal":
    init_param = src_args["init_param"] if src_args["init_param"] else 0.01
    init = keras.initializers.TruncatedNormal(stddev=src_args["init_stdd"])
elif init == "lsuv":
    init = keras.initializers.RandomNormal(stddev=1)

if src_args["regularize"]:
    reg = keras.regularizers.l2(src_args["l2_reg"])
else:
    reg = None

input_layer = keras.layers.Input(shape=(1, 16000))
input_block = kapre.time_frequency.MFCC(
    n_mfcc=int(src_args["n_mfcc"]),
    n_mels=int(src_args["n_mels"]),
    n_dft=int(src_args["n_dft"]),
    n_hop=int(src_args["n_hop"]),
    power_melgram=src_args["power_melgram"],
    return_decibel_melgram=src_args["return_decibel_melgram"],
    trainable_kernel=src_args["trainable_kernel"],
    trainable_fb=src_args["trainable_fb"],
)(input_layer)
if src_args["delta_delta"]:
    input_block = kapre.utils.DeltaDelta(n=2)(input_block)
    input_block = keras.layers.Permute((1, 3, 2))(input_block)
    input_block = keras.layers.Reshape((src_args["n_mfcc"]*3*5,20))(input_block)
else:
    input_block = keras.layers.Permute((1, 3, 2))(input_block)
    input_block = keras.layers.Reshape((src_args["n_mfcc"]*5,20))(input_block)

input_block = keras.layers.Permute((2, 1))(input_block)

time = lambda x, y:  keras.layers.TimeDistributed(x)(y)

rnn_block = time(keras.layers.Dense(100, activation=src_args["activation"]), input_block)
rnn_block = time(keras.layers.BatchNormalization(), rnn_block)
rnn_block = time(keras.layers.Dropout(drop), rnn_block)
rnn_block = time(keras.layers.Dense(50, activation=src_args["activation"]), rnn_block)
rnn_block = time(keras.layers.BatchNormalization(), rnn_block)
rnn_block = time(keras.layers.Dropout(drop), rnn_block)

rnn_block = keras.layers.Bidirectional(
    keras.layers.GRU(
        50,
        activation=src_args["activation"],
        dropout=drop,
        recurrent_dropout=drop,
        return_sequences=True,
#         return_state=True,
    ))(rnn_block)
rnn_block = keras.layers.Bidirectional(
    keras.layers.GRU(
        50,
        activation=src_args["activation"],
        dropout=drop,
        recurrent_dropout=drop,
    ))(rnn_block)
rnn_block = keras.layers.BatchNormalization()(rnn_block)
rnn_block = keras.layers.Dropout(drop)(rnn_block)

output_layer = keras.layers.Dense(num_cat, activation="softmax")(rnn_block)

rnn_model = keras.Model(inputs=input_layer, outputs=output_layer)
rnn_model.summary()

log_base = "logs/rnn/{}/".format(current_time)
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=log_base + 'tb',
        batch_size=src_args["batch_size"],
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
        patience=src_args["early_patience"], verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        factor=src_args["lr_step"], patience=src_args["lr_patience"], verbose=1, min_lr=1e-7)
]

rnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='nadam',
    metrics=['accuracy'])


def launchTensorBoard():
    import os
    os.system('pkill tensorboard')
    os.system('tensorboard --logdir=' + log_base + 'tb')
    return


t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

val_data = next(ex_generator(
    batch_size=sum(ex_df.state.isin(src_args["val_state"])),
    shuffle=False,
    state=src_args["val_state"],
    vol_range=0,
    displacement=0,
    p_transform=0))

traing_gen = ex_generator(
    batch_size=src_args["batch_size"],
    shuffle=True,
    state=src_args["train_state"],
    vol_range=src_args["vol_range"],
    shift=src_args["shift"],
    p_transform=src_args["p_transform"])

history = rnn_model.fit_generator(
    generator=traing_gen,
    steps_per_epoch=sum(ex_df.state.isin(
        src_args["train_state"])) / src_args["batch_size"],
    epochs=200,
    verbose=1,
    max_queue_size=100,
    callbacks=callbacks,
    validation_data=val_data
)

out_dict = src_args
out_dict["train_loss"] = np.min(history.history["loss"])
out_dict["train_acc"] = np.max(history.history["acc"])
out_dict["val_loss"] = np.min(history.history["val_loss"])
out_dict["val_acc"] = np.max(history.history["val_acc"])
out_dict["current_time"] = current_time
elapsed = (arrow.now() - start_time).seconds
out_dict["elapsed"] = elapsed
out_dict["epochs"] = len(history.history["loss"])
out_dict["epoch_dur"] = elapsed / len(history.history["loss"])
out_dict["end_time"] = arrow.now().to('US/Eastern').format('YYYY-MM-DD-HH-mm')

with open("logs/rnn-log", "a") as f:
    f.writelines(str(out_dict) + "\n")

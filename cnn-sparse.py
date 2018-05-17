import keras
import keras.backend as K
import kapre
import pandas as pd
import scipy.io.wavfile as wav
import numpy as np
import random
import arrow
import threading
from soph import soph_scaler, center_wave, ex_generator, MFCC

# https://github.com/keunwoochoi/kapre

ex_df = pd.read_pickle("data/ex_df.pkl")
current_time = arrow.now().to('US/Eastern').format('YYYY-MM-DD-HH-mm')


src_args = {
    "dropout_prob": .5,
    "activation": "elu",
    "batch_size": 64,
    "l2_reg": 0.0005,
    "init_stdd": 0.01,
    "cnn_blocks": [
        [[180, 5]],
        [[200, 1], [220, 3]],
        [[220, 1], [240, 3]],
        [[240, 1], [260, 3]],
        [[260, 1], [280, 2]],
        [[280, 1], [300, 2]],
    ],
    "mel_trainable": False,
    "train_state": ["train"],
    "val_state": ["val", "test"],
    "p_transform": 0,
    "vol_range": 0,
    "displacement": 0,
    "shift": 0,
    "n_mels": 63,
    "n_dft": 512,
    "fmin": 0,
}
num_cat = 12
print(src_args)

act = src_args["activation"]
kern = (3, 3)
drop = src_args["dropout_prob"]
cnn_blocks = src_args["cnn_blocks"]
init = keras.initializers.TruncatedNormal(stddev=src_args["init_stdd"])
reg = keras.regularizers.l2(src_args["l2_reg"])

input_layers = [
    keras.layers.InputLayer(input_shape=(1, 16000)),

    kapre.time_frequency.MFCC(n_mfcc=40,
                              n_mels=128,
                              n_dft=512,
                              power_melgram=2.0,
                              return_decibel_melgram=True,),
    # keras.layers.BatchNormalization(),
]

cnn_layers = [
    keras.layers.Conv2D(
        128,
        (20,10),
        padding="valid",
        activation=act,
        kernel_initializer=init,
        kernel_regularizer=reg,
        bias_regularizer=reg),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(drop),
    keras.layers.Conv2D(
        128,
        (15,10),
        padding="valid",
        activation=act,
        kernel_initializer=init,
        kernel_regularizer=reg,
        bias_regularizer=reg),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(drop),
    keras.layers.Conv2D(
        128,
        (7,10),
        padding="valid",
        activation=act,
        kernel_initializer=init,
        kernel_regularizer=reg,
        bias_regularizer=reg),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(drop),
]

output_layers = [
    keras.layers.Flatten(),
    keras.layers.Dense( 256, activation=src_args["activation"]),
    keras.layers.Dropout(src_args["dropout_prob"]),
    keras.layers.Dense( num_cat, activation="softmax"),
]

cnn_model = keras.Sequential(input_layers + cnn_layers + output_layers)
cnn_model.summary()


log_base = "logs/cnn/{}/".format(current_time)
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir= log_base+'tb',
        batch_size=src_args["batch_size"],
        histogram_freq=0,
        write_grads=False,
        write_images=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=log_base+'model-checkpoint.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1),
    keras.callbacks.CSVLogger(log_base+'training.log'),
    keras.callbacks.EarlyStopping(patience=5, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=1, verbose=1, min_lr=1e-8)
]

cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='nadam',
    metrics=['accuracy'])

def launchTensorBoard():
    import os
    os.system('pkill tensorboard')
    os.system('tensorboard --logdir=' + log_base+'tb')
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

cnn_model.fit_generator(
    generator=ex_generator(
        batch_size=src_args["batch_size"],
        shuffle=True,
        state=src_args["train_state"],
        shift=src_args["shift"],
        vol_range=src_args["vol_range"],
        displacement=src_args["displacement"],
        p_transform=src_args["p_transform"]),
    steps_per_epoch=sum(ex_df.state.isin(src_args["train_state"])) / src_args["batch_size"],
    epochs=200,
    verbose=1,
    max_queue_size=100,
    callbacks=callbacks,
    validation_data=val_data
)

import keras
import keras.backend as K
import kapre
import pandas as pd
import scipy.io.wavfile as wav
import numpy as np
import random
import arrow
import threading
from soph import soph_scaler, center_wave, ex_generator

# https://github.com/keunwoochoi/kapre
            
    
    
ex_df = pd.read_pickle("data/ex_df.pkl")
current_time = arrow.now().to('US/Eastern').format('YYYY-MM-DD-HH-mm')

src_args = {
    "dropout_prob": .5,
    "activation": "elu",
    "batch_size": 32,
    "l2_reg": 0.0005,
    "init_stddev": 0.01,
    "cnn_blocks": [
        [[64, 3]],
        [[64, 3]],
        [[128, 3]],
        [[128, 3]],
        [[256, 3]],
        [[256, 3]],
        [[512, 3]],
    ],
    "mel_trainable": False,
    "train_state": ["train"],
    "val_state": ["val"],
    "p_transform": 0,
    "vol_range": 0,
    "displacement": 0,
    "shift":0,
}
num_cat = 12
num_blocks = len(src_args["cnn_blocks"])
act = src_args["activation"]
drop = src_args["dropout_prob"]
init = keras.initializers.TruncatedNormal(src_args["init_stddev"])

input_layers = [
    keras.layers.InputLayer(input_shape=(1, 16000)),
    kapre.time_frequency.Melspectrogram(
        n_mels=40,
        sr=16000, 
        return_decibel_melgram=True,
        n_dft=1024,
        fmin=300,
    ),
#     kapre.augmentation.AdditiveNoise(power=0.01),
#     kapre.utils.Normalization2D(str_axis='freq'),
]

#melspectrogram output: None x freq_steps x time_steps x 1

cnn_layers = [
    keras.layers.Conv2D(64,
                        kernel_size=(8,20),
#                         strides=(1, 1),
                        padding="valid", 
                        activation=act, 
                        kernel_initializer=init),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((3,1), padding='same'),
    keras.layers.Dropout(drop),
    
    keras.layers.Conv2D(64,
                        kernel_size=(4,10), 
                        padding="valid", 
                        activation=act, 
                        kernel_initializer=init),
    keras.layers.BatchNormalization(),
#     keras.layers.MaxPooling2D((2,1)),
    keras.layers.Dropout(drop),
    
#     keras.layers.Conv2D(256,
#                         kernel_size=(3,3), 
#                         padding="valid", 
#                         activation=act, 
#                         kernel_initializer=init),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(drop),
]
    
class_layers = [
    keras.layers.Flatten(),
    
#     keras.layers.Dense(128, activation=act, kernel_initializer=init),
#     keras.layers.Dropout(drop),
    
    keras.layers.Dense(32, activation="linear", kernel_initializer=init),
    keras.layers.Dropout(drop),
    
    keras.layers.Dense(128, activation=act, kernel_initializer=init),
    keras.layers.Dropout(drop),
    
    keras.layers.Dense(num_cat, activation="softmax"),
]
cnn_model = keras.Sequential(input_layers+cnn_layers+class_layers)
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
    keras.callbacks.EarlyStopping(patience=10, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=1, verbose=1, min_lr=1e-5)
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

# (x,y,cw) = next(ex_generator(
#         batch_size=sum(ex_df.state.isin(src_args["train_state"])),
#         shuffle=False,
#         state=src_args["train_state"],
#         vol_range=0,
#         displacement=0,
#         p_transform=0))

# cnn_model.fit(
#     x=x,
#     y=y,
#     batch_size=src_args["batch_size"],
#     sample_weight=cw,
#     validation_data=val_data,
#     epochs=200,
#     verbose=1,
#     callbacks=callbacks,
# )
    

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
    
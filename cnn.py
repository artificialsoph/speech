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
from lsuv_init import LSUVInit
from soph import ex_generator, center_wave

# https://github.com/keunwoochoi/kapre

ex_df = pd.read_pickle("data/ex_df.pkl")

src_args = {
    "dropout_prob": .1,
    "activation": "elu",
    "batch_size": 512,
    "regularize": True,
    "l2_reg": 0.001,
    "init": "glorot_normal",                   # glorot, tnormal, lsuv
    "init_param": 0.05,
    "filters_start": 20,
    "filters_step": 20,
    "kernel_size": 8,
    "cnn_pad": "same",
    "cnn_stride": 1,
    "cnn_stack": 1,
    "retain": 20,
    "n_mfcc": 40,
    "n_mels": 40,
    "n_dft": 512,
    "n_hop": 256,
    "pool": "max",
    "pool_pad": "same",
    "batch_normalize": True,
    "train_state": ["train"],
    "val_state": ["val", "test"],
    "p_transform": 0,
    "vol_range": .1,
    "shift": 0,
    "delta_delta": True,
    "lr_step": .5,
    "lr_patience": 1,
    "power_melgram": 2.0,
    "return_decibel_melgram": True,
    "trainable_fb": False,
    "trainable_kernel": False,
    "early_patience": 4,
}
num_cat = 12
pprint.pprint(src_args)
start_time = arrow.now()
current_time = start_time.to('US/Eastern').format('YYYY-MM-DD-HH-mm')

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
# input_block = kapre.time_frequency.Melspectrogram(
#     n_mels=int(src_args["n_mels"]),
#     n_dft=int(src_args["n_dft"]),
#     n_hop=int(src_args["n_hop"]),
#     power_melgram=src_args["power_melgram"],
#     return_decibel_melgram=src_args["return_decibel_melgram"],
#     trainable_kernel=src_args["trainable_kernel"],
#     trainable_fb=src_args["trainable_fb"],
# )(input_layer)

if src_args["delta_delta"]:
    input_block = kapre.utils.DeltaDelta(n=2)(input_block)

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


def pool(x):
    """ add a pool layer to X according to src_args or
    """

    _, nin1, nin2, _ = x.shape.as_list()

    k = 2
    s = 2
    if src_args["pool_pad"] == "valid":
        p = 0
    else:
        p = k

    def calc_dim(x): return int(((x + 2 * p - k) / s) + 1)

    if (calc_dim(nin1) < 1) or (calc_dim(nin2) < 1):
        return x, False

    if src_args["pool"] == "max":
        return keras.layers.MaxPool2D(padding=src_args["pool_pad"])(x), True
    elif src_args["pool"] == "avg":
        return keras.layers.AvgPool2D(padding=src_args["pool_pad"])(x), True


def conv(x, i=0, j=0):
    """ add a conv layer to X according to src_args or
    """

    _, nin1, nin2, _ = x.shape.as_list()

    stop_vals = (1, None)

    if np.any([d in stop_vals for d in (nin1, nin2)]):
        return x, False

    k = max(2, min(nin1 // 2, nin2 // 2, src_args["kernel_size"]))
    k = max(k // (j + 1), 2)
    s = min(nin1 // 2, nin2 // 2, src_args["cnn_stride"])

    if src_args["cnn_pad"] == "valid":
        p = 0
    else:
        p = k

    def calc_dim(x): return int(((x + 2 * p - k) / s) + 1)

    if (calc_dim(nin1) < 1) or (calc_dim(nin2) < 1):
        return x, False

    x = keras.layers.Conv2D(
        filters=int(src_args["filters_start"] + i * src_args["filters_step"]),
        kernel_size=int(k),
        padding=src_args["cnn_pad"],
        strides=int(src_args["cnn_stride"]),
        activation=src_args["activation"],
        kernel_initializer=init,
        kernel_regularizer=reg,
        bias_regularizer=reg,
        name="conv_{}.{}_{}".format(i, j, k))(x)

    if src_args["batch_normalize"]:
        x = keras.layers.BatchNormalization()(x)

    return x, True


cnn_block = input_block
continue_flag = True
cnn_i = 0
cnn_stack = src_args["cnn_stack"] if src_args["cnn_stack"] else 1
assert cnn_stack >= 1

while continue_flag:

    for j in range(cnn_stack):
        cnn_block, cnn_flag = conv(cnn_block, cnn_i, j)
        if not cnn_flag:
            break

    if not cnn_flag:
        continue_flag = False
        break
    cnn_i += 1

    cnn_block, pool_flag = pool(cnn_block)
    if not pool_flag:
        continue_flag = False
        break

    if (src_args["activation"] == "selu"):
        cnn_block = keras.layers.AlphaDropout(
            src_args["dropout_prob"])(cnn_block)
        # shape = cnn_block.shape.as_list()
        # if shape[1] is None:
        #     shape[1] = 1
        # if shape[2] is None:
        #     shape[2] = 2
        # cnn_block.set_shape(shape)

    else:
        cnn_block = keras.layers.Dropout(src_args["dropout_prob"])(cnn_block)

out_block = cnn_block
out_block = keras.layers.Flatten()(out_block)
output_layer = keras.layers.Dense(num_cat, activation="softmax")(out_block)

cnn_model = keras.Model(inputs=input_layer, outputs=output_layer)
cnn_model.summary()

if src_args["init"] == "lsuv":

    init_param = src_args["init_param"] if src_args["init_param"] else 0.1

    lsuv_batch = 2048
    test_df = ex_df.loc[ex_df.state == "submission", :].copy()
    test_df = test_df.sample(frac=1).reset_index()
    x = np.zeros((lsuv_batch, 1, 16000))
    for x_i, j in enumerate(range(lsuv_batch)):

        x[x_i, ...] = center_wave(test_df.fn[j])

    model = LSUVInit(cnn_model, x, margin=init_param)

log_base = "logs/cnn/{}/".format(current_time)
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

cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='nadam',
    metrics=['sparse_categorical_accuracy'])


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

history = cnn_model.fit_generator(
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

with open("logs/cnn-log", "a") as f:
    f.writelines(str(out_dict) + "\n")

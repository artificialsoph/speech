import keras
import keras.backend as K
import kapre
import pandas as pd
import scipy.io.wavfile as wav
import numpy as np
import random
import arrow
import threading
import hyperopt
import pprint
from hyperopt import hp
from soph import soph_scaler, center_wave, ex_generator

# https://github.com/keunwoochoi/kapre

ex_df = pd.read_pickle("data/ex_df.pkl")

src_space = {
    "dropout_prob": hp.uniform("dropout_prob",0,.6), #[0,1]
    "activation": hp.choice("activation", ("elu", "relu")), #{ elu, relu}
    "batch_size": 64, #2^int[1,7]
    "regularize": hp.choice("regularize", (True, False)), #{T,F}
    "l2_reg": hp.loguniform("l2_reg",-10,-1), #log([e-8,e-1])
    "init_stdd": hp.loguniform("init_stdd",-10,-1), #log([e-8,e-1])
    "filters_start": hp.quniform("filters_start",40,100,20), #[40,200]
    "filters_step": hp.quniform("filters_step",0,60,10), #[0,60]
    "kernel_size": hp.quniform("kernel_size",2,10,1), #[1,10]
    "cnn_pad": hp.choice("cnn_pad", ("same", "valid")), #{same,valid}
    "cnn_stride": hp.quniform("cnn_stride",1,2,1), #{1,2}
    "n_mfcc": hp.quniform("n_mfcc",16,100,2), #int[16,100]
    "n_mels": hp.quniform("n_mels",16,200,4), #int[16,200]
    "n_dft": 512, #2^int[8,12]
    "n_hop": 256, #2^int[7,12]
    "pool": hp.choice("pool", ("max", "avg")), #{max,avg}
    "pool_pad": hp.choice("pool_pad", ("same", "valid")), #{same,valid}
    "batch_normalize": hp.choice("batch_normalize", (True, False)), #{T,F}
    "train_state": ["train"],
    "val_state": ["val", "test"],
    "p_transform": hp.uniform("p_transform",0,.75), #[0,1]
    "vol_range": hp.uniform("vol_range",0,.5), #[0,1]
    "displacement": hp.uniform("displacement",0,.5), #[0,1]
}
num_cat = 12


def objective(src_args):
    
    pprint.pprint(src_args)
    current_time = arrow.now().to('US/Eastern').format('YYYY-MM-DD-HH-mm')
    
    input_layer = keras.layers.Input(shape=(1, 16000))
    input_block = kapre.time_frequency.MFCC(n_mfcc=int(src_args["n_mfcc"]),
                                  n_mels=int(src_args["n_mels"]),
                                  n_dft=int(src_args["n_dft"]),
                                  power_melgram=2.0,
                                  return_decibel_melgram=True,)(input_layer)

    def pool(x):
        if src_args["pool"] == "max":
            return keras.layers.MaxPool2D(padding=src_args["pool_pad"])(x)
        elif src_args["pool"] == "avg":
            return keras.layers.AvgPool2D(padding=src_args["pool_pad"])(x)

    drop = src_args["dropout_prob"]
    init = keras.initializers.TruncatedNormal(stddev=src_args["init_stdd"])
    if src_args["regularize"]:
        reg = keras.regularizers.l2(src_args["l2_reg"])
    else:
        reg = None

    def cnn(x, i=0):
        return keras.layers.Conv2D(
            filters=int(src_args["filters_start"] + i*src_args["filters_step"]),
            kernel_size=int(src_args["kernel_size"]),
            padding=src_args["cnn_pad"],
            strides=int(src_args["cnn_stride"]),
            activation=src_args["activation"],
            kernel_initializer=init,
            kernel_regularizer=reg,
            bias_regularizer=reg)(x)


    def calc_cnn_dim(nin):
        k = src_args["kernel_size"]
        if src_args["cnn_pad"] == "valid":
            p = 0
        else:
            p = k
        s = src_args["cnn_stride"]

        return int(((nin + 2 * p - k) / s) + 1)


    def calc_cnn_size(layer, layer_type):
        if layer_type == "cnn":
            k = src_args["kernel_size"]
            if src_args["cnn_pad"] == "valid":
                p = 0
            else:
                p = k
            s = src_args["cnn_stride"]
        elif layer_type == "pool":
            k = 2
            if src_args["pool_pad"] == "valid":
                p = 0
            else:
                p = k
            s = 2
        calc_dim = lambda x: int(((x + 2 * p - k) / s) + 1)
        layer_in = np.array(layer.shape.as_list())
        layer_out = (layer_in[0], 
                     calc_dim(layer_in[1]),
                     calc_dim(layer_in[2]),
                     layer_in[3])

        return np.array(layer_out)[1:]


    cnn_block = input_block
    continue_flag = True
    cnn_i = 0
    while continue_flag:

        cnn_flag = np.all(calc_cnn_size(cnn_block, "cnn") > 0)

        cnn_flag = cnn_flag & (sum(cnn_block.shape.as_list()[1:3]) > 2)
        cnn_flag = cnn_flag & (cnn_i < 20)
        if cnn_flag:
            cnn_block = cnn(cnn_block, cnn_i)
            cnn_i += 1
        else:
            continue_flag = False
            break

    #     print(calc_cnn_size(cnn_block, "pool"))
        pool_size_flag = np.all(calc_cnn_size(cnn_block, "pool") > 0)
        if pool_size_flag:
            cnn_block = pool(cnn_block)

        if src_args["batch_normalize"]:
            cnn_block = keras.layers.BatchNormalization()(cnn_block)

        cnn_block = keras.layers.Dropout(src_args["dropout_prob"])(cnn_block)


    out_block = cnn_block
    out_block = keras.layers.Flatten()(out_block)
    output_layer = keras.layers.Dense(num_cat, activation="softmax")(out_block)

    cnn_model = keras.Model(inputs=input_layer, outputs=output_layer)
    cnn_model.summary()

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
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1),
        keras.callbacks.CSVLogger(log_base + 'training.log'),
        keras.callbacks.EarlyStopping(patience=4, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.1, patience=1, verbose=1, min_lr=1e-8)
    ]

    cnn_model.compile(
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

    history = cnn_model.fit_generator(
        generator=ex_generator(
            batch_size=src_args["batch_size"],
            shuffle=True,
            state=src_args["train_state"],
            vol_range=src_args["vol_range"],
            displacement=src_args["displacement"],
            p_transform=src_args["p_transform"]),
        steps_per_epoch=sum(ex_df.state.isin(
            src_args["train_state"])) / src_args["batch_size"],
        epochs=200,
        verbose=1,
        max_queue_size=100,
        callbacks=callbacks,
        validation_data=val_data
    )
    
    obj = np.min(history.history["val_loss"])
    with open("logs/opt-log", "w") as f:
        f.writelines("{}, {}".format(current_time, obj))
    
    return obj
    
hyperopt.fmin(objective, src_space, algo=hyperopt.tpe.suggest, max_evals=200, verbose=1)
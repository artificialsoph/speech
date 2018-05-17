import keras
import numpy as np

print("loading data...")

xy = np.load('data/xy.npz')

x = np.concatenate((xy["x"], xy["x_bg"]))
y = np.concatenate((xy["y"], xy["y"]))

num_cat = y.max() + 1

print("finished loading data.")

dropout_prob = .3
init_stddev = 0.01

ff_layers = [
    keras.layers.InputLayer(input_shape=x.shape[1:]),
    keras.layers.Flatten(),
    keras.layers.Dense(1000, activation="relu"),
    keras.layers.Dropout(dropout_prob),
    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dropout(.5*dropout_prob),

    # Classification
    keras.layers.Dense(
        num_cat,
        activation="softmax",
        kernel_initializer=keras.initializers.TruncatedNormal(
            stddev=init_stddev)),
]
ff_model = keras.Sequential(ff_layers)
ff_model.summary()


callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='logs/ff/weights.{epoch:02d}-{val_acc:.3f}.hdf5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1),
    keras.callbacks.CSVLogger('logs/ff/training.log'),
    keras.callbacks.EarlyStopping(patience=6, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.1, patience=2, verbose=1, min_lr=1e-6)
]

ff_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Nadam(lr=0.001),
    metrics=['accuracy'])

ff_model.fit(
    x=x,
    y=y,
    validation_data=(xy["x_val"], xy["y_val"]),
    epochs=100,
    verbose=1,
    callbacks=callbacks,
    initial_epoch=0)
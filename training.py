import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, schedules
import keras.callbacks
from keras.layers.merge import Concatenate, Add
from keras import backend as be
from keras.constraints import max_norm

def model_1():
    input1 = Input(shape=(30,))

    x = Dense(30)(input1)
    x = Activation("tanh")(x)
    x = Dropout(0.20)(x)
    x = Dense(30)(x)
    x = Activation("tanh")(x)
    x = Dense(30)(x)
    x = Activation("tanh")(x)
    x = Dense(30)(x)
    x = Activation("tanh")(x)
    x = Dropout(0.20)(x)
    x = Dense(30)(x)
    x = Activation("tanh")(x)
    x = Dense(15)(x)
    x = Activation("tanh")(x)

    x = Concatenate()([input1, x])
    x = Dense(45)(x)
    x = Activation("tanh")(x)
    x = Dense(45)(x)
    x = Activation("tanh")(x)
    x = Dropout(0.10)(x)
    x = Dense(30)(x)
    x = Activation("tanh")(x)
    x = Dense(15)(x)
    x = Activation("tanh")(x)

    x = Concatenate()([input1, x])
    x = Dense(45)(x)
    x = Activation("tanh")(x)
    x = Dense(45)(x)
    x = Activation("tanh")(x)
    x = Dropout(0.10)(x)
    x = Dense(30)(x)
    x = Activation("tanh")(x)
    x = Dense(15)(x)
    x = Activation("tanh")(x)

    x = Concatenate()([input1, x])
    x = Dense(45)(x)
    x = Activation("tanh")(x)
    x = Dense(45)(x)
    x = Activation("tanh")(x)
    x = Dropout(0.10)(x)
    x = Dense(30)(x)
    x = Activation("tanh")(x)
    x = Dense(15)(x)
    x = Activation("tanh")(x)

    x = Concatenate()([input1, x])
    x = Dense(45)(x)
    x = Activation("tanh")(x)
    x = Dense(45)(x)
    x = Activation("tanh")(x)
    x = Dropout(0.10)(x)
    x = Dense(30)(x)
    x = Activation("tanh")(x)
    x = Dense(15)(x)
    final = Activation("tanh")(x)

    model = Model(inputs=input1,outputs=final)
    return model

def model_2():
    input_cords = Input(shape=(30,))

    x = Dense(1024, kernel_constraint=max_norm(1))(input_cords)
    x = BatchNormalization(trainable = False)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, kernel_constraint=max_norm(1))(x)
    x = BatchNormalization(trainable = False)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, kernel_constraint=max_norm(1))(x)
    x = BatchNormalization(trainable = False)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Concatenate()([input_cords, x])

    x = Dense(1024, kernel_constraint=max_norm(1))(x)
    x = BatchNormalization(trainable = False)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, kernel_constraint=max_norm(1))(x)
    x = BatchNormalization(trainable = False)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Concatenate()([input_cords, x])

    x = Dense(45)(x)
    output_cords = Activation("relu")(x)

    model = Model(inputs=input_cords,outputs=output_cords)
    return model

def MSE(y_true, y_pred):
    return be.sqrt(be.sum(be.square(y_true - y_pred), axis=-1, keepdims=True))

def train_network_model_1():
    file = '/content/gdrive/My Drive/Final/data_sets/set86.npy'
    data = np.load(file)

    np.random.seed(30)
    np.random.shuffle(data)

    y_train = data[:,:,2]
    X_data = data[:,:,0:2]
    X_train = X_data.reshape(X_data.shape[0], 30)

    def lr_schedule(epoch, lr):
        if (epoch != 0 and epoch % 10 == 0):
            return (lr * 0.96 ** (1 + epoch / 200))
        else:
            return lr

    my_callbacks_1 = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto'),
    keras.callbacks.ModelCheckpoint(filepath='/content/gdrive/My Drive/Final/weights/weight_model_1.hdf5', verbose=1, save_best_only=True),
    keras.callbacks.LearningRateScheduler(lr_schedule)]

    model = model_1()
    model.compile(optimizer=keras.optimizers.RMSprop(), loss = MSE)
    model.fit(X_train, y_train, nb_epoch=200, validation_split=0.2, shuffle=True, batch_size = 1024, callbacks = my_callbacks_1, verbose=1)

def train_network_model_2():
    file = '/content/gdrive/My Drive/Final/data_sets/set86.npy'
    data = np.load(file)

    np.random.seed(30)
    np.random.shuffle(data)

    y_train = data.reshape((data.shape[0], 45))
    X_data = data[:,:,0:2]
    X_train = X_data.reshape(X_data.shape[0], 30)

    def lr_schedule(epoch, lr):
        if (epoch != 0 and epoch % 10 == 0):
            return (lr * 0.96 ** (1 + epoch / 200))
        else:
            return lr
      
    my_callbacks_2 = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto'),
    keras.callbacks.ModelCheckpoint(filepath='/content/gdrive/My Drive/Final/weights/weight_model_2.hdf5', verbose=1, save_best_only=True),
    keras.callbacks.LearningRateScheduler(lr_schedule)]

    model = model_2()
    model.compile(optimizer= keras.optimizers.Adam(), loss = MSE)
    model.fit(X_train, y_train, nb_epoch=200, validation_split=0.2, shuffle=True, batch_size = 256, callbacks = my_callbacks_2, verbose=1)

#train_network_model_1()
train_network_model_2()

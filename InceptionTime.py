import numpy as np
import h5py
from scipy.io import arff
import math
import random
import warnings
import gc
import traceback
import pandas as pd
import seaborn as sns
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    roc_auc_score, average_precision_score
from imblearn.metrics import geometric_mean_score
from openpyxl import load_workbook
import time
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import normalize

# Important: set visible GPUs before importing any Tensorflow or Keras library
# GPU device IDs: (GeForce GTX 1080Ti = 0 / TITAN V = 1 / TITAN V = 2)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session, clear_session
from keras.layers import Conv1D, Input, concatenate, Dense, Flatten, Dropout, BatchNormalization, Activation, \
    MaxPooling1D, SimpleRNN, CuDNNLSTM, CuDNNGRU, Bidirectional, Reshape, Permute, RepeatVector, multiply, Lambda, dot,\
    LocallyConnected1D, Masking
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, EarlyStopping
from keras.utils import Sequence
from keras.utils.vis_utils import plot_model
import keras


def _inception_module(output_directory, input_shape, nb_classes, verbose, build, batch_size,
                 nb_filters, use_residual, use_bottleneck, depth, kernel_size, nb_epochs,X_train, X_val,X_test,
                 y_train,y_val,y_test,input_tensor, stride=1, activation='linear'):
    bottleneck_size = 32
    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size,kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        x = keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception)
        conv_list.append(x)
        


    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)
    
    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    x = Dropout(0.2)(x)
    return x

def build_model(output_directory, input_shape, nb_classes, verbose, build, batch_size,
                 nb_filters, use_residual, use_bottleneck, depth, kernel_size, nb_epochs,X_train, X_val,X_test,
                 y_train,y_val,y_test):
    input_layer = Input(input_shape)
    print('hona bai')
    print('shape',input_layer.shape)
    input_layer_rs = Reshape((500,1))(input_layer)
    input_res = input_layer_rs

    for d in range(depth):
        if d == 0:

            x = _inception_module(output_directory, input_shape, nb_classes, verbose, build, batch_size,
                 nb_filters, use_residual, use_bottleneck, depth, kernel_size, nb_epochs,X_train, X_val,X_test,
                 y_train,y_val,y_test,input_layer_rs,stride=1, activation='linear')
            x = Dropout(0.2)(x)

        else:
            x = _inception_module(output_directory, input_shape, nb_classes, verbose, build, batch_size,
                 nb_filters, use_residual, use_bottleneck, depth, kernel_size, nb_epochs,X_train, X_val,X_test,
                 y_train,y_val,y_test,x, stride=1, activation='linear')
            x = Dropout(0.2)(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                  min_lr=0.0001)

    file_path = output_directory + 'best_model.hdf5'

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                       save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    callbacks = [reduce_lr, model_checkpoint]

    return model

def train(output_directory, input_shape, nb_classes, verbose, build, batch_size,
                 nb_filters, use_residual, use_bottleneck, depth, kernel_size, nb_epochs,X_train, X_val,X_test,
                 y_train,y_val,y_test):

    if build == True:
        model = build_model(output_directory, input_shape, nb_classes, verbose, build, batch_size,
                 nb_filters, use_residual, use_bottleneck, depth, kernel_size, nb_epochs,X_train, X_val,X_test,
                 y_train,y_val,y_test)
        print(model.summary())
        plot_model(model, to_file='model.png', show_shapes=True)
        model.save_weights(output_directory + 'model_init.hdf5')

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                  min_lr=0.0001)
    file_path = output_directory + 'best_model.hdf5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                       save_best_only=True)
    print(type(model))
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    
    callbacks = [reduce_lr, model_checkpoint]
    steps_per_epoch_train = math.ceil(len(y_train) / batch_size)
    steps_per_epoch_val = math.ceil(len(y_val) / batch_size)
    steps_per_epoch_test = math.ceil(len(y_test) / batch_size)
    history = model.fit(X_train,y_train,batch_size=batch_size, 
                                                      epochs=150, validation_data=(X_val,y_val),
                                                      callbacks=callbacks)
    model.save(output_directory)
    model.save_weights(output_directory + 'model_weights.hdf5')
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def test(output_directory, input_shape, nb_classes, verbose, build, batch_size,
                 nb_filters, use_residual, use_bottleneck, depth, kernel_size, nb_epochs,X_train, X_val,X_test,
                 y_train,y_val,y_test):
    
    model = load_model(output_directory + 'best_model.hdf5')
    print('Training data shape:',X_train.shape)
    print('Validation data shape:',X_val.shape)
    print('Test data shape:',X_test.shape)

    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    predictions_test = model.predict(X_test)

    print('Training data shape:',X_train.shape)
    print('Validation data shape:',X_val.shape)
    print('Test data shape:',X_test.shape)

    print('Predictions train shape:', predictions_train.shape)
    print('Predictions test shape:', predictions_test.shape)

    # round predictions
    rounded_train = [round(x[0]) for x in predictions_train]
    rounded_val = [round(x[0]) for x in predictions_val]
    rounded_test = [round(x[0]) for x in predictions_test]

    train_accuracy_score = round(accuracy_score(y_train, rounded_train), 3)
    train_precision_score = round(precision_score(y_train, rounded_train), 3)
    train_recall_score = round(recall_score(y_train, rounded_train), 3)
    train_f1_score = round(f1_score(y_train, rounded_train), 3)
    train_confusion_matrix = confusion_matrix(y_train, rounded_train)

    print("Train Accuracy :: {0}".format(train_accuracy_score))
    print("Train Precision :: {0}".format(train_precision_score))
    print("Train Recall  :: {0}".format(train_recall_score))
    print("Train F1-score  :: {0}".format(train_f1_score))
    print("Train Confusion matrix\n", train_confusion_matrix)

    val_accuracy_score = round(accuracy_score(y_val, rounded_val), 3)
    val_precision_score = round(precision_score(y_val, rounded_val), 3)
    val_recall_score = round(recall_score(y_val, rounded_val), 3)
    val_f1_score = round(f1_score(y_val, rounded_val), 3)
    val_confusion_matrix = confusion_matrix(y_val, rounded_val)

    print('Val results:')
    print("Val Accuracy :: {0}".format(val_accuracy_score))
    print("Val Precision :: {0}".format(val_precision_score))
    print("Val Recall  :: {0}".format(val_recall_score))
    print("Val F1-score  :: {0}".format(val_f1_score))
    print("Val Confusion matrix\n", train_confusion_matrix)


    test_accuracy_score = round(accuracy_score(y_test, rounded_test), 3)
    test_precision_score = round(precision_score(y_test, rounded_test), 3)
    test_recall_score = round(recall_score(y_test, rounded_test), 3)
    test_f1_score = round(f1_score(y_test, rounded_test), 3)
    test_confusion_matrix = confusion_matrix(y_test, rounded_test)

    print('Test results:')
    print("Test Accuracy  :: {0}".format(test_accuracy_score))
    print("Test Precision :: {0}".format(test_precision_score))
    print("Test Recall  :: {0}".format(test_recall_score))
    print("Test F1-score  :: {0}".format(test_f1_score))
    print("Test Confusion matrix\n", test_confusion_matrix)



        
def main():
    
    # do not allow Tensorflow to allocate all the GPU's memory to avoid the 'out of memory' problem
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True    # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    train_path='/Data/FordA_TRAIN.arff'
    test_path='/Data/FordA_TEST.arff'

    
    sequence_length = 500  # the length of the time series
    window_length = 20  # the length (data points) of the window
    window_step = window_length  # indicates how much to slice de window over the time series
    time_steps = int(
        ((sequence_length - window_length) / window_step) + 1)  # number of windows in which the time series are divided

    batch_size = 32  # the size of the batch
    n_epochs = 150  # number of epochs in training
    n_conv_layers = 3  # number of convolutional layers
    conv_filters = [64,32,16]  # number of filters on each convolutional layer
    n_rnn_layers = 1  # number of recurrent layers
    n_rnn_units = 64  # number of units on each recurrent layer
    rnn_layer_type = 'Bi-LSTM'  # , 'Bi-LSTM', 'GRU', 'Bi-GRU',
#                       'SimpleRNN'the type of the recurrent layer (LSTM, Bi-LSTM, GRU, Bi-GRU, SimpleRNN)
    learning_rate = 10e-6  # the value of the learning rate

    df_train_val = arff.loadarff(train_path)
    df_test = arff.loadarff(test_path)
    df_tv = pd.DataFrame(df_train_val[0])
    df_test = pd.DataFrame(df_test[0])
    X_tv = df_tv.iloc[:,:500]
    y_tv =df_tv.iloc[:,500:501]
    X_test = df_test.iloc[:,:500]
    y_test =df_test.iloc[:,500:501]
    X_train,X_val,y_train,y_val = train_test_split(X_tv,y_tv,test_size=0.2, random_state=42)
    
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    
    y_train = [0 if y_train.values[i]==b'-1' else 1 for i in range(y_train.values.shape[0])]
    y_val = [0 if y_val.values[i]==b'-1' else 1 for i in range(y_val.values.shape[0])]
    y_test = [0 if y_test.values[i]==b'-1' else 1 for i in range(y_test.values.shape[0])]

    X_train = normalize(X_train)
    X_val = normalize(X_val)
    X_test = normalize(X_test)
    
    steps_per_epoch_train = math.ceil(len(y_train) / batch_size)
    steps_per_epoch_val = math.ceil(len(y_val) / batch_size)
    steps_per_epoch_test = math.ceil(len(y_test) / batch_size)
    

    
    output_directory = '/model/InceptionTime_normalized_model.h5'
    input_shape = (time_steps*window_length)
    nb_classes = 1
    nb_classes=2
    verbose=False
    build=True
    batch_size=32
    nb_filters=32
    use_residual=True
    use_bottleneck=True
    depth=6
    kernel_size=40
    nb_epochs=1000
    bottleneck_size = 32

    try:
 
        verbose = True
        build= True
        input_shape =(time_steps*window_length,)
        model = train(output_directory, input_shape, nb_classes, verbose, build, batch_size,
                 nb_filters, use_residual, use_bottleneck, depth, kernel_size, nb_epochs,X_train, X_val,X_test,
                 y_train,y_val,y_test)
        test(output_directory, input_shape, nb_classes, verbose, build, batch_size,
                 nb_filters, use_residual, use_bottleneck, depth, kernel_size, nb_epochs,X_train, X_val,X_test,
                 y_train,y_val,y_test)

#         model = Classifier_INCEPTION.buil_model(input_shape=input_shape, nb_classes=nb_classes)
        
        # free memory
        clear_session()

        for i in range(20): gc.collect()  # it is hard to do garbage collection so it is better to repe


    except Exception as error:
        print('error')
        traceback.print_exc(error)
        print(error)
        clear_session()


if __name__ == '__main__':
    main()

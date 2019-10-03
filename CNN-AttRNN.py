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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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


# custom class to early stop the learning process if the selected metric reaches a specific value
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        """ Method called at the initialization of the class (Constructor).

            Args:
                monitor (string): the name of the metric to be monitored
                value (double): value to reach by the selected metric
                verbose (int): the level of logs to be displayed (0-2)
        """
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        """ Method called at the end of every epoch. Checks whether the metric has reached the specified value.
            If it does so, it stops the training process

            Args:
                epoch (int): the name of the metric to be monitored
                logs (dictionary): logs of the training process
        """
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %03d: early stopping THR" % epoch)
            self.model.stop_training = True

def attention_3d_block(hidden_states):
    # hidden_states.shape = (batch_size, time_steps, hidden_size)
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention
    # Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(
        hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    print(score.shape)
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh',
                             name='attention_vector')(pre_activation)
    return attention_vector

def model_attention_applied_after_lstm(rnn_layer_type, n_rnn_units, return_sequences,x,input_dim,name):
    lstm_out = get_recurrent_layer(rnn_layer_type, n_rnn_units, return_sequences, x)
    attention_mul = attention_3d_block(lstm_out)
    return attention_mul


def get_recurrent_layer(rnn_layer_type, n_rnn_units, return_sequences, input_tensor):
    """ generate the specified recurrent layer type

        Args:
            rnn_layer_type (string): the type of the recurrent layer (LSTM, Bi-LSTM, GRU, Bi-GRU, SimpleRNN)
            n_rnn_units (int): number of units on each recurrent layer
            return_sequences (bool): whether to return the sequence or not
            input_tensor (tensor): input tensor of the corresponding layer

        Returns:
            keras.layers: recurrent layer of required type

    """
    if rnn_layer_type == 'LSTM':
        print('rnn input shape::',input_tensor.shape)
        rnn_layer = CuDNNLSTM(units=n_rnn_units, return_sequences=return_sequences)(input_tensor)
    elif rnn_layer_type == 'Bi-LSTM':
        rnn_layer = Bidirectional(CuDNNLSTM(units=n_rnn_units, return_sequences=return_sequences))(input_tensor)
    elif rnn_layer_type == 'GRU':
        rnn_layer = CuDNNGRU(units=n_rnn_units, return_sequences=return_sequences)(input_tensor)
    elif rnn_layer_type == 'Bi-GRU':
        rnn_layer = Bidirectional(CuDNNGRU(units=n_rnn_units, return_sequences=return_sequences))(input_tensor)
    elif rnn_layer_type == 'SimpleRNN':
        rnn_layer = SimpleRNN(units=n_rnn_units, return_sequences=return_sequences)(input_tensor)
    else:
        warnings.warn('Job aborted! Please, set a valid recurrent layer type (LSTM, Bi-LSTM, GRU, Bi-GRU, SimpleRNN)')
        raise SystemExit

    return rnn_layer



def conv1d_lstm(n_conv_layers, conv_filters, n_rnn_layers, n_rnn_units, rnn_layer_type, time_steps, window_length, max_pooling=False):
    """ generate the convolutional + LSTM architecture

        Args:
            n_conv_layers (int): number of convolutional layers
            conv_filters (int): number of filters on each convolutional layer
            n_rnn_layers (int): number of recurrent layers
            n_rnn_units (int): number of units on each recurrent layer
            rnn_layer_type (string): the type of the recurrent layer (LSTM, Bi-LSTM, GRU, Bi-GRU, SimpleRNN)
            time_steps (int): number of windows in which the time series are divided
            window_length (int): the length (data point) of the window
            max_pooling (bool): whether to add max_pooling layer or not

        Returns:
            Model: the model of the network without being compiled

    """
    
    input_layer=Input((time_steps, window_length), name='input_layer')
    print(input_layer.shape)
    x = Reshape((time_steps, window_length,1))(input_layer)

    for layer in range(n_conv_layers):
        if layer == 0:
            x = TimeDistributed(Conv1D(filters=conv_filters[layer], kernel_size=3, padding='same'),
                                name='conv_layer_{0}'.format(layer))(x)

        else:
            x = TimeDistributed(Conv1D(filters=conv_filters[layer], kernel_size=3, padding='same'),
                                name='conv_layer_{0}'.format(layer))(x)

        x = TimeDistributed(BatchNormalization(), name='bn_layer_{0}'.format(layer))(x)
        x = TimeDistributed(Activation('relu'), name='activation_layer_{0}'.format(layer))(x)
        x = TimeDistributed(Dropout(0.2))(x)

        if max_pooling:
            x = TimeDistributed(MaxPooling1D(2, padding='same'))(x)
            
    x = TimeDistributed(MaxPooling1D(2, padding='same'),name = 'conv_max_pooling')(x)

    print('1:: ',x.shape)
    flatten_layer= TimeDistributed(Flatten())(x)
    print('2:: ',flatten_layer.shape)

    return_sequences = True
    input_dim = int(flatten_layer.shape[1])
    
    x = model_attention_applied_after_lstm(rnn_layer_type, n_rnn_units, return_sequences,flatten_layer,input_dim,layer)
    
    x = Dropout(0.2)(x)

    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model




def get_cnn_lstm_model(n_conv_layers, conv_filters, n_rnn_layers, n_rnn_units, rnn_layer_type, time_steps,
                       window_length, learning_rate,
                       max_pooling=False, multi_gpu=False, n_gpu=2):
    """ select the way in which the model is generated depending on the chosen parameters

        Args:
            n_conv_layers (int): number of convolutional layers
            conv_filters (int): number of filters on each convolutional layer
            n_rnn_layers (int): number of recurrent layers
            n_rnn_units (int): number of units on each recurrent layer
            rnn_layer_type (string): the type of the recurrent layer (LSTM, Bi-LSTM, GRU, Bi-GRU)
            time_steps (int): number of windows in which the time series are divided
            window_length (int): the length (data points) of the window
            learning_rate (double): the value of the learning rate
            max_pooling (bool): whether to add max_pooling layer or not
            multi_gpu (bool): whether to train the model in multiple GPUs or not
            n_gpu (int): number of GPUs to use in the training process in case multi_gpu=True

        Returns:
            Model: compiled model

    """
    if multi_gpu:
        with tf.device('/cpu:0'):
            model = conv1d_lstm(n_conv_layers, conv_filters, n_rnn_layers, n_rnn_units, rnn_layer_type, time_steps,
                                window_length, max_pooling)
            model = multi_gpu_model(model, gpus=n_gpu)
    else:
        model = conv1d_lstm(n_conv_layers, conv_filters, n_rnn_layers, n_rnn_units, rnn_layer_type, time_steps,
                            window_length, max_pooling)

    adam = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    #plot_model(model, to_file='model.png', show_shapes=True)
    model.summary()

    return model


def main():
    
    # do not allow Tensorflow to allocate all the GPU's memory to avoid the 'out of memory' problem
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True    # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    train_path='/data/jlabaien/data/FordA/FordA_TRAIN.arff'
    test_path='/data/jlabaien/data/FordA/FordA_TEST.arff'

    
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
    learning_rate = 0.00001  # the value of the learning rate

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

    # Normalize data
    X_train = normalize(X_train,axis=0)
    X_val = normalize(X_val,axis=0)
    X_test = normalize(X_test,axis=0)
    
    # Reshape (ts,time_steps,window_length)
    X_train = X_train.reshape(X_train.shape[0],time_steps,window_length)
    X_val = X_val.reshape(X_val.shape[0],time_steps,window_length)
    X_test = X_test.reshape(X_test.shape[0],time_steps,window_length)
    
    steps_per_epoch_train = math.ceil(len(y_train) / batch_size)
    steps_per_epoch_val = math.ceil(len(y_val) / batch_size)
    steps_per_epoch_test = math.ceil(len(y_test) / batch_size)
    
    try:
        
        model = get_cnn_lstm_model(n_conv_layers, conv_filters, n_rnn_layers, n_rnn_units,
                                                       rnn_layer_type,
                                                       time_steps, window_length, learning_rate)
        #     early_stop = EarlyStoppingByLossVal(monitor='val_loss', value=0.02, verbose=1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        history = model.fit(X_train,y_train, steps_per_epoch=steps_per_epoch_train,
                                                          epochs=n_epochs, validation_data=(X_val,y_val),
                                                          validation_steps=steps_per_epoch_test, callbacks=[es])
        model.save('/data/jlabaien/PHD/FordA/model/CNN_OneAttRNN_{0}_model.h5'.format(rnn_layer_type))
        model.save_weights('/data/jlabaien/PHD/FordA/model/CNN_OneAttRNN_{0}_weights.h5'.format(rnn_layer_type))

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
        
        # free memory
        clear_session()
        del model, generator_train, generator_test,
        del predictions_train, predictions_test
        del rounded_train, rounded_test
        for i in range(20): gc.collect()  # it is hard to do garbage collection so it is better to repe


    except Exception as error:
        print('error')
        traceback.print_exc(error)
        print(error)
        clear_session()


if __name__ == '__main__':
    main()

from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

features_train = pd.read_csv("../dengue_features_train.csv")
labels_train = pd.read_csv("../dengue_labels_train.csv")
features_output_data = pd.read_csv("../dengue_features_test.csv")
features_output_format = pd.read_csv("../submission_format.csv")

kc_data = pd.DataFrame(features_train, columns=['city',
                                                 'year',
                                                'weekofyear',
                                                # 'ndvi_ne',
                                                # 'ndvi_nw',
                                                # 'ndvi_se',
                                                # 'ndvi_sw',
                                                'precipitation_amt_mm',
                                                'reanalysis_air_temp_k',
                                                'reanalysis_avg_temp_k',
                                                'reanalysis_dew_point_temp_k',
                                                'reanalysis_max_air_temp_k',
                                                'reanalysis_min_air_temp_k',
                                                'reanalysis_precip_amt_kg_per_m2',
                                                'reanalysis_relative_humidity_percent',
                                                'reanalysis_sat_precip_amt_mm',
                                                'reanalysis_specific_humidity_g_per_kg',
                                                'reanalysis_tdtr_k',
                                                'station_avg_temp_c',
                                                'station_diur_temp_rng_c',
                                                'station_max_temp_c',
                                                'station_min_temp_c',
                                                'station_precip_mm'])
kc_data['total_cases'] = pd.DataFrame(labels_train,columns=['total_cases'])
#kc_data['week_start_date'] = pd.to_numeric(features_train.week_start_date.str.slice(8, 10))

kc_data_out = pd.DataFrame(features_output_data, columns=['city',
                                                'year',
                                                'weekofyear',
                                                # 'ndvi_ne',
                                                # 'ndvi_nw',
                                                # 'ndvi_se',
                                                # 'ndvi_sw',
                                                'precipitation_amt_mm',
                                                'reanalysis_air_temp_k',
                                                'reanalysis_avg_temp_k',
                                                'reanalysis_dew_point_temp_k',
                                                'reanalysis_max_air_temp_k',
                                                'reanalysis_min_air_temp_k',
                                                'reanalysis_precip_amt_kg_per_m2',
                                                'reanalysis_relative_humidity_percent',
                                                'reanalysis_sat_precip_amt_mm',
                                                'reanalysis_specific_humidity_g_per_kg',
                                                'reanalysis_tdtr_k',
                                                'station_avg_temp_c',
                                                'station_diur_temp_rng_c',
                                                'station_max_temp_c',
                                                'station_min_temp_c',
                                                'station_precip_mm'])

# Separate data into cities

iq_data = kc_data.loc[kc_data.city == 'iq']
iq_data_out = kc_data_out.loc[kc_data_out.city == 'iq']

iq_data.drop('city', axis=1)
iq_data_out.drop('city', axis=1)
iq_data.fillna(iq_data.interpolate(), inplace=True)
iq_data_out.fillna(iq_data_out.interpolate(), inplace=True)

sj_data = kc_data.loc[kc_data.city == 'sj']
sj_data_out = kc_data_out.loc[kc_data_out.city == 'sj']
sj_data.drop('city', axis=1)
sj_data_out.drop('city', axis=1)
sj_data.fillna(sj_data.interpolate(), inplace=True)
sj_data_out.fillna(sj_data_out.interpolate(), inplace=True)

kc_data.drop('city',axis=1)
kc_data.drop('city',axis=1)

label_col = 'total_cases'
print(kc_data.describe())

def train_validate_test_split(df, train_part, validate_part, test_part, seed=None):
    np.random.seed(seed)
    total_size = train_part + validate_part + test_part
    train_percent = train_part / float(total_size)
    validate_percent = validate_part / float(total_size)
    test_percent = test_part / total_size
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = perm[:train_end]
    validate = perm[train_end:validate_end]
    test = perm[validate_end:]
    return train, validate, test

#train_size, valid_size, test_size = (70, 25, 5)
kc_train, kc_valid, kc_test = train_validate_test_split(kc_data,
                              train_part=100,
                              validate_part=0,
                              test_part=0,
                              seed=2020)

kc_y_train = kc_data.loc[kc_train, [label_col]]
kc_x_train = kc_data.loc[kc_train, :].drop(label_col, axis=1)
kc_y_valid = kc_data.loc[kc_valid, [label_col]]
kc_x_valid = kc_data.loc[kc_valid, :].drop(label_col, axis=1)
# kc_out_test = kc_data_out.drop('city', axis=1)

print('Size of training set: ', len(kc_x_train))
print('Size of validation set: ', len(kc_x_valid))
print('Size of test set: ', len(kc_test), '(not converted)')

def norm_stats(df1, df2):
    dfs = df1.append(df2)
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)

def norm_stats_single(df1):
    dfs = df1
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)

def z_score(col, stats):
    m, M, mu, s = stats
    df = pd.DataFrame()
    for c in col.columns:
        df[c] = (col[c]-mu[c])/s[c]
    return df

stats = norm_stats(kc_x_train, kc_x_valid)
stats_out = norm_stats_single(kc_data_out)
arr_x_train = np.array(z_score(kc_x_train, stats))
# arr_x_train['city'] = pd.DataFrame(features_train, columns=['city']).
arr_y_train = np.array(kc_y_train)
arr_x_valid = np.array(z_score(kc_x_valid, stats))
#arr_y_train['city'] = pd.DataFrame(features_train,columns=['city'])
arr_y_valid = np.array(kc_y_valid)

arr_output = np.array(z_score(kc_data_out,stats_out))

print('Training shape:', arr_x_train.shape)
print('Training samples: ', arr_x_train.shape[0])
print('Validation samples: ', arr_x_valid.shape[0])

def basic_model_1(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(y_size))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)

def basic_model_2(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(20, activation="relu"))
    t_model.add(Dense(y_size))
    print(t_model.summary())
    t_model.compile(loss='mean_squared_error',
        optimizer=Adam(),
        metrics=[metrics.mae])
    return(t_model)

def basic_model_3(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(x_size, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(int(x_size*2), activation="relu", kernel_initializer='normal',
        kernel_regularizer=regularizers.l1(1e-5), bias_regularizer=regularizers.l1(1e-5)))
    t_model.add(Dropout(0.3))
    t_model.add(Dense(int(x_size*.75), activation="relu", kernel_initializer='normal',
        kernel_regularizer=regularizers.l1_l2(1e-3), bias_regularizer=regularizers.l1_l2(1e-3)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(int(y_size*7), activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return(t_model)

model = basic_model_3(arr_x_train.shape[1], arr_y_train.shape[1])
model.summary()


epochs = 60
batch_size = 128

print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [
    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2)
    # ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0)
     TensorBoard(log_dir='/tmp/keras_logs/model_84', write_graph=True, write_images=True),
    EarlyStopping(monitor='mean_absolute_error', patience=50, verbose=0)
]

history = model.fit(arr_x_train, arr_y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=0, # Change it to 2, if wished to observe execution
    #validation_data=(arr_x_valid, arr_y_valid),
    callbacks=keras_callbacks)


train_score = model.evaluate(arr_x_train, arr_y_train, verbose=0)
#valid_score = model.evaluate(arr_x_valid, arr_y_valid, verbose=0)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))
# print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))

output = model.predict(arr_output)
features_output_format.drop('total_cases', axis=1)
features_output_format['total_cases'] = np.rint(output).astype(int)

features_output_format.to_csv('dengai_predictions_neural_networks_new4.csv', index=False)

def plot_hist(h, xsize=6, ysize=10):
    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.show()

    return

plot_hist(history.history, xsize=8, ysize=12)

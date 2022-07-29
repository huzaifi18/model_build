# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:19:41 2021
Capacity Baterai Prediction Using LSTM
"""
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.compat.v1.random.set_random_seed(1)

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from math import sqrt
import os
from sklearn.model_selection import KFold
# import shutil
import json
import re

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import *

from model import *
# from model_t2v_orig import *
import params_multi as pr

def preprocess(dataset):
    scalers = {}
    for i in range(dataset.shape[1]):
      scalers[i] = MinMaxScaler(feature_range=(0, 1))
      dataset[:,i,:] = scalers[i].fit_transform(dataset[:,i,:])
    return dataset, scalers

def extract_VIT_capacity(x_datasets, y_datasets, seq_len, hop, sample):
    x = [] # VITC = inputs voltage, current, temperature (in vector) + capacity (in scalar)
    y = [] # target capacity (in scalar)
    z = [] # cycle index
    SS = [] # scaler
    VITC = [] # temporary input

    for x_data, y_data in zip(x_datasets, y_datasets):
        # Load VIT from charging profile
        x_df = read_csv(x_data).dropna()
        x_df = x_df[['cycle','voltage_battery','current_battery','temp_battery']]
        x_df['cycle'] = x_df['cycle']+1
        x_len = len(x_df.cycle.unique()) #- seq_len

        # Load capacity from discharging profile
        y_df = read_csv(y_data).dropna()
        y_df['cycle_idx'] = y_df.index+1
        y_df = y_df[['capacity', 'cycle_idx']]
        y_df = y_df.values # Convert pandas dataframe to numpy array
        y_df = y_df.astype('float32') # Convert values to float
        y_len = len(y_df) #- seq_len

        data_len = np.int32(np.floor((y_len - seq_len-1)/hop)) + 1
        for i in range(y_len):
            cy = x_df.cycle.unique()[i]
            df = x_df.loc[x_df['cycle']==cy]
            # Voltage measured
            le = len(df['voltage_battery']) % sample
            vTemp = df['voltage_battery'].to_numpy()
            if le != 0:
                vTemp = vTemp[0:-le]
            vTemp = np.reshape(vTemp, (len(vTemp)//sample,-1)) #, order="F")
            vTemp = vTemp.mean(axis=0)
            # Current measured
            iTemp = df['current_battery'].to_numpy()
            if le != 0:
                iTemp = iTemp[0:-le]
            iTemp = np.reshape(iTemp, (len(iTemp)//sample,-1)) #, order="F")
            iTemp = iTemp.mean(axis=0)
            # Temperature measured
            tTemp = df['temp_battery'].to_numpy()
            if le != 0:
                tTemp = tTemp[0:-le]
            tTemp = np.reshape(tTemp, (len(tTemp)//sample,-1)) #, order="F")
            tTemp = tTemp.mean(axis=0)
            # Capacity measured
            cap = np.array([y_df[i, 0]])
            # Combined
            VITC_inp = np.concatenate((vTemp, iTemp, tTemp, cap))
            VITC.append(VITC_inp)

        # Normalize using MinMaxScaler
        df_VITC = DataFrame(VITC).values
        scaled_x, scaler = preprocess(df_VITC[:, :, np.newaxis])
        scaled_x = scaled_x.astype('float32')[:, :, 0] # Convert values to float
        ceritanya_x = scaled_x[(hop*1):(hop*1+seq_len), :]
        ceritanya_y = scaled_x[hop*1+seq_len,-1]

        # Create input data
        for i in range(data_len):
            x.append(scaled_x[(hop*i):(hop*i+seq_len), :])
            y.append(scaled_x[hop*i+seq_len, -1])
            # z.append(y_df[hop*i+seq_len, 1])
        SS.append(scaler)
        # import pdb; pdb.set_trace()
    return np.array(x), np.array(y)[:, np.newaxis], SS

def plot_loss(trainPredict, trainY, history, model_name, save_dir, model_dir):
    # Plot model loss
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, model_dir, 'train_loss.png'))

def plot_pred(predict, true, history, model_name, save_dir, model_dir, name):    # Plot test prediction
    predict = predict.reshape(predict.shape[0])
    true = true.reshape(true.shape[0])
    fig=plt.figure(figsize=(12, 4),dpi=150)
    plt.plot(predict,label='Prediction')
    plt.plot(true,label='True')
    plt.xlabel('Number of Cycle',fontsize=13)
    plt.ylabel('DIscharge Capacity (Ah)',fontsize=13)
    # plt.title(str(model_name.__name__)+' Prediction of Discharge Capacity of Test Data (B0005)',fontsize=14)
    plt.legend(loc='upper right',fontsize=12)
    plt.savefig(os.path.join(save_dir, model_dir, name+'.png'))

def main():
    # Load dataset
    pth = pr.pth
    train_x_files = [os.path.join(pth,'charge/train',f) for f in os.listdir(os.path.join(pth, 'charge/train'))]
    train_x_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    train_y_files = [os.path.join(pth,'discharge/train',f) for f in os.listdir(os.path.join(pth, 'discharge/train'))]
    train_y_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    test_x_data = [os.path.join(pth,'charge/test',f) for f in os.listdir(os.path.join(pth, 'charge/test'))]
    test_y_data = [os.path.join(pth,'discharge/test',f) for f in os.listdir(os.path.join(pth, 'discharge/test'))]
    print("train X:", train_x_files)
    print("train Y:", train_y_files)

    # k-fold iteration
    folds = list(KFold(n_splits=pr.k, shuffle=True, random_state=pr.random).split(train_x_files))
    val_loss = []
    test_loss = []
    for j, (train_idx, val_idx) in enumerate(folds):
        # Split train_files into train and val
        print('\nFold', j+1)
        train_x_data = [train_x_files[train_idx[i]] for i in range(len(train_idx))]
        train_y_data = [train_y_files[train_idx[i]] for i in range(len(train_idx))]
        val_x_data = [train_x_files[val_idx[i]] for i in range(len(val_idx))]
        val_y_data = [train_y_files[val_idx[i]] for i in range(len(val_idx))]
        print("train:", train_x_data, train_y_data)
        print("val:", val_x_data, val_y_data)
        print("test:", test_x_data,  test_y_data)
        # import pdb; pdb.set_trace()

        # Create dataset for train, val, and test
        trainX, trainY, SS_tr = extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len, pr.hop, pr.sample)
        valX, valY, SS_val = extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len, pr.hop, pr.sample)
        testX, testY, SS_tt = extract_VIT_capacity(test_x_data, test_y_data, pr.seq_len, pr.hop, pr.sample)
        print('Input shape: {}'.format(trainX.shape))
        print('Input shape: {}'.format(trainY.shape))
        # import pdb; pdb.set_trace()

        # Setup model # All models are defined in model.py
        model_name = pr.model
        print('Model:', str(model_name.__name__))
        if model_name == GatedConv1D:
            model = model_name(input_shape=(pr.seq_len, trainX.shape[-1]), nb_filter=pr.nb_filter, 
                                nb_kernel=pr.nb_kernel, nb_stride=pr.nb_stride, dropout_rate=pr.dropout_rate)
        elif model_name == Transformer:
            model = Transformer(input_shape=(pr.seq_len, trainX.shape[-1]),
                                num_hid=trainX.shape[-1], #feat_dim=31
                                time_steps=pr.seq_len,
                                time_embedding=pr.time_embed,
                                num_head=pr.num_head,
                                num_layers_enc=pr.num_block,
                                ff_dim=pr.ff_dim,
                                dropout=pr.dropout_enc)
        else:
            model = model_name(input_shape=(pr.seq_len, trainX.shape[-1]), hidden_dim=pr.hidden_dim)

        # Train and save best model
        save_dir = pr.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_dir = pr.model_dir+'_k'+str(j+1)
        if not os.path.exists(os.path.join(save_dir,model_dir)):
            os.makedirs(os.path.join(save_dir, model_dir))
        # checkpoint = ModelCheckpoint(os.path.join(save_dir,model_dir,str(model_name.__name__)+'.h5'), 
							# monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min', baseline=None, restore_best_weights=False)
        optim = Adam(learning_rate=pr.lr)
        model.compile(loss='mse', optimizer=optim)
        model.build(input_shape=(None, pr.seq_len, trainX.shape[-1]))
        model.summary()
        # history = model.fit(trainX, trainY, validation_data=(valX, valY),
        #     batch_size=pr.batch_size, epochs=pr.epochs, callbacks=[checkpoint], shuffle=True)
        history = model.fit(trainX, trainY, validation_data=(valX, valY),
            batch_size=pr.batch_size, epochs=pr.epochs, shuffle=True)
        # shutil.copy('params.py', os.path.join(save_dir, model_dir))
        model.save(pr.save_dir+"/"+model_dir+"/"+"saved_model_and_weights")
        # model.save_weights(pr.save_dir + "/"+ model_dir+"/"+"weights") # Menyimpan bobot
        print("Bobot dan model tersimpan")
        # Evaluate on validation and test data
        val_results = model.evaluate(valX, valY)
        val_loss.append(val_results)
        print('Val loss:', val_results)

        results = model.evaluate(testX, testY)
        test_loss.append(results)
        print('Test loss:', results)

        # Predictions
        trainPredict = model.predict(trainX)
        valPredict = model.predict(valX)
        testPredict = model.predict(testX)
        # import pdb; pdb.set_trace()

        # Inverse transform MinMaxScaler / denormalize
        inv_valY = SS_val[0][30].inverse_transform(valY)
        inv_valPredict = SS_val[0][30].inverse_transform(valPredict)

        inv_testY = SS_tt[0][30].inverse_transform(testY)
        inv_testPredict = SS_tt[0][30].inverse_transform(testPredict)
        # import pdb; pdb.set_trace()

        # Get evaluation metrics on test data: MAE, MSE, MAPE, RMSE (full cycles)
        test_mae = mean_absolute_error(inv_testY, inv_testPredict)
        test_mse = mean_squared_error(inv_testY, inv_testPredict)
        test_mape = mean_absolute_percentage_error(inv_testY, inv_testPredict)
        test_rmse = np.sqrt(mean_squared_error(inv_testY, inv_testPredict))
        print('\nTest Mean Absolute Error: %f MAE' % test_mae)
        print('Test Mean Square Error: %f MSE' % test_mse)
        print('Test Mean Absolute Percentage Error: %f MAPE' %test_mape)
        print('Test Root Mean Squared Error: %f RMSE' % test_rmse)
        # import pdb; pdb.set_trace()

        # Save evaluation metrics
        with open(os.path.join(save_dir, model_dir, 'eval_metrics.txt'), 'w') as f:
            f.write('Train data: ')
            f.write(json.dumps(train_x_data))
            f.write('\nVal data: ')
            f.write(json.dumps(val_x_data))
            f.write('\nTest data: ')
            f.write(json.dumps(test_x_data))
            f.write('\n\nTest Mean Absolute Error: ')
            f.write(json.dumps(str(test_mae)))
            f.write('\nTest Mean Square Error: ')
            f.write(json.dumps(str(test_mse)))
            f.write('\nTest Mean Absolute Percentage Error: ')
            f.write(json.dumps(str(test_mape)))
            f.write('\nTest Root Mean Squared Error: ')
            f.write(json.dumps(str(test_rmse)))

        # Save test prediction to text file
        testPred_file = open(os.path.join(save_dir, model_dir, 'test_predict.txt'), 'w')
        for row in inv_testPredict:
            np.savetxt(testPred_file, row)
        testPred_file.close()

        testY_file = open(os.path.join(save_dir, model_dir, 'test_true.txt'), 'w')
        for row in inv_testY:
            np.savetxt(testY_file, row)
        testY_file.close()

        # Plot graph
        plot_loss(trainPredict, trainY, history, model_name, save_dir, model_dir)
        plot_pred(inv_valPredict, inv_valY, history, model_name, save_dir, model_dir, name='val_pred')
        plot_pred(inv_testPredict, inv_testY, history, model_name, save_dir, model_dir, name='test_pred')

        # import pdb; pdb.set_trace() # comment this out to run k-fold iteration or type "continue" to run the next iteration

    # Save list of k-fold test loss to text file
    with open(os.path.join(save_dir, model_dir, 'test_loss.txt'), 'w') as f:
        f.write(json.dumps(test_loss))
    with open(os.path.join(save_dir, model_dir, 'val_loss.txt'), 'w') as f:
        f.write(json.dumps(val_loss))

if __name__ == "__main__":
    main()

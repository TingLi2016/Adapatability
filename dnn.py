import time
start_time = time.time()


###Loading packages
import warnings
warnings.filterwarnings('ignore')

import os
from os import listdir
from os.path import isfile, join
import itertools
import math
import keras
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,balanced_accuracy_score

from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from keras import optimizers
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(6)

he_normal = initializers.he_normal()

# define base model
def create_model(n_dim, node, activation, optimizer):

    # create model
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(node, input_dim = n_dim, kernel_initializer=he_normal, activation=activation, kernel_regularizer=l2(para[4]), activity_regularizer=l2(para[5])))#
    NN_model.add(BatchNormalization())

    # The Output Layer :
    NN_model.add(Dense(1, activation='sigmoid'))

    # Compile model
    NN_model.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return NN_model

###fit model
def fit_model(X_train, y_train, X_validation, y_validation, n, model_path, model, batch_size):
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train), y_train)
    ###fit model
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=n, batch_size=batch_size, class_weight=class_weights, shuffle=False)#
    model.save(model_path)
    return model

def model_predict(X, y, model, col_name):
    y_pred = model.predict(X)
    y_pred_class = np.where(y_pred > 0.5, 1, 0)
    pred_result = pd.DataFrame()
    pred_result['id'] = y.index
    pred_result['y_true'] = y.values
    pred_result['prob_'+col_name] = y_pred
    pred_result['class_'+col_name] = y_pred_class

    result=measurements(y, y_pred_class, y_pred)
    return pred_result, result

def measurements(y_test, y_pred, y_pred_prob):
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob) 
    sensitivity = metrics.recall_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    specificity = TN/(TN+FP)
    npv = TN/(TN+FN)
    return [TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc, balanced_accuracy]
    
def dim_reduce(df, test_df, path1, col_name1):
    
    X = df.iloc[:, 3:]
    print(X.shape)
    y = df.loc[:, 'y_true']
    X_test = test_df.iloc[:, 3:]
    print(X_test.shape)
    y_test = test_df.loc[:, 'y_true']

    sc = StandardScaler()
    #sc = MinMaxScaler()
    sc.fit(X)
    X = sc.transform(X)
    X_test = sc.transform(X_test)    
    

    ###load the best model
    best_model = load_model(path1 + '/' +col_name1 + '_weights.h5')
    
    ### predict test set
    #print("after training: ", X)
    test_class, test_result = model_predict(X_test, y_test, best_model, col_name1)
    #print('test_result:', test_result)

    train_class, train_result= model_predict(X, y, best_model, col_name1)
    #print('train_result:', train_result)

    K.clear_session()
    tf.reset_default_graph() 
    return test_class, test_result, train_class, train_result 


def sep_performance(df):
    cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC', 'Balanced_accuracy']
    for i, col in enumerate(cols):
        if i == 0:
            df[col] = df.value.str.split(',').str[i].str.split('[').str[1].values
        elif i == len(cols)-1:
            df[col] = df.value.str.split(',').str[i].str.split(']').str[0].values
        else:
            df[col] = df.value.str.split(',').str[i].values

    for i, col in enumerate(cols):
        if i < 4:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
            df[col] = round(df[col], 3)
    del df['value']
            
    return df

def reform_result(results):
    df = pd.DataFrame(data=results.items())
    ###reform the result data format into single colum
    df = df.rename(columns={0:'name', 1:'value'})
    df['name'] = df['name'].astype('str')
    df['value'] = df['value'].astype('str')
    df = sep_performance(df)
    return df

def dili_prediction(data_path, var, result_path, model_path):
    ###data
    data = pd.read_csv(data_path+'/validation/validation_probabilities_' + var + '.csv')
    test = pd.read_csv(data_path+'/test/test_probabilities_' + var + '.csv')

    path2 = result_path + '/validation_class'
    path3 = result_path + '/validation_performance'
    path4 = result_path + '/test_class'
    path5 = result_path + '/test_performance'


    #initial performance dictionary
    test_results={}
    train_results={}

    ###get the prediction
    test_class, test_result, train_class, train_result  = dim_reduce(data, test, model_path, var)

    test_results[var]=test_result
    test_class.to_csv(path4+'/test_'+var+'.csv')


    train_results[var]=train_result
    train_class.to_csv(path2+'/validation_'+var+'.csv')

    reform_result(test_results).to_csv(path5+'/test_'+var+'.csv')
    reform_result(train_results).to_csv(path3+'/validation_'+var+'.csv')

    print("--- %s seconds ---" % (time.time() - start_time))

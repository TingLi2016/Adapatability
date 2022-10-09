import pandas as pd
import numpy as np

import os
from os import listdir
from os.path import isfile, join
import itertools

from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score



def sep_performance(df):
    df = df.rename(columns={'0':'name', '1':'value'})
    print(df.columns)
    cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'PPV', 'F1', 'MCC']
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
    df['seed'] = df['name'].str.split('seed_').str[1].str.split('_paras').str[0]
    df['model'] = df['name'].str.split('_').str[0]
    del df['value']
    del df['Unnamed: 0']
            
    return df

def read_files(basepath, method, var):
    path = basepath + '/' + method + '/' + method + '_' + var + '/validation_performance'
    file = listdir(path)
    df = pd.read_csv(join(path, file[0]))
    df = sep_performance(df)
    return df


def selectedBase(basepath, var, mccpath):
    
    knn = read_files(basepath, 'knn', var)
    lr = read_files(basepath, 'lr', var)
    svm = read_files(basepath, 'svm', var)
    rf = read_files(basepath, 'rf', var)
    xgboost = read_files(basepath, 'xgboost', var)


    data = pd.concat([knn, lr, svm, rf, xgboost], axis = 0)
    mcc = data[(data.MCC >= np.percentile(data.MCC.values, 5)) & (data.MCC <= np.percentile(data.MCC.values, 95))]

    mcc.to_csv(mccpath + '/mcc_' + var + '.csv')
    return mcc


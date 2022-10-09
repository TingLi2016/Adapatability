#!/account/tli/anaconda3/bin/python

import sys
var=sys.argv[1]
test_year1 = sys.argv[2]
test_year2 = sys.argv[3]

add_year1 = sys.argv[4]
add_year2 = sys.argv[5]

add_year3 = sys.argv[6]
add_year4 = sys.argv[7]


import time
start_time = time.time()

### Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

### import scripts
import base_knn
import base_lr
import base_svm
import base_rf
import base_xgboost

import get_mcc

import validation_predictions_combine
import test_predictions_combine

import dnn

### please update the following path 

current_path = '/account/tli/DeepDILIsupport/script/github3'

base_path = current_path + '/output'
mcc_path = current_path + '/output/mcc'
probability_path = current_path + '/output/data'
result_path = current_path + '/output/result'
model_path = current_path + '/weights'

### read data
data = pd.read_csv(current_path + '/DILI_dataset.csv',low_memory=False)


### run the scripts
data_split = pd.read_csv(current_path + '/data_split' + '/' + var +'.csv')
base_knn.generate_baseClassifiers(data, var, test_year1, test_year2, add_year1, add_year2, add_year3, add_year4, base_path + '/knn/knn_' + var, data_split)
base_lr.generate_baseClassifiers(data, var, test_year1, test_year2, add_year1, add_year2, add_year3, add_year4, base_path + '/lr/lr_' + var, data_split)
base_svm.generate_baseClassifiers(data, var, test_year1, test_year2, add_year1, add_year2, add_year3, add_year4, base_path + '/svm/svm_' + var, data_split)
base_rf.generate_baseClassifiers(data, var, test_year1, test_year2, add_year1, add_year2, add_year3, add_year4, base_path + '/rf/rf_' + var, data_split)
base_xgboost.generate_baseClassifiers(data, var, test_year1, test_year2, add_year1, add_year2, add_year3, add_year4, base_path + '/xgboost/xgboost_' + var, data_split)

mcc = get_mcc.selectedBase(base_path, var, mcc_path)

validation_predictions_combine.combine_probabilities(base_path, mcc, probability_path+'/validation', var)
test_predictions_combine.combine_probabilities(base_path, mcc, probability_path+'/test', var)

dnn.dili_prediction(probability_path, var, result_path, model_path)



print("--- %s seconds ---" % (time.time() - start_time))


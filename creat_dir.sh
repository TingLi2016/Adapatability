#!/bin/bash

echo "[start]"
echo `date`

###please update the current_path to your directory

current_path='/account/tli/DeepDILIsupport/script/github3'



echo "make base classifiers directory"
base_path0=$current_path/output
mkdir -p $base_path0
mkdir -p $base_path0/logs
mkdir -p $base_path0/knn
mkdir -p $base_path0/lr
mkdir -p $base_path0/svm
mkdir -p $base_path0/rf
mkdir -p $base_path0/xgboost

echo "make directory for selected base classifiers"
mkdir -p $base_path0/mcc

echo "make probability directory"
mkdir -p $base_path0/data/test
mkdir -p $base_path0/data/validation

echo "make directory for final results"
base_path=$base_path0/result
mkdir -p $base_path
mkdir -p $base_path/validation_performance
mkdir -p $base_path/test_performance
mkdir -p $base_path/validation_class
mkdir -p $base_path/test_class


echo "[end]"
echo `date`

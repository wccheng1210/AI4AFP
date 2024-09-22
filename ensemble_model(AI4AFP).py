#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
from PC6_encoding import get_PC6_features_labels
from doc2vec import get_Doc2Vec_features_labels
from bert_encoding import get_bert_features_labels

# Load data
pos_train_data = './seq_data/afp_pos_seq_len50ensembletrain_2108.fasta'
neg_train_data = './seq_data/afp_neg_seq_len50ensembletrain_2108.fasta'
pos_valid_data = './seq_data/afp_pos_seq_len50ensemblevalid_301.fasta'
neg_valid_data = './seq_data/afp_neg_seq_len50ensemblevalid_301.fasta'
pos_test_data = './seq_data/afp_pos_seq_len50independant_test_602.fasta'
neg_test_data = './seq_data/afp_neg_seq_len50independant_test_602.fasta'

# Encoding through pc6 pretrained 
pc6_train_features, pc6_train_labels = get_PC6_features_labels(pos_train_data, neg_train_data,length=50)
reshape_pc6_train_features = pc6_train_features.reshape(pc6_train_features.shape[0],-1)

pc6_valid_features, pc6_valid_labels = get_PC6_features_labels(pos_valid_data, neg_valid_data,length=50)
reshape_pc6_valid_features = pc6_valid_features.reshape(pc6_valid_features.shape[0],-1)

pc6_test_features, pc6_test_labels = get_PC6_features_labels(pos_test_data, neg_test_data,length=50)
reshape_pc6_test_features = pc6_test_features.reshape(pc6_test_features.shape[0],-1)

# Encoding through Doc2Vec pretrained
doc2vec_train_features, doc2vec_train_labels = get_Doc2Vec_features_labels(pos_train_data, neg_train_data, './Doc2Vec_model/AFP_doc2vec.model')
reshape_doc2vec_train_features=doc2vec_train_features.reshape((doc2vec_train_features.shape[0],doc2vec_train_features.shape[1],1))

doc2vec_valid_features, doc2vec_valid_labels = get_Doc2Vec_features_labels(pos_valid_data, neg_valid_data, './Doc2Vec_model/AFP_doc2vec.model')
reshape_doc2vec_valid_features=doc2vec_valid_features.reshape((doc2vec_valid_features.shape[0],doc2vec_valid_features.shape[1]))

doc2vec_test_features, doc2vec_test_labels = get_Doc2Vec_features_labels(pos_test_data, neg_test_data, './Doc2Vec_model/AFP_doc2vec.model')
reshape_doc2vec_test_features=doc2vec_test_features.reshape((doc2vec_test_features.shape[0],doc2vec_test_features.shape[1]))

# Encoding through prot_bert_bfd pretrained
bert_train_features, bert_train_labels = get_bert_features_labels(pos_train_data, neg_train_data, 'Rostlab/prot_bert_bfd', MAX_LEN=50)
bert_valid_features, bert_valid_labels = get_bert_features_labels(pos_valid_data, neg_valid_data, 'Rostlab/prot_bert_bfd', MAX_LEN=50)
bert_test_features, bert_test_labels = get_bert_features_labels(pos_test_data, neg_test_data, 'Rostlab/prot_bert_bfd', MAX_LEN=50)

# Labels to nunpy format
train_labels = np.array(pc6_train_labels)
valid_labels = np.array(pc6_valid_labels)
test_labels = np.array(pc6_test_labels)

from sklearn import svm
from sklearn import ensemble
import joblib
from model import train_pc6_model
from model import train_doc2vec_model
from bert_prediction import get_bert_prediction, train_bert_model, run_bert_prediction
from model_tools import learning_curve, evalution_metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Create directory
if not os.path.isdir('ensemble_model/AI4AFP'):
    os.makedirs('ensemble_model/AI4AFP/pc6')
    os.makedirs('ensemble_model/AI4AFP/doc2vec')
    os.makedirs('ensemble_model/AI4AFP/bert')

# PC6 model
# SVM
pc6_svc = svm.SVC()
pc6_svc_fit = pc6_svc.fit(reshape_pc6_train_features, pc6_train_labels)
joblib.dump(pc6_svc, './ensemble_model/AI4AFP/pc6/pc6_features_svm.pkl')

pc6_svc = joblib.load('./ensemble_model/AI4AFP/pc6/pc6_features_svm.pkl')
pc6_svm_labels_score = pc6_svc.predict(reshape_pc6_valid_features)
pc6_svm_res = evalution_metrics(pc6_valid_labels, pc6_svm_labels_score)

# RF
pc6_forest = ensemble.RandomForestClassifier(n_estimators = 100)
pc6_forest_fit = pc6_forest.fit(reshape_pc6_train_features, pc6_train_labels)
joblib.dump(pc6_forest, './ensemble_model/AI4AFP/pc6/pc6_features_forest.pkl')

pc6_forest = joblib.load('./ensemble_model/AI4AFP/pc6/pc6_features_forest.pkl')
pc6_rf_labels_score = pc6_forest.predict(reshape_pc6_valid_features)
pc6_rf_res = evalution_metrics(pc6_valid_labels, pc6_rf_labels_score)

# CNN
pc6_train_data_, pc6_test_data_, pc6_train_labels_, pc6_test_labels_ = train_test_split(pc6_train_features, pc6_train_labels, test_size= 0.1, random_state = 1, stratify = pc6_train_labels)
pc6_t_m = train_pc6_model(pc6_train_data_, pc6_train_labels_, pc6_test_data_, pc6_test_labels_, 'ensemble', path = './ensemble_model/AI4AFP/pc6')
learning_curve(pc6_t_m.history)

pc6_cnn_model = load_model('./ensemble_model/AI4AFP/pc6/ensemble_best_weights.h5')
pc6_cnn_labels_score = pc6_cnn_model.predict(pc6_valid_features)
pc6_cnn_res = evalution_metrics(pc6_valid_labels, pc6_cnn_labels_score)

# Doc2vec model
# SVM
doc2vec_svc = svm.SVC()
doc2vec_svc_fit = doc2vec_svc.fit(doc2vec_train_features, doc2vec_train_labels)
joblib.dump(doc2vec_svc, './ensemble_model/AI4AFP/doc2vec/doc2vec_features_svm.pkl')

doc2vec_svc = joblib.load('./ensemble_model/AI4AFP/doc2vec/doc2vec_features_svm.pkl')
doc2vec_svm_labels_score = doc2vec_svc.predict(doc2vec_valid_features)
doc2vec_svm_res = evalution_metrics(doc2vec_valid_labels, doc2vec_svm_labels_score)

# RF
doc2vec_forest = ensemble.RandomForestClassifier(n_estimators = 100)
doc2vec_forest_fit = doc2vec_forest.fit(doc2vec_train_features, doc2vec_train_labels)
joblib.dump(doc2vec_forest, './ensemble_model/AI4AFP/doc2vec/doc2vec_features_forest.pkl')

doc2vec_forest = joblib.load('./ensemble_model/AI4AFP/doc2vec/doc2vec_features_forest.pkl')
doc2vec_rf_labels_score = doc2vec_forest.predict(doc2vec_valid_features)
doc2vec_rf_res = evalution_metrics(doc2vec_valid_labels, doc2vec_rf_labels_score)

# CNN
doc2vec_train_data_, doc2vec_test_data_, doc2vec_train_labels_, doc2vec_test_labels_ = train_test_split(reshape_doc2vec_train_features, doc2vec_train_labels, test_size= 0.1, random_state = 1, stratify = pc6_train_labels)
doc2vec_t_m = train_doc2vec_model(doc2vec_train_data_, doc2vec_train_labels_, doc2vec_test_data_, doc2vec_test_labels_, 'ensemble', path = './ensemble_model/AI4AFP/doc2vec')
learning_curve(doc2vec_t_m.history)

doc2vec_cnn_model = load_model('./ensemble_model/AI4AFP/doc2vec/ensemble_best_weights.h5')
doc2vec_cnn_labels_score = doc2vec_cnn_model.predict(reshape_doc2vec_valid_features)
doc2vec_cnn_res = evalution_metrics(doc2vec_valid_labels, doc2vec_cnn_labels_score)

# BERT model
bert_labels = get_bert_prediction(pos_train_data, neg_train_data, pos_valid_data, neg_valid_data, 'Rostlab/prot_bert_bfd', './ensemble_model/AI4AFP/bert', MAX_LEN=50)
bert_labels_score = run_bert_prediction(pos_valid_data, neg_valid_data, 'Rostlab/prot_bert_bfd', './ensemble_model/AI4AFP/bert', MAX_LEN=50)
bert_labels_score = np.array(bert_labels_score)
bert_res = evalution_metrics(bert_valid_labels, bert_labels_score)

# Find CNN model threshold
from model_tools import evalution_metrics, findThresIndex
from sklearn import metrics
(pc6_fpr, pc6_tpr, pc6_thresholds) = metrics.roc_curve(pc6_valid_labels, pc6_cnn_labels_score)
(doc2vec_fpr, doc2vec_tpr, doc2vec_thresholds) = metrics.roc_curve(doc2vec_valid_labels, doc2vec_cnn_labels_score)

pc6_thresidx = findThresIndex(pc6_tpr, pc6_fpr)
pc6_thres = pc6_thresholds[pc6_thresidx]

doc2vec_thresidx = findThresIndex(doc2vec_tpr, doc2vec_fpr)
doc2vec_thres = doc2vec_thresholds[doc2vec_thresidx]
print('PC6_CNN_thres=' + str(pc6_thres))
print('Doc2vec_CNN_thres=' +  str(doc2vec_thres))

# Generate ensemble model input
from model import train_ensemble_model
valid_size = len(pc6_valid_labels)

ensembleX_list = []
ensembleY_list = []
for i in range(valid_size):
    score_list = []
    score_list.append(float(pc6_rf_labels_score[i]))
    score_list.append(float(pc6_svm_labels_score[i]))
    if(pc6_cnn_labels_score[i][0] >= pc6_thres):
        score_list.append(float('1'))
    else:
        score_list.append(float('0'))
    
    score_list.append(float(doc2vec_rf_labels_score[i]))
    score_list.append(float(doc2vec_svm_labels_score[i]))
    if(doc2vec_cnn_labels_score[i][0] >= doc2vec_thres):
        score_list.append(float('1'))
    else:
        score_list.append(float('0'))
    score_list.append(float(bert_labels_score[i]))
    ensembleX_list.append(score_list)
ensembleX = np.array(ensembleX_list)
ensembleY = np.array(pc6_valid_labels)

e_m = train_ensemble_model(ensembleX, ensembleY, 'ensemble', path = './ensemble_model/AI4AFP')

# Get independent data size
in_size = len(pc6_test_labels)

pc6_rf_test_score = pc6_forest.predict(reshape_pc6_test_features)
pc6_svm_test_score = pc6_svc.predict(reshape_pc6_test_features)
pc6_cnn_test_score = pc6_cnn_model.predict(pc6_test_features)

doc2vec_rf_test_score = doc2vec_forest.predict(doc2vec_test_features)
doc2vec_svm_test_score = doc2vec_svc.predict(doc2vec_test_features)
doc2vec_cnn_test_score = doc2vec_cnn_model.predict(reshape_doc2vec_test_features)

bert_test_score = run_bert_prediction(pos_test_data, neg_test_data, 'Rostlab/prot_bert_bfd', './ensemble_model/AI4AFP/bert', MAX_LEN=50)

in_ensembleX_list = []
in_ensembleY_list = []
for i in range(in_size):
    score_list = []
    score_list.append(float(pc6_rf_test_score[i]))
    score_list.append(float(pc6_svm_test_score[i]))
    if(pc6_cnn_test_score[i][0] >= pc6_thres):
        score_list.append(float('1'))
    else:
        score_list.append(float('0'))
    
    score_list.append(float(doc2vec_rf_test_score[i]))
    score_list.append(float(doc2vec_svm_test_score[i]))
    if(doc2vec_cnn_test_score[i][0] >= doc2vec_thres):
        score_list.append(float('1'))
    else:
        score_list.append(float('0'))
    score_list.append(float(bert_test_score[i]))
    in_ensembleX_list.append(score_list)
in_ensembleX = np.array(in_ensembleX_list)
in_ensembleY = np.array(pc6_test_labels)

ensemble_model = load_model('./ensemble_model/AI4AFP/ensemble_best_weights.h5')
in_score = ensemble_model.predict(in_ensembleX)
final_score = evalution_metrics(pc6_test_labels, in_score)
print("Final score:")
print(final_score)
print("PC6 RF score:")
print(pc6_rf_res)
print("PC6 SVM score:")
print(pc6_svm_res)
print("PC6 CNN score:")
print(pc6_cnn_res)
print("Doc2vec RF score:")
print(doc2vec_rf_res)
print("Doc2vec SVM score:")
print(doc2vec_svm_res)
print("Doc2vec CNN score:")
print(doc2vec_cnn_res)
print("BERT score:")
print(bert_res)
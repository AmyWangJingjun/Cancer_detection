#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 01:44:18 2017

@author: Iris
"""

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
import pickle
from sklearn.decomposition import TruncatedSVD as tsvd

sms = pd.read_csv('training_text.txt', encoding= 'UTF8', 
                    sep="\\|\\|",header=None, engine='python', 
                    names=['cli_data'])
sms3 = pd.read_table('test__mod_text.txt', encoding= 'UTF8', sep="\\|\\|",header=None, engine='python', names=['cli_data'])
sms2= pd.read_csv('training_variants.txt', encoding= 'UTF8', 
                    header=None,delimiter=',', engine='python', 
                    names=['gene','mutation','classification'])

sms4= pd.read_csv('test_variants.csv', encoding= 'UTF8', 
                    header=None,delimiter=',', engine='python', 
                    names=['gene','mutation','classification'])
#X1 = sms.cli_data
#y1= sms2.classification
X_train = sms.cli_data
y_train = sms2.classification
X_test = sms3.cli_data
y_test = sms4.classification
RAD_NUM = [12]
ACCURACY_NB = dict()
log_loss_NB = dict()
ACCURACY_LOG = dict()
log_loss_LOG = dict()
ACCURACY_SVC = dict()
log_loss_SVC = dict()
ACCURACY_LSVC = dict()
log_loss_LSVC = dict()
for i in RAD_NUM:
    print("now running random_num{}".format(i))
#    X = X1[pd.notnull(X1)]
#    y = y1[pd.notnull(y1)]
#    X_train, X_test1, y_train, y_test = train_test_split(X, y, random_state=i)
    print(type(X_train))
    weighted=y_train.value_counts()
    weight_dict=dict()
    for key in weighted.keys():
        weight_dict[key]=1/ (weighted[key]/3316)
    vect = CountVectorizer(analyzer = 'word', stop_words = 'english', encoding= 'UTF8',
                           decode_error = 'ignore', ngram_range = (1, 3), min_df =0.00005)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    TRUN_DIMS = [70]
    for m in TRUN_DIMS:
        print("now truncating {}".format(m))
        svd = tsvd(n_components = m)
        X_train_reduced= svd.fit_transform(X_train_dtm)
        X_test_reduced = svd.transform(X_test_dtm)
        C_NUM = [ 0.5]
        for c in C_NUM:
            TOL = [0.00001]
            for t in TOL:
                #for naive bayes model
                nb = MultinomialNB()
                nb.fit(X_train_dtm, y_train)
                y_pred_class_nb = nb.predict(X_test_dtm)
                y_pred_prob_nb = nb.predict_proba(X_test_dtm)
                accuracy_nb = metrics.accuracy_score(y_test, y_pred_class_nb)
                log_loss_nb = metrics.log_loss(y_test, y_pred_prob_nb)
                NB_NAME = "nb_inst_ran{}_dims{}_C{}_tol{}.pickle".format(i, m, c, t)
                key_nb=  "nb_inst_ran{}_dims{}_C{}_tol{}".format(i, m, c, t)
                print("acurracy for nb is {}".format(accuracy_nb))
                print("log loss for nb is {}".format(log_loss_nb))
                ACCURACY_NB[key_nb] = accuracy_nb
                log_loss_NB[key_nb] = log_loss_nb
#               for logistic regression model
                print("now running random_num{} truncating {} C_num{} tol_num{}".format(i, m, c, t))
                logreg = LogisticRegression(C = c, tol = t, class_weight=weight_dict)
                logreg.fit(X_train_reduced, y_train)
                LOG_NAME = "log_inst_ran{}_dims{}_C{}_tol{}.pickle".format(i, m, c, t)
                key_log =  "log_inst_ran{}_dims{}_C{}_tol{}".format(i, m, c, t)
                y_pred_prob_log = logreg.predict_proba(X_test_reduced)
                log_loss_log = metrics.log_loss(y_test, y_pred_prob_log)
                y_pred_class_log = logreg.predict(X_test_reduced)
                accuracy_log = metrics.accuracy_score(y_test, y_pred_class_log)
                log_loss_LOG[key_log] = log_loss_log
                ACCURACY_LOG[key_log] = accuracy_log
                print("acurracy for log is {}".format(accuracy_log))
                print("log loss for log is {}".format(log_loss_log))
                #SVCmodel
                svc=SVC(C = c, tol = t, class_weight=weight_dict, probability=True)   
                svc.fit(X_train_reduced,y_train)
                y_pred_class_svc = svc.predict(X_test_reduced)
                y_pred_prob_svc = svc.predict_proba(X_test_reduced)
                log_loss_svc = metrics.log_loss(y_test, y_pred_prob_svc)
                accuracy_svc = metrics.accuracy_score(y_test, y_pred_class_svc)
                SVC_NAME = "svc_inst_ran{}_dims{}_C{}_tol{}.pickle".format(i, m, c, t)
                key_svc = "svc_inst_ran{}_dims{}_C{}_tol{}".format(i, m, c, t)
                ACCURACY_SVC[key_svc] = accuracy_svc
                log_loss_SVC[key_svc] = log_loss_svc
                confusion_m = metrics.confusion_matrix(y_test, y_pred_class_svc)
                print('\n')
                print(confusion_m)
                print('\n')
                print("acurracy for svc is {}".format(accuracy_svc))
                print("log_loss for svc is {}".format(log_loss_svc))               
  #linearSVC model
 

                output = pd.DataFrame( data = y_pred_class_svc )
                output.columns = ['classification']
                output['ID'] = range(1,369)
                output = output[['ID','classification']]
                output.to_csv(r'result2.txt', header=None, index=None, sep=' ', mode='a')
                

import cPickle
import gzip
import os
import sys
import copy


import numpy
from numpy import genfromtxt
import numpy as np
import csv
import gzip,cPickle

import theano
import theano.tensor as T

def load_ag_news():
    total_len = 231
    csvFile_test = "/home/maochengzhi/workspacetestData/ag_news_csv/test.csv"
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

    test_y = []
    raw_test_x = []
    test_x = []    
    with open(csvFile_test,'rb') as cf:
        spamreader = csv.reader(cf,delimiter = ',')
        for row in spamreader:
            #print ','.join(row)
            test_y.append(ord(row[0]) - ord('0'))
            oneline = ','.join(row[1:])        
            line_index = []
            for each in oneline:
                if each.lower() in alphabet:
                    line_index.append(alphabet.index(each.lower())+1)   #start from 1
                else:
                    line_index.append(0)
            length = len(line_index)
            sentence=[]
            sentence = np.asarray(sentence)
            if length<=total_len:
                for each in line_index:
                    temp = np.zeros([70],dtype='float32')
                    if each>0:
                        temp[each-1]=1                
                    sentence = np.concatenate([sentence,temp],axis=0)
                sentence = np.concatenate([sentence,np.zeros([(total_len - length)*70],dtype='float32')],axis=0)
            else:
                for each in line_index[0:total_len]:
                    temp = np.zeros([70],dtype='float32')
                    if each>0:
                        temp[each-1]=1
                    sentence = np.concatenate([sentence,temp],axis=0)

            test_x.append(sentence);

    test_y = np.asarray(test_y)
    test_set=(test_x,test_y)

    csvFile_train = "./../data/ag_news_csv/train.csv"
    train_y = []
    raw_train_x = []
    train_x = []
    cnt=0
    with open(csvFile_train,'rb') as cf:
        spamreader = csv.reader(cf,delimiter = ',')
        for row in spamreader:
            #print ','.join(row)
            cnt = cnt + 1
            if cnt % 10000 == 0:
                print cnt,'finished'
                

            train_y.append(ord(row[0]) - ord('0'))
            oneline = ','.join(row[1:])        
            line_index = []
            for each in oneline:
                if each.lower() in alphabet:
                    line_index.append(alphabet.index(each.lower())+1)   #start from 1
                else:
                    line_index.append(0)
            length = len(line_index)
            sentence=[]
            sentence = np.asarray(sentence)
            if length<=total_len:
                for each in line_index:
                    temp = np.zeros([70],dtype='float32')
                    if each>0:
                        temp[each-1]=1                
                    sentence = np.concatenate([sentence,temp],axis=0)
                sentence = np.concatenate([sentence,np.zeros([(total_len - length)*70],dtype='float32')],axis=0)
            else:
                for each in line_index[0:total_len]:
                    temp = np.zeros([70],dtype='float32')
                    if each>0:
                        temp[each-1]=1
                    sentence = np.concatenate([sentence,temp],axis=0)

            train_x.append(sentence);

    train_y = np.asarray(train_y)

    train_set=(train_x,train_y)
    
    rval=[train_set,test_set]
    '''
    print 'start saving'
    f=file('ag_news_data_sim.txt','wb')
    cPickle.dump(rval,f)
    f.close()
    print 'saved'
    '''
    return rval

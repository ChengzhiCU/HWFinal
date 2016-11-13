import numpy as np
import csv
import gzip,cPickle

from numpy import genfromtxt


alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

test_y = []
raw_test_x = []
test_x = []

total_len = 1014
cnt = 0
csvFile_test = "./ag_news_csv/test.csv"
with open(csvFile_test,'rb') as cf:
    spamreader = csv.reader(cf,delimiter = ',')
    for row in spamreader:
        #print ','.join(row)
        cnt = cnt + 1
        if cnt%500 ==0:
            print 'test finished',cnt,'sentences'
        if cnt==10:
            f=file('try.txt','wb')
            cPickle.dump(test_x,f)
            f.close()
            print 'save '
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
        #sentence = np.array(sentence)
        if length<=total_len:
            for each in line_index:
                #temp = np.zeros([70],dtype='float32')
                temp = [0]*70
                if each>0:
                    temp[each-1]=1                
                #sentence = np.concatenate([sentence,temp],axis=0)
                sentence = sentence+temp
            #sentence = np.concatenate([sentence,np.zeros([(255 - length)*70],dtype='float32')],axis=0)
            #sentence = sentence + np.zeros([(255 - length)*70],dtype='float32')
            l=(total_len - length)*70
            sentence = sentence + [0]*l
        else:
            for each in line_index[0:total_len]:
                #temp = np.zeros([70],dtype='float32')
                temp =[0]*70
                if each>0:
                    temp[each-1]=1
                #sentence = np.concatenate([sentence,temp],axis=0)
                sentence = sentence + temp

        #test_x = np.concatenate([test_x,sentence],axis=0);
        #print sentence
        test_x = test_x + sentence
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)
#print test_x
print type(test_x)
test=(test_x,test_y)

csvFile_train = "./ag_news_csv/train.csv"
train_y = []
raw_train_x = []
train_x = []
cnt = 0
with open(csvFile_train,'rb') as cf:
    spamreader = csv.reader(cf,delimiter = ',')
    for row in spamreader:
        #print ','.join(row)
        if cnt%500 == 0:
            print 'train',cnt,'sentences'
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
        if length<=225:
            for each in line_index:
                temp = np.zeros([70],dtype='float32')
                if each>0:
                    temp[each-1]=1                
                sentence = np.concatenate([sentence,temp],axis=0)
            sentence = np.concatenate([sentence,np.zeros([(255 - length)*70],dtype='float32')],axis=0)
        else:
            for each in line_index[0:255]:
                temp = np.zeros([70],dtype='float32')
                if each>0:
                    temp[each-1]=1
                sentence = np.concatenate([sentence,temp],axis=0)

        #train_x.append(sentence)
        train_x = np.concatenate([train_x, sentence],asix = 0)

train_y = np.asarray(train_y)
train_set=(train_x,train_y)

f=file('ag_news_data.txt','wb')
cPickle.dump((train,test),f)
f.close()
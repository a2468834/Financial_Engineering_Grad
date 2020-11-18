#   : Convolutional Autoencoder - Module
#    
#   Date:        2020/11/18
#   CourseID:    10910QF 510300
#   Course:      Special Topics on Financial Engineering (Graduated)
#   
#   Writer_ID:   109062631
#   Writer_Name: Wang, Chuan-Chun
#   Environment: 
#      Software: Python 3.8.5 on 64-bit Windows 10 Pro (2004)
#      Hardware: Intel i7-10510U, 16GB DDR4 non-ECC ram, and no discrete GPU
import math
import matplotlib.pyplot as pyplot
import numpy
import os
import pandas
import pathlib
import pickle
import random
import re
from   scipy.special import softmax
import struct
import sys
import time
import torch
import torch
import torch.nn
from   torch import optim
import warnings
from   imblearn.over_sampling import RandomOverSampler


########## Classes ##########
class CONST:
    # Define some constants
    row_pixel = lambda : 26
    col_pixel = lambda : 26
    drop_col  = lambda : ['id', 'member_id', 'url', 'desc']


class FUNC:
    # Lambda functions
    tokenize = lambda text : re.findall(r'\w+', text.lower())


def readCSV(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        input_df = pandas.read_csv(fp)
    return input_df.rename(columns={'Unnamed: 0': 'index'})


def compColDescription():
    with open('DataDictionary.xlsx', 'rb') as fp1, open('./Training/X_train.csv', 'r') as fp2:
        input_df = pandas.read_excel(fp1, skiprows=[0], sheet_name='LoanStats', header=None)
        xlsx_col = [FUNC.tokenize(i)[0] for i in list(input_df[0])] # Discard whitespace at the end of string
        data_col = FUNC.tokenize(fp2.readline())
    
    for i in xlsx_col:
        if i not in data_col:
            print(i)
        else:
            pass


def saveDataFrameToPK():
    train = {}
    train['X'] = readCSV('./Training/X_train.csv')
    train['Y'] = readCSV('./Training/Y_train.csv')
    with open('./Training/train.pk', 'wb') as f:
        pickle.dump(train, f)
    
    del train
    
    test = {}
    test['X'] = readCSV('./Testing/X_test.csv')
    test['Y'] = readCSV('./Testing/Y_test.csv')
    with open('./Testing/test.pk', 'wb') as f:
        pickle.dump(test, f)


def saveToTXT(pandas_df, file_path):
    pandas_df.to_csv(file_path, header=None, index=None, sep=' ', mode='w')
    #numpy.savetxt("out.csv", ndarray, delimiter=",")


def dropDFColumn(pandas_df):
    return pandas_df.drop(CONST.drop_col, axis=1)


# Please be careful of the shape to 'data_X' and 'data_Y', both object can be Dataframe or numpy-ndarray.
# data_X.shape = (n_samples, n_features)
# data_Y.shape = (n_samples,)
# E.G., train_OS['X'], train_OS['Y'] = overSampling(train['X'], train['Y'].drop(['index'], axis=1))
def overSampling(data_X, data_Y):
    ros = RandomOverSampler(random_state=0) # Use naive oversampling
    return ros.fit_resample(data_X, data_Y)


# Main function
if __name__ == "__main__":
    #saveDataFrameToPK()
    
    with open('./Training/train.pk', 'rb') as f:
        train = pickle.load(f) # train = {'X': X_train.csv DataFrame, 'Y': Y_train.csv DataFrame}
    
    '''
    with open('./Training/train_over_sample.pk', 'rb') as f:
        train_OS = pickle.load(f)
    
    with open('./Testing/test.pk', 'rb') as f:
        test = pickle.load(f) # test = {'X': X_test.csv DataFrame, 'Y': Y_test.csv DataFrame}
    '''


# Other codes
#print(train_data['X'].info(memory_usage='deep', verbose=True))
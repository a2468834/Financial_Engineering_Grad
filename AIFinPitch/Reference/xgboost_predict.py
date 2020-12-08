#  Date:         2020/11/18
#  CourseID:     10910QF 510300
#  Course:       Special Topics on Financial Engineering (Graduated)
#
#  Writer_ID:    109062631
#  Writer_Name:  Wang, Chuan-Chun
#  Environment:
#    [Configuration 1]
#      SW:  Python 3.8.5 on 64-bit Windows 10 Pro (2004)
#      HW:  Intel i7-10510U, 16GB DDR4 non-ECC ram, and no discrete GPU
#    [Configuration 2]
#      SW:  Python 3.8.5 on Ubuntu 20.04.1 LTS (Linux 5.4.0-54-generic x86_64)
#      HW:  AMD Ryzen 5 3400G, 64GB DDR4 non-ECC ram, and no discrete GPU
import itertools
import math
import matplotlib.pyplot as pyplot
from   multiprocessing import Pool
import numpy
import os
import pandas
import pickle
import random
from   sklearn.metrics import f1_score
from   sklearn.model_selection import ParameterGrid, train_test_split
from   sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import time
import xgboost
import zlib


########## Classes ##########
class CONST:
    # Define some constants
    vt_ratio   = lambda : 0.1 # validation.size / training.size ratio, i.e., 10-fold cross-validation
    batch_size = lambda : 8192
    epoch_num  = lambda : 100
    lr_rate    = lambda : 0.5


########## Functions ##########


########## Functions ##########
if __name__ == "__main__":
    # Display pandas DataFrame and numpy ndarray without truncation
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    numpy.set_printoptions(threshold=sys.maxsize)
    
    # Read .pk files
    # 'train' is a dict {'X': train['X'].csv DataFrame after dropping cols and rows, 'Y': Y_train.csv DF after...}
    # 'test' is a dict {'X': X_test.csv DataFrame} (do not have 'Y')
    try:
        with open('./PKs/test.pk', 'rb') as f1, open('./PKs/train.pk', 'rb') as f2:
            print("Reading test.pk")
            test = pickle.load(f1)
            train = pickle.load(f2)
    except:
        print("Please run 'save_pk.py' first (to generate train.pk and test.pk)")
        exit()
    
    
    
    print(list(train['X'].columns))
    print()
    print(list(test['X'].columns))
    exit()
    
    # Un-finished !!!
    
    print("Predict testing data")
    test_DM = xgboost.DMatrix(test['X'])
    
    model = pickle.load(open(sys.argv[1], "rb"))
    predict = model.predict(test_DM)
    
    print(predict)
    


########## Other codes ##########
#logging.info(train_data['X'].info(memory_usage='deep', verbose=True))
#pool.map(saveHist, [(train['X'][col_name], './img/'+str(col_name)+'.png') for col_name in train['X'].columns])
'''
for i in train['X'].columns:
    pyplot.hist(train['X'][col_name])
    pyplot.gcf().savefig('./img/'+str(col_name)+'.png')
    pyplot.clf() # Clear the current figure and axes
'''
#group_list = [list(combined_df.columns)[i*2: (i+1)*2] for i in range(int(len(combined_df.columns)/2)+1)] 
'''
logging.info(len(list(combined_df.columns)))
for col_name in combined_df.columns:
    if col_name == 'loan_status':
        pass
    else:
        combined_df.boxplot(column=[col_name], by='loan_status')
        pyplot.savefig(col_name+'.png')
        pyplot.close('all')
exit()
'''
#import logging
'''
def saveHist(pack_tuple):
    ndarray, file_name = pack_tuple
    pyplot.hist(ndarray)
    pyplot.gcf().savefig(file_name)
    pyplot.clf() # Clear the current figure and axes
'''
'''
# Read .pk files
# 'train' is a dict {'X': train['X'].csv DataFrame after dropping cols and rows, 'Y': Y_train.csv DF after...}
# 'test' is a dict {'X': X_test.csv DataFrame} (do not have 'Y')
try:
    with open('./PKs/test.pk', 'rb') as f1, open('./PKs/train.pk', 'rb') as f2:
        print("Reading test.pk")
        test = pickle.load(f1)
        train = pickle.load(f2)
except:
    print("Please run 'save_pk.py' first (to generate train.pk and test.pk)")
    exit()
'''
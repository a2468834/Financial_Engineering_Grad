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
    vt_ratio   = lambda : 0.2 # validation.size / training.size ratio, i.e., 10-fold cross-validation
    batch_size = lambda : 8192
    epoch_num  = lambda : 100
    lr_rate    = lambda : 0.5


########## Functions ##########
def xgb_f1_score(predict, target, threshold=0.5):
    target = target.get_label()
    predict_bin = (predict > threshold).astype(int)
    return 'f1', f1_score(target, predict_bin)


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
        with open('./PKs/train.pk', 'rb') as f:
            print("Reading train.pk")
            train = pickle.load(f)
    except:
        print("Please run 'save_pk.py' first (to generate train.pk and test.pk)")
        exit()
    
    
    # Split whole training dataset into training part and validation part
    train_train, train_valid = {}, {}
    train_train['X'], train_valid['X'], train_train['Y'], train_valid['Y'] = train_test_split(train['X'], train['Y'], test_size=CONST.vt_ratio())
        
    # Convert ndarray to DMatrix
    train_DM  = xgboost.DMatrix(train_train['X'], label=train_train['Y'])
    valid_DM  = xgboost.DMatrix(train_valid['X'], label=train_valid['Y'])
    watchlist = [(train_DM, 'train'),(valid_DM, 'val')]
    
    # Find optimal parameters of training models
    param_grid = {'gamma':          [round(i, 4) for i in list(numpy.arange(0, 10+1, 2))],
                  'm_child_weight': [round(i, 4) for i in list(numpy.arange(0, 10+1, 2))],
                  'lambda':         [round(i, 4) for i in list(numpy.arange(0, 10+1, 2))],
                  'alpha':          [round(i, 4) for i in list(numpy.arange(0, 10+1, 2))]}
    param_grid = list(ParameterGrid(param_grid))
    
    print("Strat training")
    for i in param_grid:
        start = time.time()
        param_instance = {'gpu_id':           0,
                          'objective':        'binary:logistic',
                          'tree_method':      'gpu_hist',
                          'max_depth':        25,
                          'eta':              0.3,
                          'gamma':            i['gamma'],
                          'min_child_weight': i['m_child_weight'],
                          'lambda':           i['lambda'],
                          'alpha':            i['alpha']}
        print(param_instance)
        model = xgboost.train(param_instance, train_DM, num_boost_round=20, feval=xgb_f1_score, evals=watchlist)
        
        # Save sgboost model
        file_name = str(int(time.time())) + '_model.pk'
        pickle.dump(model, open(file_name, "wb"))
        print("Model have been saved as {}.".format(file_name))
        print("Time: {:.2f}\n\n\n".format(time.time()-start))


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
#'gpu_id':int(sys.argv[1])

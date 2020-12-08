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
from   sklearn.model_selection import ParameterGrid, train_test_split
from   sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from   tensorboardX import SummaryWriter
import time
import torch
import torch.nn
import torch.nn.functional
from   torch import optim
import zlib

from torchvision import datasets, transforms

########## Classes ##########
class CONST:
    # Define some constants
    vt_ratio      = lambda : 0.2 # validation.size / training.size ratio, i.e., 10-fold cross-validation
    batch_size    = lambda : 8192
    epoch_num     = lambda : 25
    lr_rate       = lambda : 1e-4
    dp_ratio      = lambda : 0.4
    id_col        = lambda : ['id', 'member_id']
    redundant_col = lambda : ['annual_inc_joint',             'collection_recovery_fee', 
                              'debt_settlement_flag_date',    'desc', 
                              'earliest_cr_line',             'emp_title', 
                              'fico_range_high',              'hardship_end_date', 
                              'hardship_length',              'hardship_reason', 
                              'hardship_start_date',          'issue_d', 
                              'last_credit_pull_d',           'last_pymnt_d', 
                              'num_sats',                     'payment_plan_start_date', 
                              'sec_app_earliest_cr_line',     'settlement_date', 
                              'tot_hi_cred_lim',              'total_il_high_credit_limit', 
                              'url',                          'zip_code']
    mean_fill     = lambda : ['bc_open_to_buy',               'dti', 
                              'mths_since_recent_bc_dlq',     'mths_since_recent_revol_delinq', 
                              'pct_tl_nvr_dlq',               'sec_app_mths_since_last_major_derog']
    median_fill   = lambda : ['all_util',                     'avg_cur_bal', 
                              'bc_util',                      'dti_joint', 
                              'hardship_last_payment_amount', 'hardship_payoff_balance_amount', 
                              'il_util',                      'inq_fi', 
                              'inq_last_12m',                 'max_bal_bc', 
                              'mo_sin_old_il_acct',           'mths_since_last_delinq', 
                              'mths_since_last_major_derog',  'mths_since_last_record', 
                              'mths_since_rcnt_il',           'mths_since_recent_bc', 
                              'mths_since_recent_inq',        'num_rev_accts', 
                              'open_act_il',                  'open_il_24m', 
                              'open_rv_12m',                  'open_rv_24m', 
                              'orig_projected_additional_accrued_interest', 
                              'percent_bc_gt_75',             'revol_bal_joint', 
                              'sec_app_fico_range_high',      'sec_app_fico_range_low', 
                              'sec_app_num_rev_accts',        'sec_app_open_acc', 
                              'sec_app_open_act_il',          'sec_app_revol_util', 
                              'settlement_amount',            'settlement_percentage', 
                              'settlement_term',              'total_bal_il']
    zlibCompress   = lambda x : zlib.compress(pickle.dumps(x), level=9) # Use highest level of compression
    zlibDecompress = lambda x : pickle.loads(zlib.decompress(x))
    percent2f      = lambda x : (float(x.strip('%')) + 0.0) * 1e-2 # Adding 0.0 is used to prevent negative zero
    month2i        = lambda x : int(x.strip(' months'))
    category2i     = lambda x, ct_dict: ct_dict[x] if x in ct_dict else 0
    haveCUDA      = lambda : 'cuda' if torch.cuda.is_available() else 'cpu'


class DROP:
    col_dict = {}
    row_dict = {}


class FCModel(torch.nn.Module):
    def __init__(self, input_dim, dp_ratio):
        super(FCModel, self).__init__()
        self.fc1 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=input_dim, out_features=1024),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=dp_ratio)
                   )
        self.fc2 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=1024, out_features=1024),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=dp_ratio)
                   )
        self.fc3 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=1024, out_features=1024),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=dp_ratio)
                   )
        self.fc4 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=1024, out_features=512),
                       torch.nn.ReLU()
                   )
        self.fc5 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=512, out_features=128),
                       torch.nn.ReLU()
                   )
        self.fc6 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=128, out_features=32),
                       torch.nn.ReLU()
                   )
        self.fc7 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=32, out_features=2),
                       torch.nn.Softmax(dim=1),
                       
                   )
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)[:, 0]
        return x


########## Functions ##########
def readNPY():
    train, test = {}, {}
    train['X'] = numpy.load('./PKs/X_train.npy')
    train['Y'] = numpy.load('./PKs/Y_train.npy')
    test['X'] = numpy.load('./PKs/X_test.npy')
    test['Y'] = numpy.load('./PKs/Y_test.npy')
    
    mean = numpy.mean(train['X'][:,-109:],axis=0)#對直線取mean
    max_ = numpy.max(train['X'][:,-109:],axis=0)
    
    train['X'][:,-109:] /= max_
    test['X'][:,-109:] /= max_
    
    train['X'] = train['X'].astype("float32")

    Nindex = numpy.where(train['Y']==0)
    Yindex = numpy.where(train['Y']==1)
    Nindex = list(Nindex[0])
    Yindex = list(Yindex[0])
    
    sample_index = random.sample(Nindex,298637)
    sample_index.extend(Yindex)
    random.shuffle(sample_index)

    train['X'] = train['X'][sample_index,:]
    train['Y'] = train['Y'][sample_index]
    return train, test


def haveCUDA():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def binPrediction(predict, threshold=0.5):
    predict_bin = (predict > threshold).float() * 1
    return (2.0 * precision * recall) / (precision + recall + 1e-8)


def readCSV(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        input_df = pandas.read_csv(fp)
    return input_df.rename(columns={'Unnamed: 0': 'upload_index'})


########## Main function ##########
if __name__ == "__main__":
    # Display pandas DataFrame and numpy ndarray without truncation
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    numpy.set_printoptions(threshold=sys.maxsize)
    
    '''
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
    '''
    
    # Read not-by-me .npy files
    train, test = readNPY()
    del train
    
    test_tensor = {}
    test_tensor['X'] = torch.Tensor(test['X'])
    test_tensor['Y'] = torch.Tensor(test['Y'])
    
    fc_model = torch.load(sys.argv[1]).to('cpu')
    fc_model.eval()
    predict = fc_model(test_tensor['X'])
    predict = predict.detach().numpy()
    predict = numpy.where(predict>0.5, 'Y', 'N')
    #predict = predict[:,0]
    
    test['Y'] = readCSV('Y_test.csv')
    test['Y']['loan_status'] = predict
    test['Y'].to_csv('final.csv',index=0)
    

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
#      COLumns whose dtype is the string
drop.col_dict['str'] = [col_name for col_name in train['X'].columns if train['X'][col_name].dtype == 'object']

#      COLumns that contain more than 10% nan values
drop.col_dict['almost_nan'] = []
for col_name in train['X'].columns:
    if train['X'][col_name].isnull().sum()/len(train['X'][col_name]) > 0.1:
        drop.col_dict['almost_nan'].append(col_name)
drop.col_dict['almost_nan'].sort()
'''

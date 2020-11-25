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
from   imblearn.over_sampling import RandomOverSampler
import itertools
import math
import matplotlib.pyplot as pyplot
from   multiprocessing import Pool
import numpy
import pandas
import pickle
import random
import re
import time
import torch
import torch.nn
from   torch import optim


########## Classes ##########
class CONST:
    # Define some constants
    row_pixel  = lambda : 26
    col_pixel  = lambda : 26
    tv_ratio   = lambda : 0.8 # training validation ratio
    batch_size = lambda : 256
    id_col     = lambda : ['id', 'member_id']
    # columns with imbalance values
    imb_col    = lambda : ['collection_recovery_fee', 'collections_12_mths_ex_med', 'delinq_2yrs', 
                           'delinq_amnt', 'dti', 'hardship_dpd', 'hardship_last_payment_amount', 
                           'max_bal_bc', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_rcnt_il', 
                           'mths_since_recent_bc', 'num_accts_ever_120_pd', 'num_tl_30dpd', 
                           'num_tl_90g_dpd_24m', 'num_tl_120dpd_2m', 'open_il_12m', 'open_il_24m', 
                           'orig_projected_additional_accrued_interest', 'policy_code', 'pub_rec', 
                           'pub_rec_bankruptcies', 'recoveries', 'revol_bal', 'revol_bal_joint', 
                           'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 
                           'tax_liens', 'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 
                           'total_bal_ex_mort', 'total_bal_il', 'total_bc_limit', 'total_cu_tl', 
                           'total_il_high_credit_limit', 'total_rec_late_fee', 'total_rev_hi_lim', 
                           'acc_now_delinq', 'annual_inc', 'annual_inc_joint', 'avg_cur_bal', 
                           'chargeoff_within_12_mths']
    # columns with nan rows more than 10%
    nan_col    = lambda : ['mths_since_last_delinq', 'mths_since_last_record', 
                           'mths_since_last_major_derog', 'annual_inc_joint', 'dti_joint', 
                           'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 
                           'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 
                           'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 
                           'inq_last_12m', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 
                           'mths_since_recent_revol_delinq', 'revol_bal_joint', 
                           'sec_app_fico_range_low', 'sec_app_fico_range_high', 
                           'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 
                           'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts', 
                           'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 
                           'sec_app_mths_since_last_major_derog', 'deferral_term', 'hardship_amount', 
                           'hardship_dpd', 'orig_projected_additional_accrued_interest', 
                           'hardship_payoff_balance_amount', 'hardship_last_payment_amount', 
                           'settlement_amount', 'settlement_percentage', 'settlement_term']


class FUNC:
    # Lambda functions
    tokenize = lambda text : re.findall(r'\w+', text.lower())


class DROP:
    col_dict = {}
    row_dict = {}


class FCModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(FCModel, self).__init__()
        self.input = torch.nn.Sequential(
                            torch.nn.Linear(in_features=input_dim, out_features=100),
                            torch.nn.ReLU()
                     )
        self.hidden1 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=100, out_features=100),
                           torch.nn.ReLU()
                       )
        self.hidden2 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=100, out_features=100),
                           torch.nn.ReLU()
                       )
        self.hidden3 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=100, out_features=100),
                           torch.nn.ReLU()
                       )
        self.hidden4 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=100, out_features=100),
                           torch.nn.ReLU()
                       )
        self.hidden5 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=100, out_features=100),
                           torch.nn.ReLU()
                       )
        self.output = torch.nn.Sequential(
                          torch.nn.Linear(in_features=100, out_features=1),
                          torch.nn.Sigmoid()
                      )
        
    def forward(self, x):
        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.output(x)
        return x


########## Functions ##########
def readCSV(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        input_df = pandas.read_csv(fp)
    return input_df.rename(columns={'Unnamed: 0': 'upload_index'})


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


def saveToTXT(pandas_df, file_path):
    pandas_df.to_csv(file_path, header=None, index=None, sep=' ', mode='w')
    #numpy.savetxt("out.csv", ndarray, delimiter=",")


# Caution! High Memory Consumption!
# Please be careful of the shape to 'data_X' and 'data_Y', both object can be Dataframe or numpy-ndarray.
# data_X.shape = (n_samples, n_features)
# data_Y.shape = (n_samples,)
# E.G., train_OS['X'], train_OS['Y'] = overSampling(train['X'], train['Y'].drop(['index'], axis=1))
def overSamplingToPK(data_X, data_Y):
    ros = RandomOverSampler(random_state=0) # Use naive oversampling
    
    with open('./Training/train_over_sample.pk', 'wb') as f:
        pickle.dump(ros.fit_resample(data_X, data_Y), f)


def saveDataFrameToPK():
    pool = Pool()
    drop = DROP()
    
    # Read raw csv data into two pandas DataFrames
    train = {}
    train['X'], train['Y'] = readCSV('./Training/X_train.csv'), readCSV('./Training/Y_train.csv')
    
    # Find all ROWs that contain only NaN
    drop.row_dict['all_nan'] = pool.map(checkAllNaNRow, itertools.product(train['X'].index, [train['X'].drop(CONST.id_col()+['upload_index'], axis=1)]))
    drop.row_dict['all_nan'] = list(set(drop.row_dict['all_nan']))
    drop.row_dict['all_nan'].sort()
    drop.row_dict['all_nan'].pop(0) # Pop out the 1st element '-1'
    
    # Find all COLumns whose dtype is 'object' (i.e., string)
    drop.col_dict['str'] = [col_name for col_name in train['X'].columns if train['X'][col_name].dtype == 'object']
    
    # Find all COLumns that contain more than 10% nan values
    drop.col_dict['almost_nan'] = []
    for col_name in train['X'].columns:
        if train['X'][col_name].isnull().sum()/len(train['X'][col_name]) > 0.1:
            drop.col_dict['almost_nan'].append(col_name)
    drop.col_dict['almost_nan'].sort()
    
    # COLumns contain only id (excluded 'upload_index')
    drop.col_dict['id'] = CONST.id_col()
    
    # Drop rows that found at previous steps
    for key in drop.row_dict.keys():
        train['X'] = train['X'].drop(drop.row_dict[key], errors='ignore')
        train['Y'] = train['Y'].drop(drop.row_dict[key], errors='ignore')
    
    # Drop columns that found at previous steps
    for key in drop.col_dict.keys():
        train['X'] = train['X'].drop(drop.col_dict[key], axis=1, errors='ignore')
    
    # Fill nan value with mean of that columns
    train['X'] = train['X'].fillna(train['X'].mean())
    
    # Store Dataframe into .pk file
    with open('./Training/train.pk', 'wb') as f:
        pickle.dump(train, f)
    
    pool.close()
    exit()


def checkAllNaNRow(pack_tuple):
    row_index, pandas_df = pack_tuple
    if pandas_df.iloc[row_index].isnull().all() == True:
        return row_index
    else:
        return -1


def saveHist(pack_tuple):
    ndarray, file_name = pack_tuple
    pyplot.hist(ndarray)
    pyplot.gcf().savefig(file_name)
    pyplot.clf() # Clear the current figure and axes


# return training_part, validation_part
def splitDataset(total_dataset):
    division_index = int(CONST.tv_ratio()*len(total_dataset['X'].index))
    
    train_part = {'X' : total_dataset['X'][:division_index], 'Y' : total_dataset['Y'][:division_index]}
    valid_part = {'X' : total_dataset['X'][division_index:], 'Y' : total_dataset['Y'][division_index:]}
    return train_part, valid_part


# Main function
if __name__ == "__main__":
    # Display pandas DataFrame without truncation
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    
    # Generate .pk files
    #saveDataFrameToPK() # Would automatically exit() after saving DataFrame to .pk
    
    # Read .pk file
    # 'train' is a dict {'X': X_train.csv DataFrame after dropping cols and rows, 'Y': Y_train.csv DF after...}
    with open('./Training/train.pk', 'rb') as f:
        train = pickle.load(f)
    
    train_part, valid_part = splitDataset(train)
    
    train_part['Y'] = torch.tensor(train_part['Y']['loan_status'].values.astype(numpy.float64))
    train = torch.tensor(train.drop('Target', axis = 1).values.astype(np.float32)) 
    train_tensor = torch.utils.data.TensorDataset(train, train_target) 
    train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)
    
    
    model = FCModel(input_dim=len(train['X'].columns))
    
    
    
    #numpy.savetxt('out.txt', train['X']['inq_last_6mths'].values)
    
    
    '''
    for i in train['X'].columns:
        pyplot.hist(train['X'][col_name])
        pyplot.gcf().savefig('./img/'+str(col_name)+'.png')
        pyplot.clf() # Clear the current figure and axes
    '''
    #pool.close()
    '''
    with open('./Training/train_over_sample.pk', 'rb') as f:
        train_OS = pickle.load(f)
    
    with open('./Testing/test.pk', 'rb') as f:
        test = pickle.load(f) # test = {'X': X_test.csv DataFrame, 'Y': Y_test.csv DataFrame}
    '''


# Other codes
#print(train_data['X'].info(memory_usage='deep', verbose=True))
#pool.map(saveHist, [(train['X'][col_name], './img/'+str(col_name)+'.png') for col_name in train['X'].columns])

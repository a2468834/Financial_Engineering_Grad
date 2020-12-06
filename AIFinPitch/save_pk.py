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
from   sklearn.preprocessing import LabelEncoder
import time
import zlib


########## Classes ##########
class CONST:
    # Define some constants
    tv_ratio      = lambda : 0.8 # training validation ratio
    batch_size    = lambda : 8192
    epoch_num     = lambda : 100
    lr_rate       = lambda : 0.5
    dp_ratio      = lambda : 0.3
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
    percent2f      = lambda x : (float(str(x).strip('%')) + 0.0) * 1e-2 # Adding 0.0 is used to prevent negative zero
    month2i        = lambda x : int(str(x).strip(' months')) if str(x) != 'nan' else 0
    category2i     = lambda x, ct_dict: ct_dict[x] if x in ct_dict else 0


class DROP:
    col_dict = {}
    row_dict = {}


########## Functions ##########
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


def readCSV(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        input_df = pandas.read_csv(fp)
    return input_df.rename(columns={'Unnamed: 0': 'upload_index'})


def processTrain(train):
    pool, drop = Pool(), DROP()
    
    # FIND columns and rows with specific attributes
    # (1) Rows contain only NaN
    all_nan_rows = pool.map(checkAllNaNRow, itertools.product(train['X'].index, [train['X'].drop(CONST.id_col()+['upload_index'], axis=1)]))
    all_nan_rows = list(set(all_nan_rows)) # Remove duplicated rows
    all_nan_rows.remove(-1) # Pop out the 1st element '-1'
    drop.row_dict['all_nan'] = all_nan_rows
    
    
    # (2) Columns contain only id
    drop.col_dict['id'] = CONST.id_col() + ['upload_index']
    
    # (3) Columns that are redundant or contain too many attributes
    drop.col_dict['redundant'] = CONST.redundant_col()
    
    # (4) 'title' column
    drop.col_dict['title'] = ['title']
    
    
    # DROP columns and rows
    # (1) Rows
    for key in drop.row_dict.keys():
        train['X'] = train['X'].drop(drop.row_dict[key], errors='ignore')
        train['Y'] = train['Y'].drop(drop.row_dict[key], errors='ignore')
    
    # (2) Columns
    for key in drop.col_dict.keys():
        train['X'] = train['X'].drop(drop.col_dict[key], axis=1, errors='ignore')
        train['Y'] = train['Y'].drop(drop.col_dict[key], axis=1, errors='ignore')
    
    
    # FILL nan value
    # (1) With Mean of that columns
    for key in CONST.mean_fill():
        train['X'][key] = train['X'][key].fillna(train['X'][key].mean())
    
    # (2) With Medians
    for key in CONST.median_fill():
        train['X'][key] = train['X'][key].fillna(train['X'][key].mean())
    
    # (3) With specific tricks
    train['X']['deferral_term'].fillna(value=2.0, inplace=True)
    train['X']['emp_length'].fillna(value='<1 year', inplace=True)
    train['X']['hardship_amount'].fillna(value=0.0, inplace=True)
    train['X']['hardship_dpd'].fillna(value=0.0, inplace=True)
    train['X']['hardship_flag'].fillna(value='N', inplace=True)
    train['X']['hardship_loan_status'].fillna(value='Wedonotknow', inplace=True)
    train['X']['hardship_status'].fillna(value='Wedonotknow', inplace=True)
    train['X']['hardship_type'].fillna(value='Wedonotknow', inplace=True)
    train['X']['inq_last_6mths'].fillna(value=0.0, inplace=True)
    train['X']['next_pymnt_d'].fillna(value='Sep-20', inplace=True)
    train['X']['num_tl_120dpd_2m'].fillna(value=0.0, inplace=True)
    train['X']['open_acc_6m'].fillna(value=0.0, inplace=True)
    train['X']['open_il_12m'].fillna(value=0.0, inplace=True)
    train['X']['revol_util'].fillna(value='47.8%', inplace=True) # '47.8%' is the mean of that column
    train['X']['sec_app_chargeoff_within_12_mths'].fillna(value=0.0, inplace=True)
    train['X']['sec_app_collections_12_mths_ex_med'].fillna(value=0.0, inplace=True)
    train['X']['sec_app_inq_last_6mths'].fillna(value=0.0, inplace=True)
    train['X']['sec_app_mort_acc'].fillna(value=0.0, inplace=True)
    train['X']['settlement_status'].fillna(value='Wedonotknow', inplace=True)
    train['X']['total_cu_tl'].fillna(value=0.0, inplace=True)
    train['X']['verification_status_joint'].fillna(value='Wedonotknow', inplace=True)
    
    
    # Special value conversion
    train['X']['int_rate']   = train['X']['int_rate'].apply(CONST.percent2f)
    train['X']['revol_util'] = train['X']['revol_util'].apply(CONST.percent2f)
    train['X']['term']       = train['X']['term'].apply(CONST.month2i)
    
    
    # Normalization all numerical columns so far
    for col_name in train['X'].columns:
        if train['X'][col_name].dtype != 'object':
            train['X'][col_name] = (train['X'][col_name] - train['X'][col_name].mean()) / (train['X'][col_name].std() + 1e-8)
    
    
    # Label encode train['X'] on string columns whose ordering matter
    # (1) 'A'-'G' into 7-1 and others into 0
    encode_dict = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
    train['X']['grade'] = train['X']['grade'].apply(lambda x : CONST.category2i(x, encode_dict)).astype('uint8')
    del encode_dict
    
    # (2) 'A1'-'A5' into 35-31,... , 'G1'-'G5' into 5-1, and others into 0
    encode_dict = {'A1':35, 'A2':34, 'A3':33, 'A4':32, 'A5':31, 
                   'B1':30, 'B2':29, 'B3':28, 'B4':27, 'B5':26, 
                   'C1':25, 'C2':24, 'C3':23, 'C4':22, 'C5':21, 
                   'D1':20, 'D2':19, 'D3':18, 'D4':17, 'D5':16, 
                   'E1':15, 'E2':14, 'E3':13, 'E4':12, 'E5':11, 
                   'F1':10, 'F2': 9, 'F3': 8, 'F4': 7, 'F5': 6, 
                   'G1': 5, 'G2': 4, 'G3': 3, 'G4': 2, 'G5': 1}
    train['X']['sub_grade'] = train['X']['sub_grade'].apply(lambda x : CONST.category2i(x, encode_dict)).astype('uint8')
    del encode_dict
    
    # (3) Text descriptions of years into integers
    encode_dict = {'<1 year' : 0, '1 year' : 1, '2 years' : 2, '3 years' : 3,
                   '4 years' : 4, '5 years' : 5, '6 years' : 6, '7 years' : 7, 
                   '8 years' : 8, '9 years' : 9, '10+ years':19} 
    train['X']['emp_length'] = train['X']['emp_length'].apply(lambda x : CONST.category2i(x, encode_dict)).astype('uint8')
    del encode_dict
    
    
    # One-hot encode train['X'] on the remaining string columns whose ordering do not matter
    train['X'] = pandas.get_dummies(train['X'], prefix_sep='=', dummy_na=True, drop_first=True)
    
    
    # Label encode train['Y']: Label='Y' into 1 and 'N' into 0
    sklearn_LE = LabelEncoder()
    sklearn_LE.fit(['N', 'Y'])
    train['Y']['loan_status'] = sklearn_LE.transform(train['Y']['loan_status'])
    
    pool.close()
    return train


def processTest(test):
    pool,drop = Pool(), DROP()
    
    # FIND columns and rows with specific attributes
    
    # (1) Rows contain only NaN
    all_nan_rows = pool.map(checkAllNaNRow, itertools.product(test['X'].index, [test['X'].drop(CONST.id_col()+['upload_index'], axis=1)]))
    all_nan_rows = list(set(all_nan_rows)) # Remove duplicated rows
    all_nan_rows.remove(-1) # Pop out the 1st element '-1'
    drop.row_dict['all_nan'] = all_nan_rows
    
    # (2) Columns contain only id
    drop.col_dict['id'] = CONST.id_col() + ['upload_index']
    
    # (3) Columns that are redundant or contain too many attributes
    drop.col_dict['redundant'] = CONST.redundant_col()
    
    # (4) 'title' column
    drop.col_dict['title'] = ['title']
    
    
    # DROP columns and rows
    # (1) Rows: Cannot drop any row in testing data, but we will fill some values later
    pass
    
    # (2) Columns
    for key in drop.col_dict.keys():
        test['X'] = test['X'].drop(drop.col_dict[key], axis=1, errors='ignore')
        test['Y'] = test['Y'].drop(drop.col_dict[key], axis=1, errors='ignore')
    
    
    # FILL nan value
    # (1) With Mean of that columns
    for key in CONST.mean_fill():
        test['X'][key] = test['X'][key].fillna(test['X'][key].mean())
    
    # (2) With Medians
    for key in CONST.median_fill():
        test['X'][key] = test['X'][key].fillna(test['X'][key].mean())
    
    # (3) With specific tricks
    test['X']['deferral_term'].fillna(value=2.0, inplace=True)
    test['X']['emp_length'].fillna(value='<1 year', inplace=True)
    test['X']['grade'].fillna(value='B', inplace=True)
    test['X']['hardship_amount'].fillna(value=0.0, inplace=True)
    test['X']['hardship_dpd'].fillna(value=0.0, inplace=True)
    test['X']['hardship_flag'].fillna(value='N', inplace=True)
    test['X']['hardship_loan_status'].fillna(value='Wedonotknow', inplace=True)
    test['X']['hardship_status'].fillna(value='Wedonotknow', inplace=True)
    test['X']['hardship_type'].fillna(value='Wedonotknow', inplace=True)
    test['X']['inq_last_6mths'].fillna(value=0.0, inplace=True)
    test['X']['next_pymnt_d'].fillna(value='Sep-20', inplace=True)
    test['X']['num_tl_120dpd_2m'].fillna(value=0.0, inplace=True)
    test['X']['open_acc_6m'].fillna(value=0.0, inplace=True)
    test['X']['open_il_12m'].fillna(value=0.0, inplace=True)
    test['X']['revol_util'].fillna(value='47.8%', inplace=True) # '47.8%' is the mean of that column
    test['X']['sec_app_chargeoff_within_12_mths'].fillna(value=0.0, inplace=True)
    test['X']['sec_app_collections_12_mths_ex_med'].fillna(value=0.0, inplace=True)
    test['X']['sec_app_inq_last_6mths'].fillna(value=0.0, inplace=True)
    test['X']['sec_app_mort_acc'].fillna(value=0.0, inplace=True)
    test['X']['settlement_status'].fillna(value='Wedonotknow', inplace=True)
    test['X']['total_cu_tl'].fillna(value=0.0, inplace=True)
    test['X']['verification_status_joint'].fillna(value='Wedonotknow', inplace=True)
    
    # (4) Spacial unknown label is only contained by testing data
    test['X']['hardship_loan_status'].replace(to_replace='CLOSED', value='ACTIVE', inplace=True)
    
    
    # Special value conversion
    test['X']['int_rate']   = test['X']['int_rate'].apply(CONST.percent2f)
    test['X']['revol_util'] = test['X']['revol_util'].apply(CONST.percent2f)
    test['X']['term']       = test['X']['term'].apply(CONST.month2i)
    
    
    # Deal with those all-nan rows
    col_name_list = [col_name for col_name in test['X'].columns if col_name not in (CONST.id_col()+['upload_index'])]
    for index in drop.row_dict['all_nan']:
        for col_name in col_name_list:
            test['X'].loc[index, col_name] = test['X'][col_name].mode()[0]
    
    
    # Normalization all numerical columns so far
    for col_name in test['X'].columns:
        if test['X'][col_name].dtype == 'object':
            pass
        else:
            test['X'][col_name] = (test['X'][col_name] - test['X'][col_name].mean()) / (test['X'][col_name].std() + 1e-8)
    
    
    # Label encode test['X'] on string columns whose ordering matter
    # (1) 'A'-'G' into 7-1 and others into 0
    encode_dict = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
    test['X']['grade'] = test['X']['grade'].apply(lambda x : CONST.category2i(x, encode_dict)).astype('uint8')
    del encode_dict
    
    # (2) 'A1'-'A5' into 35-31,... , 'G1'-'G5' into 5-1, and others into 0
    encode_dict = {'A1':35, 'A2':34, 'A3':33, 'A4':32, 'A5':31, 
                   'B1':30, 'B2':29, 'B3':28, 'B4':27, 'B5':26, 
                   'C1':25, 'C2':24, 'C3':23, 'C4':22, 'C5':21, 
                   'D1':20, 'D2':19, 'D3':18, 'D4':17, 'D5':16, 
                   'E1':15, 'E2':14, 'E3':13, 'E4':12, 'E5':11, 
                   'F1':10, 'F2': 9, 'F3': 8, 'F4': 7, 'F5': 6, 
                   'G1': 5, 'G2': 4, 'G3': 3, 'G4': 2, 'G5': 1}
    test['X']['sub_grade'] = test['X']['sub_grade'].apply(lambda x : CONST.category2i(x, encode_dict)).astype('uint8')
    del encode_dict
    
    # (3) Text descriptions of years into integers
    encode_dict = {'<1 year' : 0, '1 year' : 1, '2 years' : 2, '3 years' : 3,
                   '4 years' : 4, '5 years' : 5, '6 years' : 6, '7 years' : 7, 
                   '8 years' : 8, '9 years' : 9, '10+ years':19} 
    test['X']['emp_length'] = test['X']['emp_length'].apply(lambda x : CONST.category2i(x, encode_dict)).astype('uint8')
    del encode_dict
    
    
    # One-hot encode test['X'] on the remaining string columns whose ordering do not matter
    test['X'] = pandas.get_dummies(test['X'], prefix_sep='=', dummy_na=True, drop_first=True)
    
    pool.close()
    return test


def checkAllNaNRow(pack_tuple):
    row_index, pandas_df = pack_tuple
    if pandas_df.iloc[row_index].isnull().all() == True:
        return row_index
    else:
        return -1


def overSample(df_X, df_Y):
    combined_df    = pandas.concat([df_X, df_Y], axis=1)    
    label_col_name = list(df_Y.columns)[0]
    
    lrgest_class_size = combined_df[label_col_name].value_counts().max()
    to_df_list        = [combined_df]
    for _, group in combined_df.groupby(label_col_name):
        surplus_num = lrgest_class_size - len(group)
        to_df_list.append(group.sample(surplus_num, replace=True)) # Sample with replacement
    combined_df = pandas.concat(to_df_list).reset_index(drop=True)
        
    # Update training dataset with oversampling version
    X_part_cols = [col_name for col_name in combined_df.columns if col_name not in [label_col_name]]
    Y_part_cols = [label_col_name]
    return combined_df[X_part_cols], combined_df[Y_part_cols]


def saveDataToPK(data, file_name):
    with open('./PKs/' + file_name, 'wb') as f:
        pickle.dump(data, f)
        print("Save {} under directory 'PKs'".format(file_name))


########## Main function ##########
if __name__ == "__main__":
    print("Generate train.pk and test.pk will take ~42 GB ram.")
    
    # Display pandas DataFrame without truncation
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    
    # Read raw *_train csv files, process data, and then store them into a .pk file
    print("Read training data .csv")
    train = {'X': readCSV('./Training/X_train.csv'), 'Y': readCSV('./Training/Y_train.csv')}
    print("Do some data cleaning on training data")
    train = processTrain(train)
    print("Oversample training data")
    train['X'], train['Y'] = overSample(train['X'], train['Y'])
    saveDataToPK(train, 'train.pk')
    del train
    
    # Read raw *_test csv files, process data, and then store them into a .pk file
    print("Read testing data .csv")
    test = {'X': readCSV('./Testing/X_test.csv'), 'Y': readCSV('./Testing/Y_test.csv')}
    print("Do some data cleaning on testing data")
    test = processTest(test)
    saveDataToPK(test, 'test.pk')
    del test

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
    percent2f      = lambda x : (float(x.strip('%')) + 0.0) * 1e-2 # Adding 0.0 is used to prevent negative zero
    month2i        = lambda x : int(x.strip(' months'))
    category2i     = lambda x, ct_dict: ct_dict[x] if x in ct_dict else 0
    have_CUDA      = lambda : 'cuda' if torch.cuda.is_available() else 'cpu'


class DROP:
    col_dict = {}
    row_dict = {}


class FCModel(torch.nn.Module):
    def __init__(self, input_dim, dp_ratio):
        super(FCModel, self).__init__()
        self.fc1 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=input_dim, out_features=512),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=dp_ratio)
                   )
        self.fc2 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=512, out_features=512),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=dp_ratio)
                   )
        self.fc3 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=512, out_features=256),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=dp_ratio)
                   )
        self.fc4 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=256, out_features=128),
                       torch.nn.ReLU()
                   )
        self.fc5 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=128, out_features=64),
                       torch.nn.ReLU()
                   )
        self.fc6 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=64, out_features=32),
                       torch.nn.ReLU()
                   )
        self.fc7 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=32, out_features=1),
                       torch.nn.Sigmoid()
                       #torch.nn.LogSoftmax(dim=1)
                   )
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        #x = x[:, 0]
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


def saveTrainToPK():
    print("\nAre yot sure that you want to generate train.pk file?")
    print("It may take a lot of memory usage (~12GB).")
    enable = str(input("Enter Y/N: ")).upper()
    
    if enable == 'Y':
        print("\nGenerate train.pk file.")
        pass
    elif enable == 'N':
        print("\nAutomatically terminates program.")
        exit()
    else:
        print("Please enter Y or N.")
        exit()
    
    pool = Pool()
    drop = DROP()
    
    # Read raw csv data into two pandas DataFrames
    train = {}
    train['X'], train['Y'] = readCSV('./Training/X_train.csv'), readCSV('./Training/Y_train.csv')
    
    # FIND columns and rows with specific attributes
    #      ROWs that contain only NaN
    drop.row_dict['all_nan'] = pool.map(checkAllNaNRow, itertools.product(train['X'].index, [train['X'].drop(CONST.id_col()+['upload_index'], axis=1)]))
    drop.row_dict['all_nan'] = list(set(drop.row_dict['all_nan']))
    drop.row_dict['all_nan'].sort()
    drop.row_dict['all_nan'].pop(0) # Pop out the 1st element '-1'
    
    #      COLumns contain only id
    drop.col_dict['id'] = CONST.id_col() + ['upload_index']
    
    #      COLumns that are redundant or contain too many attributes
    drop.col_dict['redundant'] = CONST.redundant_col()
    
    # DROP columns and rows
    #      Rows that found at previous steps
    for key in drop.row_dict.keys():
        train['X'] = train['X'].drop(drop.row_dict[key], errors='ignore')
        train['Y'] = train['Y'].drop(drop.row_dict[key], errors='ignore')
    
    #      Columns that found at previous steps
    for key in drop.col_dict.keys():
        train['X'] = train['X'].drop(drop.col_dict[key], axis=1, errors='ignore')
        train['Y'] = train['Y'].drop(drop.col_dict[key], axis=1, errors='ignore')
    
    # FILL nan value
    #                With MEAN of that columns
    for key in CONST.mean_fill():
        train['X'][key] = train['X'][key].fillna(train['X'][key].mean())
    
    #                With MEDIAN of that columns
    for key in CONST.median_fill():
        train['X'][key] = train['X'][key].fillna(train['X'][key].mean())
    
    #                With specific tricks
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
    
    train['X'] = train['X'].drop(['title'], axis=1, errors='ignore')
    
    # Special value conversion
    train['X']['int_rate'] = train['X']['int_rate'].apply(CONST.percent2f)
    train['X']['revol_util'] = train['X']['revol_util'].apply(CONST.percent2f)
    train['X']['term'] = train['X']['term'].apply(CONST.month2i)
    
    '''
    # Encode 'A'-'G' into 7-1 and others into 0
    encode_dict = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
    train['X']['grade'] = train['X']['grade'].apply(lambda x : CONST.category2i(x, encode_dict))
    del encode_dict
    '''
    train['X'] = train['X'].drop(['grade'], axis=1, errors='ignore')
    
    # Normalization numerical columns so far
    for col_name in train['X'].columns:
        if train['X'][col_name].dtype != 'object':
            train['X'][col_name] = (train['X'][col_name] - train['X'][col_name].mean())/(train['X'][col_name].max()-train['X'][col_name].min()+1e-8)
    
    # Encode 'A1'-'A5' into 35-31,... , 'G1'-'G5' into 5-1, and others into 0
    encode_dict = {'A1':35, 'A2':34, 'A3':33, 'A4':32, 'A5':31, 
                   'B1':30, 'B2':29, 'B3':28, 'B4':27, 'B5':26, 
                   'C1':25, 'C2':24, 'C3':23, 'C4':22, 'C5':21, 
                   'D1':20, 'D2':19, 'D3':18, 'D4':17, 'D5':16,
                   'E1':15, 'E2':14, 'E3':13, 'E4':12, 'E5':11,
                   'F1':10, 'F2': 9, 'F3': 8, 'F4': 7, 'F5': 6, 
                   'G1': 5, 'G2': 4, 'G3': 3, 'G4': 2, 'G5': 1} 
    train['X']['sub_grade'] = train['X']['sub_grade'].apply(lambda x : CONST.category2i(x, encode_dict))
    del encode_dict
    
    # Encode text descriptions of years into integers
    encode_dict = {'<1 year' : 0, '1 year' : 1, '2 years' : 2, '3 years' : 3,
                   '4 years' : 4, '5 years' : 5, '6 years' : 6, '7 years' : 7, 
                   '8 years' : 8, '9 years' : 9, '10+ years':19} 
    train['X']['emp_length'] = train['X']['emp_length'].apply(lambda x : CONST.category2i(x, encode_dict))
    del encode_dict
    
    # Encode the remaining string columns from categorical to numerical
    sklearn_LE  = LabelEncoder()
    for col_name in train['X'].columns:
        if train['X'][col_name].dtype == 'object':
            sklearn_LE.fit(train['X'][col_name].to_numpy())
            train['X'][col_name] = sklearn_LE.transform(train['X'][col_name])
    
    # Encode label 'Y' into 1 and 'N' into 0
    sklearn_LE.fit(['N', 'Y'])
    train['Y']['loan_status'] = sklearn_LE.transform(train['Y']['loan_status'])
    
    # Store Dataframe into .pk file
    with open('./Training/train.pk', 'wb') as f:
        pickle.dump(train, f)
        print("Save train.pk under ./Training")
    
    # Terminate the program
    pool.close()
    exit()


def saveTestToPK():
    pool = Pool()
    drop = DROP()
    
    # Read raw csv data into two pandas DataFrames
    test = {}
    test['X'], test['Y'] = readCSV('./Testing/X_test.csv'), readCSV('./Testing/Y_test.csv')
    
    # FIND columns and rows with specific attributes
    #      All ROWs that contain only NaN
    drop.row_dict['all_nan'] = pool.map(checkAllNaNRow, itertools.product(test['X'].index, [test['X'].drop(CONST.id_col()+['upload_index'], axis=1)]))
    drop.row_dict['all_nan'] = list(set(drop.row_dict['all_nan']))
    drop.row_dict['all_nan'].sort()
    drop.row_dict['all_nan'].pop(0) # Pop out the 1st element '-1'
    
    #      COLumns contain only id
    drop.col_dict['id'] = CONST.id_col() + ['upload_index']
    
    #      COLumns that are redundant or contain too many attributes
    drop.col_dict['redundant'] = CONST.redundant_col()
    
    # DROP columns and rows
    #      Rows: Fill nan value with 0 (cannot be dropped because it's the testing data)
    test['X'].iloc[drop.row_dict['all_nan']] = test['X'].iloc[drop.row_dict['all_nan']].fillna(0)
    
    #      Columns
    for key in drop.col_dict.keys():
        test['X'] = test['X'].drop(drop.col_dict[key], axis=1, errors='ignore')
        test['Y'] = test['Y'].drop(drop.col_dict[key], axis=1, errors='ignore')
    
    # FILL nan value
    #                With MEAN of that columns
    for key in CONST.mean_fill():
        test['X'][key] = test['X'][key].fillna(test['X'][key].mean())
    
    #                With MEDIAN of that columns
    for key in CONST.median_fill():
        test['X'][key] = test['X'][key].fillna(test['X'][key].mean())
    
    
    
    
    
    
    # Store Dataframe into .pk file
    with open('./Testing/test.pk', 'wb') as f:
        pickle.dump(test, f)
    
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


def have_CUDA():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def calcAccuracy(pred_tensor, truth_tensor):
    accuracy = 0.0
    
    if pred_tensor.size() != truth_tensor.size():
        print("Different dimensions to pred_tensor and truth_tensor.")
        print("pred_tensor dim={}".format(pred_tensor.size()))
        print("truth_tensor dim={}".format(truth_tensor.size()))
        exit()
    
    for index in range(pred_tensor.size(0)):
        if pred_tensor[index] > 0.5:
            each_pred_label = torch.Tensor([1.0]).to(CONST.have_CUDA())
        else:
            each_pred_label = torch.Tensor([0.0]).to(CONST.have_CUDA())
        
        if each_pred_label == truth_tensor[index].to(CONST.have_CUDA()):
            accuracy += 1.0
        else:
            pass
    
    return accuracy / pred_tensor.size(0)


def overSample(df_X, df_Y):
    combined_df    = pandas.concat([train['X'], train['Y']], axis=1)    
    label_col_name = list(train['Y'].columns)[0]
    
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


########## Main function ##########
if __name__ == "__main__":
    # Display pandas DataFrame without truncation
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    
    
    # Check 'train.pk' existence
    if os.path.isfile('./Training/train.pk') is not True:
        saveTrainToPK() # Invoke 'exit()' after storing train.pk file
    else:
        pass
    '''
    # Check 'test.pk' existence
    if os.path.isfile('./Testing/test.pk') is not True:
        saveTestToPK() # Invoke 'exit()' after storing test.pk file
    else:
        pass
    '''
    # Read .pk files
    # 'train' is a dict {'X': train['X'].csv DataFrame after dropping cols and rows, 'Y': Y_train.csv DF after...}
    # 'test' is a dict {'X': X_test.csv DataFrame} (do not have 'Y')
    with open('./Training/proper.train.pk', 'rb') as f1:
        print("Reading train.pk and test.pk")
        train = pickle.load(f1)
        #test = pickle.load(f2)
    
    #print("Oversampling datasets")
    #train['X'], train['Y'] = overSample(train['X'], train['Y'])
    
    # Shuffle dataset
    rand_int = int(time.time())
    shuffle_array = numpy.arange(train['X'].shape[0])
    random.shuffle(shuffle_array)
    train['X'] = train['X'][shuffle_array]
    train['Y'] = train['Y'][shuffle_array]
    #train['X'] = train['X'].sample(frac=1, random_state=0).reset_index(drop=True)
    #train['Y'] = train['Y'].sample(frac=1, random_state=0).reset_index(drop=True)
    
    # Split whole training dataset into training part and validation part
    #train_train_part, train_valid_part = splitDataset(train)
    
    # Prepare PyTorch DataLoader
    train_tensor = {}
    train_tensor['X'] = torch.Tensor(train['X'])
    train_tensor['Y'] = torch.Tensor(train['Y'])
    
    #train_tensor['X'] = torch.Tensor(train['X'].to_numpy())
    #train_tensor['Y'] = torch.Tensor(train['Y'].to_numpy())
    
    torch_dataset = torch.utils.data.TensorDataset(train_tensor['X'], train_tensor['Y'])
    torch_loader  = torch.utils.data.DataLoader(torch_dataset, batch_size=CONST.batch_size(), num_workers=0)
    
    # Prepare some initial steps
    print("Initializing NN model")
    fc_model  = FCModel(input_dim=train_tensor['X'].shape[1], dp_ratio=CONST.dp_ratio())
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(fc_model.parameters(), lr=CONST.lr_rate())
    fc_model.to(CONST.have_CUDA())  # or fc_model.cuda()
    fc_model.train() # Enable self.training to True
    
    print("Start training.")
    for epoch in range(CONST.epoch_num()):
        epoch_start = time.time()
        avg_loss, avg_acuy = 0.0, 0.0
        
        for data, label in torch_loader:
            data, label = data.to(CONST.have_CUDA()), label.to(CONST.have_CUDA()) # Map data into HW device
            predict = fc_model(data).to(CONST.have_CUDA())
            
            # Calculate loss and accuracy
            temp_loss = loss_func(predict.view(-1), label.view(-1))
            temp_acuy = calcAccuracy(predict.view(-1), label.view(-1))
            
            # Backward propagation
            optimizer.zero_grad() # Set all the gradients to zero before backward propragation
            temp_loss.backward()
                        
            # Performs a single optimization step.
            optimizer.step()
            
            avg_loss += temp_loss.item() * data.size(0)
            avg_acuy += temp_acuy
        
        avg_loss, avg_acuy = avg_loss / len(torch_loader), avg_acuy / len(torch_loader)
        epoch_time = time.time()-epoch_start
        print('Epoch: {}\tLoss: {:.2f}\tAccuracy: {:.2f}\tEpoch time: {:.2f}'.format(epoch+1, avg_loss, avg_acuy, epoch_time))
    
    model_file_name = 'pretrained_' + str(int(time.time())) + '.pk'
    torch.save(fc_model, model_file_name)
    print("Pre-trained model have been saved as {}.".format(model_file_name))


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
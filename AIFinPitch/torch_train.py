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
    vt_ratio      = lambda : 0.3 # validation.size / training.size ratio, i.e., 10-fold cross-validation
    batch_size    = lambda : 2**10
    epoch_num     = lambda : 50
    lr_rate       = lambda : 1e-5
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
    haveCUDA       = lambda : sys.argv[1] if torch.cuda.is_available() else 'cpu'


class DROP:
    col_dict = {}
    row_dict = {}


class FCModel(torch.nn.Module):
    def __init__(self, input_dim, dp_ratio):
        super(FCModel, self).__init__()
        self.fc1 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=input_dim, out_features=2**10),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=dp_ratio)
                   )
        self.fc2 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=2**10, out_features=2**10),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=dp_ratio)
                   )
        self.fc3 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=2**10, out_features=2**10),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(p=dp_ratio)
                   )
        self.fc4 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=2**10, out_features=2**6),
                       torch.nn.ReLU()
                   )
        self.fc5 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=2**6, out_features=2**6),
                       torch.nn.ReLU()
                   )
        self.fc6 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=2**6, out_features=2**2),
                       torch.nn.ReLU()
                   )
        self.fc7 = torch.nn.Sequential(
                       torch.nn.Linear(in_features=2**2, out_features=2),
                       torch.nn.Sigmoid(),
                       torch.nn.Softmax(dim=1)
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


# predict: torch.Tensor with size=(xxx, ), target: torch.Tensor with size=(xxx, )
def f1Score(predict, target, threshold):
    predict_bin, target_bool = (predict > threshold), (target.double() == 1.0)
    
    TP = ((predict_bin==True)  * (target_bool==True))
    TN = ((predict_bin==False) * (target_bool==False))
    FP = ((predict_bin==True)  * (target_bool==False))
    FN = ((predict_bin==False) * (target_bool==True))
    
    precision = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-9)
    recall    = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-9)
    return (2.0 * precision * recall) / (precision + recall + 1e-9)


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
    del test
    
    # Split whole training dataset into training part and validation part
    train_train, train_valid = {}, {}
    train_train['X'], train_valid['X'], train_train['Y'], train_valid['Y'] = train_test_split(train['X'], train['Y'], test_size=CONST.vt_ratio())
    
    # Prepare PyTorch DataLoader
    train_tensor = {}
    train_tensor['X'] = torch.Tensor(train_train['X'])
    train_tensor['Y'] = torch.Tensor(train_train['Y'])
    
    valid_tensor = {}
    valid_tensor['X'] = torch.Tensor(train_valid['X'])
    valid_tensor['Y'] = torch.Tensor(train_valid['Y'])
    
    torch_dataset = torch.utils.data.TensorDataset(train_tensor['X'], train_tensor['Y'])
    torch_loader  = torch.utils.data.DataLoader(torch_dataset, batch_size=CONST.batch_size(), num_workers=0)
    
    # Prepare some initial steps
    print("Initializing NN model")
    fc_model  = FCModel(input_dim=train_tensor['X'].shape[1], dp_ratio=CONST.dp_ratio())
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(fc_model.parameters(), lr=CONST.lr_rate())
    fc_model.to(CONST.haveCUDA())  # or fc_model.cuda()
    fc_model.train() # Enable self.training to True
    
    print("Start training.")
    for epoch in range(CONST.epoch_num()):
        epoch_start = time.time()
        avg_loss, avg_f1 = 0.0, 0.0
        
        for data, label in torch_loader:
            data, label = data.to(CONST.haveCUDA()), label.to(CONST.haveCUDA())
            predict = fc_model(data).to(CONST.haveCUDA())
            
            # Calculate loss and accuracy
            temp_loss = loss_func(predict.view(-1), label.view(-1))
            
            # Backward propagation
            optimizer.zero_grad() # Set all the gradients to zero before backward propragation
            temp_loss.backward()
            
            # Performs a single optimization step.
            optimizer.step()
            
            avg_loss += temp_loss.item() * data.size(0)
                
        avg_loss = avg_loss / len(torch_loader)
        epoch_time = time.time()-epoch_start
        print('Epoch: {}\tLoss: {:.2f}\tEpoch time: {:.2f}'.format(epoch+1, avg_loss, epoch_time))
    
    # Print F1 score
    fc_model.eval() # Enable pytorch model testing mode
    predict_valid     = fc_model(valid_tensor['X'].to(CONST.haveCUDA())).to(CONST.haveCUDA())
    valid_tensor['Y'] = valid_tensor['Y'].to(CONST.haveCUDA())
    f1_score_0_38 = f1Score(predict_valid.view(-1), valid_tensor['Y'].view(-1), 0.38)
    f1_score_0_50 = f1Score(predict_valid.view(-1), valid_tensor['Y'].view(-1), 0.5)
    print("F1 Score(th=0.38): {}, F1 Score(th=0.50): {}".format(f1_score_0_38, f1_score_0_50))
    
    # Train the model with whole data at last time
    fc_model.train()
    train['X'] = torch.Tensor(train['X'])
    train['Y'] = torch.Tensor(train['Y'])
    torch_dataset = torch.utils.data.TensorDataset(train['X'], train['Y'])
    torch_loader  = torch.utils.data.DataLoader(torch_dataset, batch_size=CONST.batch_size(), num_workers=0)
    
    for data, label in torch_loader:
        data, label = data.to(CONST.haveCUDA()), label.to(CONST.haveCUDA())
        predict     = fc_model(data).to(CONST.haveCUDA())
        temp_loss   = loss_func(predict.view(-1), label.view(-1))
        optimizer.zero_grad()
        temp_loss.backward()
        optimizer.step()
    
    # Print F1 score
    fc_model.eval()
    fc_model.to('cpu') 
    predict_valid     = fc_model(valid_tensor['X'].to('cpu')).to('cpu')
    valid_tensor['Y'] = valid_tensor['Y'].to('cpu')
    f1_score_0_38 = f1Score(predict_valid.view(-1), valid_tensor['Y'].view(-1), 0.38)
    f1_score_0_50 = f1Score(predict_valid.view(-1), valid_tensor['Y'].view(-1), 0.5)
    print("\n\nF1 Score(th=0.38): {}, F1 Score(th=0.50): {}".format(f1_score_0_38, f1_score_0_50))
    
    # Save model into .pk file
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
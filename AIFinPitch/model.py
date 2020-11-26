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
import itertools
import math
from   multiprocessing import Pool
import numpy
import pandas
import pickle
import random
from   sklearn.preprocessing import LabelEncoder
import time
import torch
import torch.nn
import torch.nn.functional
from   torch import optim


########## Classes ##########
class CONST:
    # Define some constants
    row_pixel  = lambda : 26
    col_pixel  = lambda : 26
    tv_ratio   = lambda : 0.8 # training validation ratio
    batch_size = lambda : 32768
    id_col     = lambda : ['id', 'member_id']
    epoch_num  = lambda : 1000
    lr_rate    = lambda : 0.25


class DROP:
    col_dict = {}
    row_dict = {}


class FCModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(FCModel, self).__init__()
        self.fc1 = torch.nn.Sequential(
                            torch.nn.Linear(in_features=input_dim, out_features=10000),
                            torch.nn.ReLU()
                     )
        self.fc2 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=10000, out_features=100),
                           torch.nn.ReLU()
                       )
        self.fc3 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=100, out_features=10),
                           torch.nn.ReLU()
                       )
        self.fc4 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=100, out_features=10),
                           torch.nn.ReLU()
                       )
        self.fc5 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=50, out_features=25),
                           torch.nn.ReLU()
                       )
        self.fc6 = torch.nn.Sequential(
                           torch.nn.Linear(in_features=10, out_features=2),
                           torch.nn.LogSoftmax(dim=1)
                       )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.fc4(x)
        #x = self.fc5(x)
        x = self.fc6(x)
        x = x[:, 0]
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
    
    # 'upload_index' is only important to X_test and Y_test, useless to 'X_train' and 'Y_train'
    train['X'] = train['X'].drop('upload_index', axis=1, errors='ignore')
    train['Y'] = train['Y'].drop('upload_index', axis=1, errors='ignore')
    
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


def have_CUDA():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def lossFunc(y_predict, y_truth):
    BCE_loss = torch.nn.functional.binary_cross_entropy(x_recon, x)
    
    return BCE_loss


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
        
    # Oversample dataset 'N' : 1834130, 'Y' : 298637
    combined_df = pandas.concat([train['X'], train['Y']], axis=1)
    largest_class_num =  combined_df['loan_status'].value_counts().max()
    
    to_df_list = [combined_df]
    for _, group in combined_df.groupby('loan_status'):
        surplus_num = largest_class_num - len(group)
        to_df_list.append(group.sample(surplus_num, replace=True)) # Sample with replacement
    combined_df = pandas.concat(to_df_list).reset_index(drop=True) 
    
    # Update training dataset with oversampling version
    X_part_cols = [col_name for col_name in combined_df.columns if col_name not in ['loan_status']]
    Y_part_cols = ['loan_status']
    train['X'] = combined_df[X_part_cols].copy() # Make a true copy, not a passing-reference
    train['Y'] = combined_df[Y_part_cols].copy() # Make a true copy, not a passing-reference
    del combined_df # Release memory space used by DataFrame
    
    # Feature normalization for X_part data
    for col_name in train['X'].columns:
        train['X'][col_name] = train['X'][col_name] - train['X'][col_name].mean()
    
    # Encode label 'Y' into 1 and 'N' into 0
    sklearn_LE = LabelEncoder()
    sklearn_LE.fit(['N', 'Y'])
    train['Y']['loan_status'] = sklearn_LE.transform(train['Y']['loan_status'])
    
    # Shuffle dataset
    rand_int = int(time.time())
    train['X'] = train['X'].sample(frac=1, random_state=0).reset_index(drop=True)
    train['Y'] = train['Y'].sample(frac=1, random_state=0).reset_index(drop=True)
    
    # Split whole training dataset into training part and validation part
    train_train_part, train_valid_part = splitDataset(train)
    
    # Prepare PyTorch DataLoader
    train_train_tensor_X = torch.Tensor(train_train_part['X'].to_numpy())
    train_train_tensor_Y = torch.Tensor(train_train_part['Y'].to_numpy())
    
    torch_dataset = torch.utils.data.TensorDataset(train_train_tensor_X, train_train_tensor_Y)
    torch_loader  = torch.utils.data.DataLoader(torch_dataset, batch_size=CONST.batch_size(), num_workers=0)
    
    # Prepare some initial steps
    hw_device = have_CUDA()
    fc_model  = FCModel(input_dim=train_train_tensor_X.shape[1])
    optimizer = torch.optim.Adam(fc_model.parameters(), lr=CONST.lr_rate())
    fc_model.cuda()  # or fc_model.to(hw_device)
    fc_model.train() # Enable self.training to True
    
    for epoch in range(CONST.epoch_num()):
        epoch_start = time.time()
        cur_loss = 0.0
        for data, label in torch_loader:
            data, label = data.cuda(), label.cuda() # Map data into HW device
            predict = fc_model(data)
            
            # Calculate loss value
            loss_func = torch.nn.MSELoss()
            temp_loss = loss_func(predict, label.view(-1))
            
            # Backward propagation
            optimizer.zero_grad() # Set all the gradients to zero before backward propragation
            temp_loss.backward()
            
            # Performs a single optimization step.
            optimizer.step()
            cur_loss += temp_loss.item() * data.size(0)
              
        cur_loss = cur_loss / len(torch_loader)
        print('Epoch: {}\tTraining loss: {:.4f}\tEpoch time: {:.2f}'.format(epoch+1, cur_loss, time.time()-epoch_start))
    '''
    for i in train['X'].columns:
        pyplot.hist(train['X'][col_name])
        pyplot.gcf().savefig('./img/'+str(col_name)+'.png')
        pyplot.clf() # Clear the current figure and axes
    '''


# Other codes
#print(train_data['X'].info(memory_usage='deep', verbose=True))
#pool.map(saveHist, [(train['X'][col_name], './img/'+str(col_name)+'.png') for col_name in train['X'].columns])

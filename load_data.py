from os.path import join as pjoin
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.preprocessing import StandardScaler as STD
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
import pickle
import torch

def load_targets(df_path, idx, target):
    df = pd.read_csv(df_path)
    return torch.FloatTensor(df.loc[idx, target].values.astype(np.float32))

def load_points(df_path):
    points_df=pd.read_csv(df_path)
    novo_df= points_df[['col','row']].to_numpy(dtype=int)
    return novo_df

def get_index(feat_df_file):
    with open(feat_df_file, 'rb') as pkl:
        df = pickle.load(pkl)
    return df.index

# ---------------------------------------
# Main class for managing the S4D Dataset
# ---------------------------------------

class ShellDataset(Dataset):
    def __init__(self,layer, model_list,out_resolution, data_dir):
        super().__init__()

        self.out_resolution=out_resolution
        self.model_list = model_list-1
        self.data_dir = data_dir
        self.X = []
        self.y = []
        
        if self.out_resolution =='46x46':
            self.data_points = load_points(pjoin(self.data_dir, f'data_points.csv'))
        else:
            self.data_points = load_points(pjoin(self.data_dir, f'dRMS_data_points.csv'))
        
        
        self.input_data_points = load_points(pjoin(self.data_dir, f'data_points.csv'))
        
        X_path_file=pjoin(self.data_dir,f'prepro_X.pt')
        if self.out_resolution == '46x46': 
            y_path_file=pjoin(self.data_dir,f'prepro_y.pt')
        else:
            y_path_file=pjoin(self.data_dir,f'prepro_dRMS_y.pt')
        
        X=torch.load(X_path_file)
        y=torch.load(y_path_file)
        
        
        self.X=X[self.model_list,:,:,:] 
        self.y=y[self.model_list,:,:,:]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------------------------
# PyTorch Lightning DataModule for the S4D Dataset
# ------------------------------------------------

class ShellDataModule(LightningDataModule):
    def __init__(self, layer, out_resolution, size = 100, scaler = 'std', num_workers = 4):
        super().__init__()

        self.layer = layer
        self.size = size
        self.scaler = scaler
        self.num_workers = num_workers
        self.out_resolution = out_resolution


    def setup(self, stage = None):
        train_pct = 0.7
        val_pct   = 0.1
        test_pct  = 0.2

        if self.size == 100:
            id_set = np.arange(1, 101)
        else:
            id_set = np.random.choice(
                np.arange(1, 101), self.size, replace = False)

        np.random.seed(0)
        train_val_idx, test_idx = train_test_split(id_set, test_size = test_pct)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size = val_pct / (train_pct + val_pct))

        data_dir_path=f'./dataset/tensores/{self.layer}_features'
        
        self.train_data = ShellDataset(self.layer, train_idx,self.out_resolution, data_dir=data_dir_path)
        self.val_data = ShellDataset(self.layer, val_idx,self.out_resolution,  data_dir=data_dir_path)
        self.test_data = ShellDataset(self.layer, test_idx, self.out_resolution, data_dir=data_dir_path)

        # Scaling data
        if self.scaler == 'minmax':
            self.scaler_obj = MMS()
        elif self.scaler == 'std':
            self.scaler_obj = STD()
        else:
            print(f'ERROR! Unknown normalization method: {scaler}')
            exit(1)


        # Transforming data after fitting scaler to training set
        self.train_data = self.normalize(self.train_data)
        self.val_data = self.normalize(self.val_data)
        self.test_data = self.normalize(self.test_data)
    
    def normalize(self, dataset):    
        for i in range(8):
            original=dataset.X[:,i,:,:]        
            ascolumns = original.reshape(-1,1)
            t = self.scaler_obj.fit_transform(ascolumns)
            transformed = t.reshape(original.shape)
            dataset.X[:,i,:,:]=torch.from_numpy(transformed)
        return dataset
    
        
    def scale_tensor(self, T, scaler):
        return torch.from_numpy(scaler.transform(T.numpy()))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = 8, shuffle = True,
                          drop_last = True, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size = 10, num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size = 1, num_workers = self.num_workers)

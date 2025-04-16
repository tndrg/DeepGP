# -*- coding: utf-8 -*-
"""
@author: Taiyu Zhu
"""

import torch
import numpy as np
import pytorch_lightning as pl
import pickle
from torch.utils.data import Dataset,DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Vectorized operations to apply min max scaling 
def min_max_scaling(df):
    return (df - df.min()) / (df.max() - df.min())

def std_scaling_chr(x_all,cm=None,cs=None):
    if cm is None:
        cm,cs = [],[]
        for x in x_all:
            channel_mean = np.mean(x, axis=0)
            channel_std = np.std(x, axis=0)
            cm.append(channel_mean)
            cs.append(channel_std)
    output = []
    for i in range(len(x_all)):
        output.append((x_all[i] - cm[i]) / cs[i])
    return output,cm,cs



class SNPPCACHRDataModule(pl.LightningDataModule):
    def __init__(self,configs):
        super().__init__()
        label = configs.label
        self.snp_transform = MinMaxScaler()
        self.cov_transform = MinMaxScaler()
        self.batch_size = configs.batch_size
        self.num_workers = configs.num_workers
        self.data_mode = configs.dm
        self.rd_mode = configs.rd_mode
        self.load_data_dir = f'{configs.data_dir}/{label}'
        self.shuffle = configs.shuffle
        self.cont = configs.cont_pheno
        self.scaling = configs.scaling
        self.snp_embed = configs.snp_embed
        self.ve = configs.ve

        with open(f'{self.load_data_dir}/{configs.rd_mode}/genes.pkl', 'rb') as f:
                self.genes_chr = pickle.load(f)
                self.genes = np.concatenate(self.genes_chr,axis=None)
        with open(f'{self.load_data_dir}/{configs.rd_mode}/pos.pkl', 'rb') as f:
                self.pos_chr = pickle.load(f)
        with open(f'{self.load_data_dir}/{configs.rd_mode}/snp_train.pkl', 'rb') as f:
                snp_data_train_ = pickle.load(f)
        with open(f'{self.load_data_dir}/{configs.rd_mode}/snp_test.pkl', 'rb') as f:
                snp_data_test = pickle.load(f)
        with open(f'{self.load_data_dir}/{configs.rd_mode}/label_train.pkl', 'rb') as f:
                train_labels = pickle.load(f)
        with open(f'{self.load_data_dir}/{configs.rd_mode}/label_test.pkl', 'rb') as f:
                test_labels = pickle.load(f)
        with open(f'{self.load_data_dir}/{configs.rd_mode}/covar_train.pkl', 'rb') as f:
                train_covs_ = pickle.load(f)
        with open(f'{self.load_data_dir}/{configs.rd_mode}/covar_test.pkl', 'rb') as f:
                covs_test = pickle.load(f) 
                  
        if configs.non_add:
            with open(f'genes_analysis/{label}/non_add_genes.pkl', 'rb') as f:
                non_add_genes = pickle.load(f)
            mask = [np.isin(i, list(non_add_genes)) for i in self.genes_chr]
            snp_data_train_ = [t[:,m] for t,m in zip(snp_data_train_,mask)]
            snp_data_test = [t[:,m] for t,m in zip(snp_data_test,mask)] 
            self.genes_chr = [g[m] for g,m in zip(self.genes_chr,mask)]
            self.genes = np.concatenate(self.genes_chr,axis=None)
            
        if len(configs.use_sim)>0:
            with open(f'{self.load_data_dir}/{configs.rd_mode}/sim/label_train_sim_{configs.use_sim}.pkl', 'rb') as f:
                train_labels = pickle.load(f)
            with open(f'{self.load_data_dir}/{configs.rd_mode}/sim/label_test_sim_{configs.use_sim}.pkl', 'rb') as f:
                test_labels = pickle.load(f)
                
        assert all((i == train_labels[0]).all() for i in train_labels)
        assert all((i == test_labels[0]).all() for i in test_labels)

        if not self.cont:
            train_labels_ = train_labels[0]-1
            test_labels = test_labels[0]-1
            train_idx, val_idx = train_test_split(np.arange(len(train_labels_)),
                                                test_size=0.15,
                                                shuffle=True,
                                                stratify=train_labels_,
                                                random_state=333)    
        else:
            train_labels_ = train_labels[0]
            test_labels = test_labels[0]
            train_idx, val_idx = train_test_split(np.arange(len(train_labels_)),
                                            test_size=0.15,
                                            shuffle=True,
                                            random_state=333)      
                                      
        snp_data_train = [chr_data[train_idx] for chr_data in snp_data_train_]
        snp_data_val = [chr_data[val_idx] for chr_data in snp_data_train_]
        covs_train,covs_val= train_covs_[train_idx],train_covs_[val_idx]
        train_labels,val_labels = train_labels_[train_idx], train_labels_[val_idx]



        if configs.scaling:
            if configs.snp_embed == 'cov':
                snp_data_train,cm,cs = std_scaling_chr(snp_data_train)
                snp_data_val,_,_ = std_scaling_chr(snp_data_val,cm,cs)
                snp_data_test,_,_ = std_scaling_chr(snp_data_test,cm,cs)
                self.snp_cm, self.snp_cs = cm,cs
            covs_train,cm,cs = std_scaling_chr([covs_train])
            covs_train = covs_train[0]
            covs_val = std_scaling_chr([covs_val],cm,cs)[0][0]
            covs_test = std_scaling_chr([covs_test],cm,cs)[0][0]
            self.cov_cm, self.cov_cs = cm, cs
            if self.cont:
                label_train,cm,cs = std_scaling_chr([train_labels])
                train_labels = label_train[0]
                val_labels = std_scaling_chr([val_labels],cm,cs)[0][0]
                test_labels = std_scaling_chr([test_labels],cm,cs)[0][0]
                self.label_mean_std = (cm[0],cs[0])
        
        self.train_data = (snp_data_train,covs_train,train_labels)
        self.val_data = (snp_data_val,covs_val,val_labels)
        self.test_data = (snp_data_test,covs_test,test_labels)

    def shuffle_targets(self):
         if self.shuffle:
              self.train_data = (self.train_data[0], self.train_data[1], np.random.permutation(self.train_data[2]))
         
    def setup(self, stage: str):
        #Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.snp_train, self.snp_val = SNPCHRDataset(self.train_data,self.data_mode), SNPCHRDataset(self.val_data,self.data_mode)
        if stage == "test":
            self.snp_test  = SNPCHRDataset(self.test_data,self.data_mode)
        if stage == "predict":
            self.snp_predict = SNPCHRDataset(self.test_data,self.data_mode)

    def train_dataloader(self):
        return DataLoader(self.snp_train,num_workers=self.num_workers,shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.snp_val,num_workers=self.num_workers,batch_size=self.batch_size)

    def test_dataloader(self):
        if self.ve is not None:
            print(f'Validating with external cohort: {self.ve}')
            return self.eval_dataloder(self.ve)
        else:
            return DataLoader(self.snp_test,num_workers=self.num_workers,batch_size=self.batch_size)

    def predict_dataloader(self):
        if self.ve is not None:
            print(f'Validating with external cohort: {self.ve}')
            return self.eval_dataloder(self.ve)
        else:
            return DataLoader(self.snp_predict,num_workers=self.num_workers,batch_size=self.batch_size)

    def eval_dataloder(self,pheno):
        with open(f'{self.load_data_dir}/{self.rd_mode}/snp_{pheno}.pkl', 'rb') as f:
            snp_data = pickle.load(f)
        with open(f'{self.load_data_dir}/{self.rd_mode}/label_{pheno}.pkl', 'rb') as f:
            labels = pickle.load(f)
        with open(f'{self.load_data_dir}/{self.rd_mode}/covar_{pheno}.pkl', 'rb') as f:
            covs = pickle.load(f)
        labels =labels[0]
        if self.scaling:
            covs = std_scaling_chr([covs],self.cov_cm, self.cov_cs)[0][0]
            if self.snp_embed == 'cov':
                snp_data,_,_ = std_scaling_chr(snp_data,self.snp_cm,self.snp_cs)
            if self.cont:
                labels = std_scaling_chr([labels],[self.label_mean_std[0]],[self.label_mean_std[1]])[0][0]
        eval_data = (snp_data,covs,labels)
        eval_data = SNPCHRDataset(eval_data,self.data_mode)
        return DataLoader(eval_data,num_workers=self.num_workers,batch_size=self.batch_size)




class SNPCHRDataset(Dataset):
    def __init__(self,data,data_mode):
        self.data_mode = data_mode
        self.transform = MinMaxScaler()
        if data_mode == 'snps':
            X,y = data[0],data[2]
            if y.ndim < 2:
                y = y[:,None]
            self.y = torch.tensor(y, dtype=torch.float32)
            self.X = [self.convert2tensor(X_chr) for X_chr in X] 
        elif data_mode == 'snps_covs':
            X,cov,y = data[0],data[1],data[2]
            if y.ndim < 2:
                y = y[:,None]
            self.y = torch.tensor(y, dtype=torch.float32)
            self.c = torch.tensor(cov, dtype=torch.float32)
            self.X = [self.convert2tensor(X_chr) for X_chr in X] 
        elif data_mode == 'synthetic':
            pass
            
    def convert2tensor(self,X):
        if X.ndim < 3:
            X = X[:,:,None]
        X = torch.tensor(X, dtype=torch.float32)
        return X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.data_mode == 'snps' or self.data_mode == 'snps_edl':
            return [X_chr[idx] for X_chr in self.X],self.y[idx]
        elif self.data_mode == 'snps_covs':
            x_all = [X_chr[idx] for X_chr in self.X]
            x_all.append(self.c[idx])
            return x_all,self.y[idx]

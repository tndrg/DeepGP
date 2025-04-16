# -*- coding: utf-8 -*-
"""
@author: Taiyu Zhu
"""

import os
import torch
import random
import csv
import json
import datetime
import pickle
import numpy as np
import pytorch_lightning as pl


from sklearn import metrics
from models import DeepGP
from utils import SNPPCACHRDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from args_generator import args_initial 



fix_seed = 33
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

base_folder = ''

model_dict = {'DeepGP':DeepGP }

# define the fieldnames of results to record
fieldnames = ['pheno','accuracy','precision','recall', 'specificity','f1','auc','mcc']

#%% set parameters
configs = args_initial()
configs.data_dir = base_folder+'pukb/genes'
print('Running experiment for the whole genome')

#%% construct dataloaders
print("Loading data...")
snpdata = SNPPCACHRDataModule(configs)
configs.enc_len = len(snpdata.genes)
configs.enc_len_chr = [len(snp_chr) for snp_chr in snpdata.genes_chr]
configs.pos = snpdata.pos_chr
#%% build a model
if configs.dm == 'snps':
    model = model_dict[configs.mn].SNP_Model_mamba(configs)  
elif configs.dm == 'snps_covs': 
    model = model_dict[configs.mn].SNPCOV_Model_mamba(configs) 

monitor_acc = 'val_auc'

if configs.save_log:
    log_name = f'{configs.mn}_{configs.dm}/{configs.label}'
    if len(configs.use_sim)>0:
         log_name = log_name+'/sim'
    logger = TensorBoardLogger(save_dir=base_folder+'DeepGP/logs/', name=log_name,version=configs.exp_name) #log_graph = True
    checkpoint_callback = ModelCheckpoint(monitor=monitor_acc,save_top_k=1,mode='max')
    callbacks = [EarlyStopping(monitor=monitor_acc, patience=configs.patience,mode="max"), checkpoint_callback]
    ecb = True
else:
    logger,ecb = False, False
    callbacks = [EarlyStopping(monitor=monitor_acc, patience=configs.patience,mode="max")]
    
trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=configs.gpus,
                    callbacks=callbacks,
                    max_epochs=30, 
                    val_check_interval=300,
                    logger=logger,
                    enable_checkpointing = ecb,
                    enable_progress_bar= ecb,
                    ) 

#%% start training
print("Start training...")

trainer.fit(model,snpdata) 

# testing
if configs.save_log:
        best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
else:
    best_model = model

trainer.test(best_model,snpdata) 

#%% evaluation
outputs = trainer.predict(best_model,snpdata)
y_pred_proba = np.concatenate([opt[0].numpy() for opt in outputs])
y_pred = np.concatenate([opt[1].numpy() for opt in outputs])
y_true = np.concatenate([opt[2].numpy() for opt in outputs])

# metrcis
accuracy = metrics.accuracy_score(y_true, y_pred)
precision = metrics.precision_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)
# Specificity = TN / (TN + FP), where TN = true negatives, FP = false positives
tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)
f1 = metrics.f1_score(y_true, y_pred)
mcc = metrics.matthews_corrcoef(y_true, y_pred)

auc = metrics.roc_auc_score(y_true, y_pred_proba)
results_dict = {'pheno': configs.label,'accuracy':accuracy,'precision':precision,
                'recall':recall, 'specificity':specificity,'f1':f1,'auc':auc,'mcc':mcc}


# open a csv to write results
now = datetime.datetime.now().strftime('%m-%d_%H-%M')
if configs.save_results:
    results_dir = base_folder+f'DeepGP/results/{configs.mn}_{configs.dm}/{configs.label}/{configs.exp_name}'
    if len(configs.use_sim)>0:
        results_dir = base_folder+f'DeepGP/results/{configs.mn}_{configs.dm}/{configs.label}/sim/{configs.exp_name}'

    if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    results_csv = f'{results_dir}/results_{now}.csv'
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    # open a json to write parameters
    with open(f'{results_dir}/params_{now}.txt', 'w') as f:
        configs.pos = None
        json.dump(configs.__dict__, f, indent=2)
    with open(results_csv, 'a', newline='') as f:
        # Create the writer
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(results_dict)
    with open(f'{results_dir}/preds.pkl', 'wb') as f:
        pickle.dump((y_true, y_pred_proba),f)

if configs.show:
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Specificity: {specificity}')
    print(f'F1 Score: {f1}')
    print(f'AUC: {auc}')
    print(f'MCC: {mcc}')

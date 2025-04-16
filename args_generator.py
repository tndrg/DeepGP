# -*- coding: utf-8 -*-
"""
@author: Taiyu Zhu
"""

import argparse
PHENO_CONT = False
def args_initial():
    parser = argparse.ArgumentParser()
    # data define
    parser.add_argument('--SNP_len', type=int, default=50, help='length of SNP windows')
    parser.add_argument('--data_dir', type=str, default=None, help='directory of datasets')

    # model define
    parser.add_argument('--mn', type=str, default='DeepGP', help='model name')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input dimension')
    parser.add_argument('--cov_dim', type=int, default=8, help='covariate dimension')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=1, help='num of SNP encoder layers')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in encoder')
    parser.add_argument('--snp_embed', type=str, default='cov', help='embedding method: cov or vocab')
    parser.add_argument('--final_pool', type=str, default='concat', help='output head type: last, concat, atten')
    parser.add_argument('--use_pcs', action='store_false',default=True, help='whether to use population structure')
    parser.add_argument('--bi_direct', action='store_true', default=False, help='whether to bi-directional Mamba')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout to avoid overfitting')
    parser.add_argument('--fused_add_norm', action='store_false', default=True, help='whether to use residuals')
    parser.add_argument('--n_heads',type=int, default=8, help='number of heads in MHSA')
    parser.add_argument('--n_fcn',type=int, default=2, help='number of hidden layers in FCN')
    parser.add_argument('--l2',type=float, default=1e-5, help='weight decay (L2) in Adam')

    # training define
    parser.add_argument('--gpus', type=int, default=1, help='gpu devices')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--keep', action='store_false', default=True, help='keep saved model if it exists')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--loss_name',  default=None, help='name of loss function')
    parser.add_argument('--max_epoch', type=int, default=100, help='maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--threshold', type=int, default=0.5, help='threshold for transforming probability to binary')  


    # display
    parser.add_argument('--show', action='store_false', default=True, help='display metrics')

    # experiment
    parser.add_argument('--exp_name', default=None, help='name of experiment')
    parser.add_argument('--save_log', default=True, help='save logs for experiment')
    parser.add_argument('--save_results', default=True, help='save results experiment')
    parser.add_argument('--label', default=None, help='target phenotype of experiment')
    parser.add_argument('--dm', default='snps_covs', help='data mode: snps or snps_covs')
    parser.add_argument('--ve', default=None, help='Name of external validation cohort')
    parser.add_argument('--rd_mode', default='ld_all', help='read data mode')
    parser.add_argument('--patience',type=int, default=10, help='patience number in early stopping')
    parser.add_argument('--use_sim', type=str, default='', help='whether to use simulated phenotype')
    parser.add_argument('--use_cov', action='store_false', default=True, help='whether to use covariate data')
    parser.add_argument('--scaling', action='store_false', default=True, help='whether to scale data')
    parser.add_argument('--non_add', action='store_true', default=False, help='whether to use non-additive genes only')
    parser.add_argument('--shuffle', action='store_true', default=False, help='whether to randomly shuffle data')
    parser.add_argument('--cont_pheno', action='store_true', default=False, help='whether the phenotype is continuous')

    args = parser.parse_args()
    return args
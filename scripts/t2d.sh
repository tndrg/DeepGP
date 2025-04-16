#!/bin/bash

# Model for T2D prediction using SNPs with covariates
python main_genome.py \
  --label T2D \
  --dm snps_covs \
  --rd_mode ld_all \
  --exp_name final_model \
  --snp_embed cov \
  --final_pool atten \
  --batch_size 64 \
  --d_model 32 \
  --e_layers 1 \
  --dropout 0.2 \
  --lr 7e-4 \
  --n_heads 4 \
  --n_fcn 2 \
  --l2 1e-4 \
  --num_workers 32 \



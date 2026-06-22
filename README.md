# DeepGP

**Multimodal deep learning enhances genomic risk prediction for cardiometabolic diseases in UK Biobank**

DeepGP is a deep learning framework for genomic risk prediction of cardiometabolic diseases. The model learns representations from LD-clumped genome-wide SNP dosage data and combines these genomic features with demographic covariates for downstream disease risk prediction.

This repository accompanies the medRxiv preprint:

> Zhu T, Ghose U, Climente-Gonzalez H, Howson JMM, Hu S, Nevado-Holgado A. *Multimodal deep learning enhances genomic risk prediction for cardiometabolic diseases in UK Biobank*. medRxiv. 2025. doi: [10.1101/2025.04.28.25326564](https://doi.org/10.1101/2025.04.28.25326564)

## Repository Scope

This repository provides:

- DeepGP model definitions and training code.
- A PLINK 2 GWAS and LD-clumping script for disease-specific SNP selection.
- The public type 2 diabetes training configuration.
- Phenotype simulation utilities.
- Documentation of the expected derived data layout.

This repository does not include individual-level UK Biobank data, derived genotype matrices, phenotype labels, covariate files, train/test splits, trained model checkpoints, or prediction outputs derived from UK Biobank participants.

## Data Availability

The study uses UK Biobank individual-level genotype and phenotype data under UK Biobank Application Number 53639. These data are subject to UK Biobank access conditions and cannot be redistributed through GitHub.

Researchers with approved UK Biobank access can reproduce the data preparation and model training steps in their own controlled research environment by using the scripts and expected file formats documented below.

- UK Biobank research access: https://www.ukbiobank.ac.uk/enable-your-research
- UK Biobank application used in this study: Application Number 53639

Model weights are not released because they are derived from restricted individual-level UK Biobank data. They are regenerated when approved users run the training pipeline on their prepared data.

## Repository Layout

```text
.
|-- README.md
|-- args_generator.py
|-- main_genome.py
|-- Phenotype_simulator.py
|-- utils.py
|-- layers/
|   |-- Embed.py
|   |-- SelfAttention_Family.py
|   `-- Transformer_EncDec.py
|-- models/
|   |-- BaseModel.py
|   `-- DeepGP.py
`-- scripts/
    |-- gwas_plink.sh
    `-- t2d.sh
```

## Requirements

The code was tested with:

```text
Python 3.10.13
PyTorch 2.1.1
PyTorch Lightning 2.0.8
CUDA 11.8
PLINK v2.00
```

The reported experiments were run on:

```text
CPU: AMD EPYC 7R13 Processor
GPU: NVIDIA A10 Tensor Core GPU
```

## Reproducibility Workflow

The analysis workflow has three main stages:

1. Prepare QC'ed genotype, phenotype, covariate, ancestry, and train/test split files within an approved UK Biobank environment.
2. Run GWAS on the European training set and apply LD clumping to select disease-specific SNPs.
3. Convert selected SNP dosages and covariates into the pickle layout expected by DeepGP, then train and evaluate the model.

The GWAS and LD-clumping step must be performed on training samples only to avoid information leakage from validation or test samples during SNP selection.

## GWAS and LD Clumping

The paper performs GWAS on the European training set followed by LD clumping. The clumping parameters are:

```text
--clump-p1 0.05
--clump-p2 0.05
--clump-r2 0.7
--clump-kb 500
```

Run:

```bash
bash scripts/gwas_plink.sh \
  --pfile /path/to/ukb_qc_autosomes \
  --pheno /path/to/cmd_phenotypes.tsv \
  --pheno-name T2D \
  --keep /path/to/T2D_eur_train.keep \
  --out-dir /path/to/results/gwas/T2D \
  --threads 32 \
  --memory 64000
```

Required inputs:

- `--pfile`: PLINK 2 `pgen/pvar/psam` prefix for QC'ed imputed autosomal genotypes.
- `--pheno`: phenotype file containing `FID`, `IID`, and one or more phenotype columns.
- `--pheno-name`: phenotype column used for the GWAS, for example `T2D`.
- `--keep`: two-column `FID IID` file containing European training samples only.
- `--out-dir`: output directory for GWAS and clumping results.

For binary case/control phenotypes, the script uses PLINK logistic regression with Firth fallback:

```bash
--glm hide-covar firth-fallback cols=+a1freq
```

This keeps the standard logistic result where it is stable and falls back to Firth regression for variants affected by quasi-complete separation. The allele-frequency column is included for quality control and downstream checks. PLINK may write a `.glm.logistic.hybrid` output file when Firth fallback is used; the script detects PLINK `.glm` outputs automatically.

The script writes:

- GWAS association output from PLINK.
- LD-clumping output from PLINK.
- A `*.snplist` file containing selected tag SNP IDs.

## DeepGP Input Format

After SNP selection, convert the selected dosage matrices and covariates into the pickle files consumed by `SNPPCACHRDataModule`.

For each phenotype and read mode, DeepGP expects:

```text
<data_dir>/<phenotype>/<rd_mode>/
|-- genes.pkl
|-- pos.pkl
|-- snp_train.pkl
|-- snp_test.pkl
|-- label_train.pkl
|-- label_test.pkl
|-- covar_train.pkl
`-- covar_test.pkl
```

Expected contents:

- `genes.pkl`: list of selected SNP IDs grouped by chromosome.
- `pos.pkl`: list of SNP genomic positions grouped by chromosome.
- `snp_train.pkl` and `snp_test.pkl`: selected SNP dosage arrays grouped by chromosome.
- `label_train.pkl` and `label_test.pkl`: phenotype labels.
- `covar_train.pkl` and `covar_test.pkl`: covariate matrices used by SNP-plus-covariate models.

For external validation cohorts, add:

```text
snp_<cohort>.pkl
label_<cohort>.pkl
covar_<cohort>.pkl
```

## Model Training

The example trains the SNP-plus-covariate DeepGP model for type 2 diabetes:

```bash
bash scripts/t2d.sh
```

This calls `main_genome.py` with:

```text
--label T2D
--dm snps_covs
--rd_mode ld_all
--snp_embed cov
--final_pool atten
```

By default, `main_genome.py` reads data from:

```text
pukb/genes/<label>/<rd_mode>/
```

To use another location, pass `--data_dir` to `main_genome.py` or update the path in the shell script.

## Outputs

When logging and result saving are enabled, DeepGP writes:

```text
DeepGP/logs/
DeepGP/results/
```

The results directory contains evaluation metrics, model parameters, and predicted probabilities for the test set.

## Phenotype Simulation

`Phenotype_simulator.py` contains utilities for simulating phenotypes with additive genetic effects, gene-by-gene interactions, and gene-by-environment interactions. The script is intended for use after loading approved genotype dosage arrays and SNP IDs into `snp_data_train` and `snps_ids_chr`.

## Acknowledgements

We thank the developers of the following open-source projects:

- [Mamba](https://github.com/state-spaces/mamba)
- [Vision Mamba](https://github.com/hustvl/Vim)

This research was conducted using the UK Biobank Resource under Application Number 53639.

## License

BSD 3-Clause License

Copyright (c) 2025, University of Oxford and Novo Nordisk. All rights reserved.

## Citation

```bibtex
@article{zhu2025deepgp,
  title = {Multimodal deep learning enhances genomic risk prediction for cardiometabolic diseases in UK Biobank},
  author = {Zhu, Taiyu and Ghose, Upamanyu and Climente-Gonzalez, Hector and Howson, Joanna M. M. and Hu, Sile and Nevado-Holgado, Alejo},
  year = {2025},
  doi = {10.1101/2025.04.28.25326564},
  publisher = {medRxiv}
}
```

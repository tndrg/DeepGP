
"""
 * Author: ZTZU
 * Date: 24 Jan 2025
 * Description: Phenotype simulation
"""

import numpy as np
import pandas as pd
import argparse
import time
import json

from datetime import datetime
from scipy import stats

base_folder = ''

def calculate_heritability(variants_df, y):
    # Calculate VA
    VA = np.sum(2 * variants_df['eff']**2 * variants_df['maf'] * (1 - variants_df['maf']))
    
    # Calculate VP for continuous trait
    VP_continuous = np.var(y)
    
    # Calculate h^2 for continuous trait
    h2_continuous = VA / VP_continuous
    
    return h2_continuous

def set_settings_from_command_line_args():
    """
    Set settings in the SettingsSingleton (S) based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Set simulation parameters.")
    parser.add_argument("--name", type=str, default="", help="Name of the simulation.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Threshold for MAF rare vs. common.")
    parser.add_argument("--num_sim", type=int, default=1, help="Number of simulations.")
    parser.add_argument("--out", type=str, default="./out", help="Path to where output should be saved.")
    parser.add_argument("--h2_snp", type=float, default=0.1, help="Heritability of SNPs")
    parser.add_argument("--pi", type=float, default=0.01, help="Proportion of of causal SNPs")
    parser.add_argument("--sigma_gxg", type=float, default=0.05, help="Variance of gene-gene interactions")
    parser.add_argument("--a_gxe", type=float, default=0.02, help="Stength of gene-environment interactions")
    parser.add_argument("--save_out", type=bool, default=False, help="Whether to save simualtion results")
    args = parser.parse_args()

    return args
    

        
def get_maf(dosage):
    p_j = np.mean(dosage) / 2       
    return min(p_j,1-p_j)

def select_variants(snps, idx_names, pi):

    all_snps_ids = np.concatenate(idx_names,axis=None)
    selected_var_num = int(len(all_snps_ids)*pi)
    sel_ids_all = np.random.choice(all_snps_ids, selected_var_num, replace=False)

    sel_dos = []
    sel_ids = []
    for snps_chrs,id_chr in zip(snps,idx_names):
         sel_dos.append(snps_chrs[:,np.isin(id_chr,sel_ids_all)])
         sel_ids.append(id_chr[np.isin(id_chr,sel_ids_all)])
    
    result = {
        'selected_variants_ids': np.concatenate(sel_ids,axis=None),
        'selected_variants_dos':np.concatenate(sel_dos,axis=1)
    }
    return result


def generate_w_ij(x_ij, p_j):
    return (x_ij - 2 * p_j) / np.sqrt(2 * p_j * (1 - p_j))

def simulate_GA2(h2_snp, b_j1, b_j2, b_j3, x_ij1, x_ij2, x_ij3):
    x_ij1,x_ij2,x_ij3  = x_ij1.transpose(),x_ij2.transpose(),x_ij3.transpose()
    w_ij1 = generate_w_ij(x_ij1, np.mean(x_ij1, axis=0) / 2)
    w_ij2 = generate_w_ij(x_ij2, np.mean(x_ij2, axis=0) / 2)
    w_ij3 = generate_w_ij(x_ij3, np.mean(x_ij3, axis=0) / 2)
    g_i = np.dot(w_ij1, b_j1) + np.dot(w_ij2, b_j2) + np.dot(w_ij3, b_j3)
    return g_i, (w_ij1,w_ij2,w_ij3), (b_j1,b_j2,b_j3)


def simulate_phenotype_GA2(variants, S):
    h2_snp = S.h2_snp
    selected = variants['selected_variants_ids']
    selected_dos = variants['selected_variants_dos']
    assert len(selected) == len(selected_dos[0])

    y = np.zeros(selected_dos.shape[0])

    m_c1 = int(len(selected) * 0.93)
    m_c2 = int(len(selected) * 0.05)
    m_c3 = int(len(selected) * 0.02)
    b_j1 = np.random.normal(0, np.sqrt(0.6 * h2_snp / m_c1), m_c1)
    b_j2 = np.random.normal(0, np.sqrt(0.2 * h2_snp / m_c2), m_c2)
    b_j3 = np.random.normal(0, np.sqrt(0.2 * h2_snp / m_c3), m_c3)
    x_ij1, x_ij2, x_ij3 = [], [], []

    tmp = []
    for idx, variant_chr in enumerate(selected):
        name = variant_chr
        geno_int_out = selected_dos[:,idx]
        maf = get_maf(geno_int_out)
        var_type = 'rare' if maf <= S.threshold else 'common'

        if idx < m_c1:
            b_j = b_j1
            x_ij1.append(geno_int_out)
        elif idx < m_c1 + m_c2:
            b_j = b_j2
            x_ij2.append(geno_int_out)
        elif idx < m_c1 + m_c2 + m_c3:
            b_j = b_j3
            x_ij3.append(geno_int_out)
        else:
            break
        tmp.append(pd.DataFrame([{'variant_id': name, 'type': var_type, 'eff': b_j[idx % len(b_j)], 'maf': maf}]))
        variants_table = pd.concat(tmp, ignore_index=True)

     # Generate genetic architecture effects (Shrestha et al. 2023)    
    g,w,_ = simulate_GA2(h2_snp, b_j1, b_j2, b_j3, np.array(x_ij1), np.array(x_ij2), np.array(x_ij3))


    # Generate GxG interaction effects (Fu et al. 2023)
    gxg = 0
    w_all = np.concatenate((w[0],w[1],w[2]),axis=1)
    sel_w = np.random.choice(m_c3, S.num_gxg, replace=False)+m_c2+m_c1
    M = len(selected)
    for ind_w in sel_w:
        X_t = w_all[:,ind_w]
        X_mt = np.delete(w_all, ind_w, 1)
        E_t = X_t.reshape(-1, 1)*X_mt
        alpha_t = np.random.normal(0, np.sqrt(S.sigma_gxg / (M - 1)), M - 1)
        gxg = gxg+ np.dot(E_t, alpha_t)

    # Generate GxE interaction effects (Kerin et al. 2020)
    tau = np.random.normal(0, np.sqrt(S.h2_tau/M), M)
    s_eps = np.random.normal(0, np.sqrt((1-S.h2_tau)), selected_dos.shape[0])
    SE = np.dot(w_all, tau) + s_eps
    gxe = S.a_gxe*(SE**2)


    # Generate GxG interaction effects (Fu et al. 2023)
    M = len(selected)
    alpha_t = np.random.normal(0, np.sqrt(S.sigma_gxg / (M - 1)), M - 1)
    E_t = np.random.randn(len(y), M - 1)
    y = y+ np.dot(E_t, alpha_t)

    cov_e =  1-(np.var(g)+np.var(gxg)+np.var(gxe))
    e = 0
    if cov_e>0:
        e = np.random.normal(0, np.sqrt(cov_e), len(geno_int_out))
    y = g+gxg+gxe+e

    result = {'variants_data': variants_table, 'y': y}     
    return result

if __name__ == '__main__':
    # User-defined variables are stored in a singleton
    S = set_settings_from_command_line_args()
    if len(S.name) <= 0:
        now = datetime.now().strftime("%m-%d-%Y-%H-%M")
        S.name = f"simulation_{now}"

    # read SNP data and covar
    snp_data_train = None
    snps_ids_chr = None
        

    print(f"Drawing {S.num_sim} simulations...", end=' ')
    start_time = time.time()
    simulated_phenos = pd.DataFrame()
    
    with open(f"{S.out}/{S.name}.json", "w") as json_file:
        for cnt in range(1, S.num_sim + 1):
            start_time = time.time()
            print(f"Performing simulation {cnt} of {S.num_sim}...")
            print(f"\t - selecting variants")
            variants = select_variants(snp_data_train,snps_ids_chr, S.pi)
            print(f"\t - simulating phenotype")
            sim_result = simulate_phenotype_GA2(variants, S)
            if S.save_out:
                json.dump(sim_result['variants_data'].to_dict(orient='records'), json_file, indent=4)
                json_file.write("\n")
            column_name = f'y_{cnt}'
            simulated_phenos[column_name] = sim_result['y']
            exec_time = round(time.time() - start_time, 2)
            print(f"\t - done in {exec_time}s.")
            h2c = calculate_heritability(sim_result['variants_data'],sim_result['y'])
            print(f'Simulated heritability: {h2c:.3f},the settings of heritability {S.h2_snp}')

    if S.save_out:
        with open(f"{S.out}/phenotypes_{S.name}.csv", "w") as csv_file:
            simulated_phenos.to_csv(csv_file, header=True, index=False)


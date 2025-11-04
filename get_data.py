import os 
import pandas as pd
import numpy as np 

genomic_sites = "./wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
tf_pwm = "./factorbookMotifPwm.txt"
pos_tf_bind = "./factorbookMotifPos.txt"

def get_all_dna_seq(genomic_sites):
    """
    This function reads a BED file containing genomic sites and returns a DataFrame with the data.
    
    Parameters:
    genomic_sites (str): Path to the BED file.
    
    Returns:
    pd.DataFrame: DataFrame containing the genomic sites data.
    """
    # Read the BED file into a DataFrame
    df = pd.read_csv(genomic_sites, sep="\t", header=None)
    
    df.columns = ['chrom', 'start', 'end', 'name']
    
    genom_seq = {
        'name': df['name'].tolist(),
        'seq' : [],
    }
    
    for _, site in df.iterrows():
        seq = get_dna_seq(site)
        genom_seq['seq'].append(seq)
        
    
    
    return df
    
    
def get_dna_seq(site):
    chrom = site['chrom']
    start = site['start']
    end = site['end']
    
    with open(f"./{chrom}.fa", "r") as f:
        lines = f.readlines()
        sequence = ''.join(lines[1:]).replace('\n', '')
        dna_seq = sequence[start:end]
    return dna_seq


def get_TF_DF():
    pwm_rows = []
    with open(tf_pwm, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("\t")
            tf_name = parts[0]
            tf_length = int(parts[1])
            print(tf_name, tf_length)
            numbers_str = "\t".join(parts[2:]).strip()
            
            base_groups = numbers_str.split()
            if len(base_groups) != 4 :
                raise ValueError(f"Expected 4 base groups, got {len(base_groups)}")
            
            A = [float(x) for x in base_groups[0].split(",") if x]
            C = [float(x) for x in base_groups[1].split(",") if x]
            G = [float(x) for x in base_groups[2].split(",") if x]
            T = [float(x) for x in base_groups[3].split(",") if x]
            
            pwm_rows.append({
                'tf_name': tf_name,
                'tf_length': tf_length,
                'A': A,
                'C': C,
                'G': G,
                'T': T,
            })
            
    pwm_df = pd.DataFrame(pwm_rows)
    return pwm_df
    
    
    

def get_score(seq, tf_df, TF):
    row = tf_df[tf_df['tf_name'] == TF].iloc[0]
    seq = seq.upper()
    A_list = row['A']
    C_list = row['C']
    G_list = row['G']
    T_list = row['T']
    
    if len(seq) < row['tf_length']:
        raise ValueError("Sequence length is shorter than TF motif length.")
    
    max_score = -np.inf
    score = 1
    #need to check all positions 0 - len(seq) - tf_length
    for i in range(len(seq) - row['tf_length'] + 1):
        window = seq[i:i+row['tf_length']]
        for j in range(len(window)):
            if window[j] == 'A':
                score += np.log(A_list[j])
            elif window[j] == 'C':
                score += np.log(C_list[j])
            elif window[j] == 'G':
                score += np.log(G_list[j])
            elif window[j] == 'T':
                score += np.log(T_list[j])
        if score > max_score:
            max_score = score        
        
    return max_score


def main():
    tf_df = get_TF_DF()
    site_df = {
    'chrom': ['chr1', 'chr1', "chr1"],
    'start': [91265, 91419, 91421], 	
    'end': [91280, 91434, 91436],
    }
    for _, site in pd.DataFrame(site_df).iterrows():
        seq = get_dna_seq(site)
        print(seq)
        print(get_score(seq, tf_df, "CTCF"))
        

if __name__ == "__main__":
    main()
    
        
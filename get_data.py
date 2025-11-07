import pandas as pd
import numpy as np 
import genome_data # importing genome data, such as chromosomes and tf_df = [tf_name, tf_length, A, C, G, T] with log_pwm values for each base
from concurrent.futures import ProcessPoolExecutor, as_completed

genomic_sites = "./wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
pos_tf_bind = "./factorbookMotifPos.txt"


def get_all_dna_seq(genomic_sites, chroms = genome_data.chromosomes):
    # Load chromosome sequences into global variables
    df = pd.read_csv(genomic_sites, sep="\t", header=None)
    
    df.columns = ['chrom', 'start', 'end', 'name']
    def get_seq(row):
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        sequence = chroms[chrom][start:end]
        return sequence.upper()
    
    df['seq'] = df.apply(get_seq, axis=1)
    df['id'] = df['chrom'].astype(str) + ':' + df['start'].astype(str) + '-' + df['end'].astype(str)
    return df[['id', 'seq']]

    
def get_score(seq, row_tf):
    A_list = row_tf['A']
    C_list = row_tf['C']
    G_list = row_tf['G']
    T_list = row_tf['T']
    tf_length = row_tf['tf_length']
    
    if len(seq) < tf_length:
        return -np.inf, None
    
    max_score = -np.inf
    max_index = None
    #need to check all positions 0 - len(seq) - tf_length
    for i in range(len(seq) - tf_length + 1):
        score = 0
        window = seq[i:i+tf_length]
        for j in range(len(window)):
            if window[j] == 'A':
                score += A_list[j]
            elif window[j] == 'C':
                score += C_list[j]
            elif window[j] == 'G':
                score += G_list[j]
            elif window[j] == 'T':
                score += T_list[j]
            else :
                score = -np.inf
            if score == -np.inf:
                break
                
        if score > max_score:
            max_score = score 
            max_index = i       
        
    return max_score, max_index

def score_single_tf(seqs, row_tf):
    tf_name = row_tf['tf_name']
    results = []
    for seq in seqs:
        score, _ = get_score(seq, row_tf)
        results.append(score)
    return tf_name, results

def parallel_score_all_tfs(seqs, tf_df, max_workers=10):
    scores = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_tf = {
            executor.submit(score_single_tf, seqs, row): row['tf_name'] for _, row in tf_df.iterrows()
        }
        for future in as_completed(future_to_tf):
            tf_name, tf_scores = future.result()
            scores[tf_name] = tf_scores
    return pd.DataFrame(scores) 

def main():
    tf_df = genome_data.tf_df
    id_seq_df = get_all_dna_seq(genomic_sites)
    print("Scoring all TFs across all sequences in parallel...")
    tf_scores = parallel_score_all_tfs(id_seq_df['seq'].tolist(), tf_df, max_workers=8)
    combined_df = pd.concat([id_seq_df, tf_scores], axis=1)
    combined_df.to_csv("tf_binding_scores.csv", index=False)
    

if __name__ == "__main__":
    main()
    
        
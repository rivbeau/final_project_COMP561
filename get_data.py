import pandas as pd
import numpy as np 
import genome_data

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
        return -np.inf
    
    max_score = -np.inf
    # max_index = None
    #need to check all positions 0 - len(seq) - tf_length
    for i in range(len(seq) - tf_length + 1):
        score = 0
        window = seq[i:i+tf_length]
        with np.errstate(divide='raise'): #so it raises an error for log 0 which try will catch
            for j in range(len(window)):
                try:
                    if window[j] == 'A':
                        score += np.log(A_list[j])
                    elif window[j] == 'C':
                        score += np.log(C_list[j])
                    elif window[j] == 'G':
                        score += np.log(G_list[j])
                    elif window[j] == 'T':
                        score += np.log(T_list[j])
                except (FloatingPointError):
                    score = -np.inf
                    break  # No need to continue if we hit a log(0)
        if score > max_score:
            max_score = score 
            # max_index = i       
        
    return max_score

def score_all_tfs(seq, tf_df):
    scores = {}
    for _, row in tf_df.iterrows():
        row = row
        tf_name = row['tf_name']
        score = get_score(seq, row)
        scores[tf_name] = score
    return pd.Series(scores)  

def main():
    tf_df = genome_data.tf_df
    id_seq_df = get_all_dna_seq(genomic_sites)
    print(id_seq_df.head())
    tf_scores = id_seq_df['seq'].apply(lambda seq: score_all_tfs(seq, tf_df))
    combined_df = pd.concat([combined_df, tf_scores], axis=1)
    combined_df.to_csv("tf_binding_scores.csv", index=False)
    
    

    # for _, site in pd.DataFrame(site_df).iterrows():
    #     seq = get_dna_seq(site)
    #     print(seq)
    #     print(get_score(seq, tf_df.iloc[0]))
        
if __name__ == "__main__":
    main()
    
        
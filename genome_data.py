import glob, os, pandas as pd, numpy as np

chromosomes = {}
tf_pwm = "./factorbookMotifPwm.txt"

def load_chromosomes():
    chrom_fasta = glob.glob("*.fa")
    for fasta_file in chrom_fasta:
        chrom_name = os.path.basename(fasta_file).replace('.fa', '')
        with open(fasta_file, "r") as f:
            lines = f.readlines()
            sequence = ''.join(lines[1:]).replace('\n', '')
            chromosomes[chrom_name] = sequence
            
def load_tf_df():
    pwm_rows = []
    with open(tf_pwm, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("\t")
            tf_name = parts[0]
            tf_length = int(parts[1])
            numbers_str = "\t".join(parts[2:]).strip()
            
            base_groups = numbers_str.split()
            if len(base_groups) != 4 :
                raise ValueError(f"Expected 4 base groups, got {len(base_groups)}")
            
            # Convert probabilities to log-odds scores for easier handling, and not additional recomputing later
            A = [np.log(float(x)) if float(x) > 0 else -np.inf for x in base_groups[0].split(",") if x]
            C = [np.log(float(x)) if float(x) > 0 else -np.inf for x in base_groups[1].split(",") if x]
            G = [np.log(float(x)) if float(x) > 0 else -np.inf for x in base_groups[2].split(",") if x]
            T = [np.log(float(x)) if float(x) > 0 else -np.inf for x in base_groups[3].split(",") if x]
            
            
            
            
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
tf_df = load_tf_df()
load_chromosomes()
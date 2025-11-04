import glob, os, pandas as pd

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
tf_df = load_tf_df()
load_chromosomes()
# get_data.py

import pandas as pd
import numpy as np
import genome_data  # chromosomes + tf_df
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------------- CONFIG ----------------
genomic_sites = "./wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
pos_tf_bind   = "./factorbookMotifPos.txt"

TARGET_TF = "CTCF"   # <- choose your TF here
WINDOW    = 100      # fixed window length (odd number is nice for centering)
EXTRA = ""        #none or test for trying MLP/SVM models 

# If you want to work on a subset while debugging, set this e.g. to 5000
MAX_POSITIVE_SITES = None   # None or an int like 5000
# ----------------------------------------


genome_data.load_chromosomes()
tf_df = genome_data.load_tf_df()
chroms = genome_data.chromosomes


# ---------- helper: sequence for one window ----------

def get_seq_for_window(chrom, start, end, chroms=chroms):
    """Extract upper-case sequence for [start, end) on given chromosome."""
    seq = chroms[chrom][start:end]
    return seq.upper()


# ---------- POSITIVE WINDOWS (label=1) ----------

def load_positive_windows(tf_name=TARGET_TF, window=WINDOW):
    """
    Build fixed-length windows centered on known binding sites for tf_name
    using factorbookMotifPos.txt.
    """
    cols = ["ignore", "chrom", "start", "end", "tf_name", "score", "strand"]
    motif_df = pd.read_csv(pos_tf_bind, sep="\t", header=None, names=cols)

    motif_tf = motif_df[motif_df["tf_name"] == tf_name].copy()
    print(f"Found {len(motif_tf)} binding sites for {tf_name}")

    # Optional subsampling for faster debugging
    if MAX_POSITIVE_SITES is not None and len(motif_tf) > MAX_POSITIVE_SITES:
        motif_tf = motif_tf.sample(MAX_POSITIVE_SITES, random_state=42)
        print(f"Subsampled to {len(motif_tf)} positive sites")

    # Build windows with progress bar
    windows = []
    print("Building positive windows...")
    for _, row in tqdm(motif_tf.iterrows(), total=len(motif_tf), desc="Pos windows"):
        chrom = row["chrom"]
        if chrom not in chroms:
            continue  # skip chromosomes we don't have

        center = (int(row["start"]) + int(row["end"])) // 2
        half   = window // 2
        start  = max(0, center - half)
        end    = start + window

        # Safety: do not go past chromosome end
        if end > len(chroms[chrom]):
            end = len(chroms[chrom])
            start = end - window
            if start < 0:
                continue

        windows.append({
            "chrom": chrom,
            "start": int(start),
            "end":   int(end),
            "strand": row["strand"]
        })

    pos_windows = pd.DataFrame(windows).drop_duplicates()

    pos_windows["label"] = 1

    # extract sequence with progress bar
    print("Extracting sequences for positive windows...")
    pos_windows["seq"] = [
        get_seq_for_window(row["chrom"], row["start"], row["end"])
        for _, row in tqdm(pos_windows.iterrows(), total=len(pos_windows), desc="Pos seq")
    ]

    return pos_windows[["chrom", "start", "end", "seq", "label"]]


# ---------- NEGATIVE WINDOWS (label=0) ----------

def load_reg_regions():
    reg_cols = ["chrom", "start", "end", "name"]
    reg_df = pd.read_csv(genomic_sites, sep="\t", header=None, names=reg_cols)
    return reg_df


def sample_negative_windows(pos_windows, reg_df, window=WINDOW, n_neg=None, seed=0):
    """
    Sample negative windows from regulatory regions that do NOT overlap
    any positive window.
    """
    rng = np.random.default_rng(seed)

    if n_neg is None:
        n_neg = len(pos_windows)

    # Map positives by chromosome for fast overlap checks
    pos_by_chrom = {}
    for chrom in pos_windows["chrom"].unique():
        sub = pos_windows[pos_windows["chrom"] == chrom]
        pos_by_chrom[chrom] = sub[["start", "end"]].values

    neg_rows = []

    reg_df = reg_df[reg_df["chrom"].isin(chroms.keys())].reset_index(drop=True)

    attempts = 0
    max_attempts = n_neg * 50  # just to avoid infinite loops

    print(f"Sampling {n_neg} negative windows...")
    from tqdm.auto import tqdm as tqdm_auto
    with tqdm_auto(total=n_neg, desc="Neg windows") as pbar:
        while len(neg_rows) < n_neg and attempts < max_attempts:
            attempts += 1

            row = reg_df.sample(1).iloc[0]
            chrom = row["chrom"]
            reg_start = int(row["start"])
            reg_end   = int(row["end"])
            length    = reg_end - reg_start
            if length < window:
                continue

            start = rng.integers(reg_start, reg_end - window)
            end   = start + window

            # check overlap with positives
            overlap = False
            if chrom in pos_by_chrom:
                for s, e in pos_by_chrom[chrom]:
                    if not (end <= s or start >= e):  # intervals intersect
                        overlap = True
                        break
            if overlap:
                continue

            # within chromosome
            if end > len(chroms[chrom]):
                continue

            neg_rows.append({"chrom": chrom, "start": int(start), "end": int(end)})
            pbar.update(1)

    neg_df = pd.DataFrame(neg_rows)
    neg_df["label"] = 0

    print("Extracting sequences for negative windows...")
    neg_df["seq"] = [
        get_seq_for_window(row["chrom"], row["start"], row["end"])
        for _, row in tqdm(neg_df.iterrows(), total=len(neg_df), desc="Neg seq")
    ]

    print(f"Sampled {len(neg_df)} negative windows (target was {n_neg})")
    return neg_df[["chrom", "start", "end", "seq", "label"]]


# ---------- OLD PWM SCORING CODE (unchanged) ----------
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
    for i in range(len(seq) - tf_length + 1):
        score = 0.0
        window = seq[i:i+tf_length]
        for j, base in enumerate(window):
            if base == 'A':
                score += A_list[j]
            elif base == 'C':
                score += C_list[j]
            elif base == 'G':
                score += G_list[j]
            elif base == 'T':
                score += T_list[j]
            else:
                score = -np.inf
                break
        if score > max_score:
            max_score = score
            max_index = i

    return max_score, max_index


# ---------- MAIN: build dataset for one TF ----------

def main():
    # 1) positives and negatives
    pos_windows = load_positive_windows(TARGET_TF, WINDOW)
    reg_df      = load_reg_regions()
    neg_windows = sample_negative_windows(pos_windows, reg_df, WINDOW, n_neg=len(pos_windows))

    dataset = pd.concat([pos_windows, neg_windows], ignore_index=True)

    # 2) (optional) add PWM score of TARGET_TF as a feature
    tf_row = tf_df[tf_df["tf_name"] == TARGET_TF].iloc[0]

    print("Computing PWM scores for all windows...")
    pwm_scores = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="PWM scores"):
        seq = row["seq"]
        score, _ = get_score(seq, tf_row)
        pwm_scores.append(score)
    dataset["pwm_score"] = pwm_scores

    # 3) save dataset
    out_name = f"tf_dataset_{TARGET_TF}_{WINDOW}_{EXTRA}bp.csv"
    dataset.to_csv(out_name, index=False)
    print(f"Saved dataset to {out_name}")
    print(dataset.head())


if __name__ == "__main__":
    main()
# shape_features.py
#
# Extract DNA shape features (MGW, ProT, Roll, HelT)
# from hg19 bigWig tracks for fixed-length windows.
#
# Includes tqdm progress bars for long-running processes.

import numpy as np
import pandas as pd
import pyBigWig 
from tqdm import tqdm


# ---- Paths to bigWig files (adjust if yours are elsewhere) ----
BW_PATHS = {
    "MGW":  "hg19.MGW.wig.bw",
    "ProT": "hg19.ProT.wig.bw",
    "Roll": "hg19.Roll.wig.bw",
    "HelT": "hg19.HelT.wig.bw",
    # now the other 9 shape features if needed
    "Rise": "hg19.Rise.wig.bw",
    "Shift": "hg19.Shift.wig.bw",
    "Slide": "hg19.Slide.wig.bw",
    "Tilt": "hg19.Tilt.wig.bw",
    "Buckle": "hg19.Buckle.wig.bw",
    "Shear": "hg19.Shear.wig.bw",
    "Stretch": "hg19.Stretch.wig.bw",
    "Stagger": "hg19.Stagger.wig.bw",
    "Opening": "hg19.Opening.wig.bw",
}


# ------------------------------------------------------------
# 1. Load/open all bigWig tracks once
# ------------------------------------------------------------
def init_shape_tracks(paths: dict = None) -> dict:
    """
    Open bigWig files and return a dict {feature_name: pyBigWig_object}.
    Call this ONCE at the start, reuse the tracks.
    """
    if paths is None:
        paths = BW_PATHS

    tracks = {}
    for name, path in paths.items():
        print(f"Opening shape track: {name} -> {path}")
        bw = pyBigWig.open(path)
        if not bw.isBigWig():
            raise RuntimeError(f"{path} is not a valid .bigWig file")
        tracks[name] = bw
    return tracks


# ------------------------------------------------------------
# 2. Extract shape features for a single (chrom, start, end) window
# ------------------------------------------------------------
def get_shape_for_window(chrom: str, start: int, end: int, bw_tracks: dict) -> dict:
    """
    Extract MGW, ProT, Roll, HelT values for [start, end) on `chrom`.
    Returns dict of feature -> numpy array (length = end-start).
    """
    length = end - start
    shapes = {}

    for feat_name, bw in bw_tracks.items():
        vals = np.array(bw.values(chrom, start, end), dtype=float)

        # Replace NaN/None with 0
        vals = np.nan_to_num(vals, nan=0.0)

        # Ensure length matches
        if vals.shape[0] > length:
            vals = vals[:length]
        elif vals.shape[0] < length:
            vals = np.pad(vals, (0, length - vals.shape[0]), mode="constant", constant_values=0.0)

        shapes[feat_name] = vals

    return shapes


# ------------------------------------------------------------
# 3. Add shape feature arrays to a dataframe
# ------------------------------------------------------------
def add_shape_arrays(df: pd.DataFrame, bw_tracks: dict) -> pd.DataFrame:
    """
    For each row with ['chrom', 'start', 'end'],
    add arrays: 'MGW', 'ProT', 'Roll', 'HelT'.

    Uses tqdm for progress visualization.
    """
    feature_names = list(bw_tracks.keys())
    
    feature_lists = {feat: [] for feat in feature_names}
    print(f"Extracting shape features for {len(df)} windows...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Shape extraction"):
        chrom = row["chrom"]
        start = int(row["start"])
        end   = int(row["end"])

        shapes = get_shape_for_window(chrom, start, end, bw_tracks)

        for feat in feature_names:
            feature_lists[feat].append(shapes[feat])    

    df = df.copy()
    for feat in feature_names:
        df[feat] = feature_lists[feat]

    return df


# ------------------------------------------------------------
# 4. Convert per-row arrays into a feature matrix
# ------------------------------------------------------------
def shapes_to_matrix(df: pd.DataFrame,
                     bw_tracks: dict) -> np.ndarray:
    """
    Convert shape arrays into 2D numpy matrix:
    [MGW_0..MGW_L-1, ProT_0.., Roll_0.., HelT_0...]

    Uses tqdm for progress bar.
    """
    feature_order = list(bw_tracks.keys())
    rows = []

    print("Building 2D feature matrix from shape arrays...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Matrix assembly"):
        parts = []
        for feat in feature_order:
            arr = np.asarray(row[feat], dtype=float)
            parts.append(arr)
        vec = np.concatenate(parts)
        rows.append(vec)

    X = np.vstack(rows)
    return X

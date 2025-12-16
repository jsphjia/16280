import json, random, argparse, os
import numpy as np
from tqdm import tqdm

"""
Apply temporal smoothing to angular velocity labels (omega) in the dataset index.

---------------------------------------------------------------------------
Purpose:
    Smooths the steering command sequence to remove high-frequency noise
    and produce cleaner target labels for learning-based control models.

Functionality:
    1. Loads an existing dataset index file (index.json or index_split.json).
    2. Extracts all "label.omega" values (angular velocities).
    3. Applies a centered moving-average filter with a configurable window size.
    4. Replaces each record’s label.omega with the smoothed value.
    5. Saves the updated records to a new JSON file (e.g., index_smooth.json).

Usage:
    python3 smooth_labels.py --index path/to/index_split.json [--out index_smooth.json] [--win 5]

Example:
    python3 smooth_labels.py --index ./dataset/index_split.json --win 5

Parameters:
    --index : Path to input dataset index.
    --out   : Output filename (optional; default = index_smooth.json in same folder).
    --win   : Half-window size for smoothing kernel (default = 5 frames).

Notes:
    • Implements a normalized moving-average kernel:
          kernel = np.ones(2*win + 1) / (2*win + 1)
    • Uses 'same' convolution mode to preserve array length.
    • Reports mean and max change in ω after smoothing for verification.
    • Helps the model learn smoother, more consistent steering responses.
---------------------------------------------------------------------------
"""


def main():
    # === CONFIG ===
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to folder containing index.json or index_split.json")
    ap.add_argument("--out", default=None, help="Output filename")
    ap.add_argument("--win", default=None, help="Smoothing wnidow size")
    args = ap.parse_args()

    if args.out is None:
        index_dir = os.path.dirname(args.index) or "."
        stem, ext = os.path.splitext(os.path.basename(args.index))
        args.out = os.path.join(index_dir, f"index_smooth{ext}")


    DATA_DIR = args.out   # folder with your index.json
    WINDOW = 5                            # smoothing half-window size (in frames)
    #
    # INPUT_JSON = os.path.join(DATA_DIR, "index_split.json")
    # OUTPUT_JSON = os.path.join(DATA_DIR, "index_smooth.json")
    INPUT_JSON = args.index
    OUTPUT_JSON = args.out
    print(INPUT_JSON)
    print(OUTPUT_JSON)
    # === LOAD ===
    with open(INPUT_JSON, "r") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {INPUT_JSON}")

    # Extract omegas
    omegas = np.array([r.get("label", {}).get("omega", 0.0) for r in records], dtype=float)

    # === SMOOTHING ===
    kernel = np.ones(2 * WINDOW + 1)
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(omegas, kernel, mode="same")

    # === ASSIGN BACK ===
    for i, r in enumerate(records):
        if "label" not in r:
            r["label"] = {}
        r["label"]["omega"] = float(smoothed[i])

    # === SAVE ===
    with open(OUTPUT_JSON, "w") as f:
        json.dump(records, f, indent=2)

    # === REPORT ===
    abs_diff = np.abs(smoothed - omegas)
    print(f"  Smoothed {len(records)} records")
    print(f"  mean |delta(omega)| = {abs_diff.mean():.4f}")
    print(f"  max  |delta(omega)| = {abs_diff.max():.4f}")
    print(f"  output saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

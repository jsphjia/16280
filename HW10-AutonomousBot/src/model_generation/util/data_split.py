#!/usr/bin/env python3
import json, random, argparse, os
from math import floor

"""
Split dataset index (index.json) into train / validation / test subsets.

---------------------------------------------------------------------------
Purpose:
    This script divides the extracted dataset (from the ROS 2 bag) into
    three subsets for supervised learning experiments. It ensures
    reproducibility through a fixed random seed and prevents empty splits
    even for small datasets.

Functionality:
    1. Loads 'index.json' — a list of dataset entries produced by the
       extraction script.
    2. Randomly shuffles all entries using a user-specified seed.
    3. Splits the data into:
           • 70% training
           • 20% validation
           • 10% testing
       (with small-dataset safeguards)
    4. Annotates each record with a new key: "split" ∈ {"train", "val", "test"}.
    5. Writes the updated list to a new file, e.g. 'index_split.json'.

Usage:
    python3 split_index.py --index path/to/index.json [--seed 42] [--out index_split.json]

Example:
    python3 split_index.py --index ./dataset/index.json

Output:
    dataset/index_split.json  → same data as input but with "split" field added.

Notes:
    • The script preserves reproducibility via a deterministic random shuffle.
    • The 70/20/10 ratio is implemented using rounding to ensure the total
      number of samples remains consistent.
    • Useful for maintaining standardized splits across multiple training runs.
---------------------------------------------------------------------------
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to index.json")
    # ap.add_argument("--out", default="index_split.json", help="Output filename")
    ap.add_argument("--out", default=None, help="Output filename")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = ap.parse_args()

    with open(args.index, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input index must be a list of records.")

    if args.out is None:
        index_dir = os.path.dirname(args.index) or "."
        stem, ext = os.path.splitext(os.path.basename(args.index))
        args.out = os.path.join(index_dir, f"{stem}_split{ext}")

    rng = random.Random(args.seed)
    idxs = list(range(len(data)))
    rng.shuffle(idxs)

    n = len(idxs)
    n_train = round(0.70 * n)
    n_val   = round(0.20 * n)
    # ensure total matches n
    n_test  = n - n_train - n_val

    # edge-case guards to avoid empty splits on very small datasets
    if n >= 3:
        if n_train == 0: n_train = 1
        if n_val   == 0: n_val   = 1
        n_test = max(1, n - n_train - n_val)

    train_set = set(idxs[:n_train])
    val_set   = set(idxs[n_train:n_train+n_val])
    test_set  = set(idxs[n_train+n_val:])

    for i, rec in enumerate(data):
        if i in train_set:
            rec["split"] = "train"
        elif i in val_set:
            rec["split"] = "val"
        else:
            rec["split"] = "test"

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Total: {n}  -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()

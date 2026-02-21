"""
Create an atom-type-filtered subset of ASE LMDB files.

This script scans all `.aselmdb` files in `DATASET_PATH` and copies only the
molecules whose atomic numbers are a subset of `ALLOWED_ATOMIC_NUMBERS`
(H, C, N, O, F, Cl by default) into matching output databases in `OUTPUT_DIR`.

Processing is parallelized over database files using Python multiprocessing.
At the end, it prints per-file and global statistics for retained molecules.
"""

import os
from os import makedirs
import numpy as np
from ase.db import connect
from tqdm import tqdm 
from glob import glob
from multiprocessing import Pool, cpu_count

ALLOWED_ATOMIC_NUMBERS = np.array([1, 6, 7, 8, 9, 17], dtype=np.int64)
HAS_ONLY_ALLOWED_ATOMS = False
DATASET_PATH = "/home/karella/Projects/hippynn/dataset/opoly_train"
OUTPUT_DIR = "/home/karella/Projects/hippynn/dataset/opoly_bens_train"
DATASET_WILDCARD = f"{DATASET_PATH}/*.aselmdb"

def process_db_file(db_file):
    """Process a single database file and return statistics."""
    with connect(db_file, readonly=True) as db:
        db_all_count = 0
        db_name = os.path.basename(db_file)
        db_total_count = db.count()
        
        with connect(f"{OUTPUT_DIR}/{db_name}", readonly=False) as out_db:
            for row in db.select(): 
                atomic_numbers = np.unique(row.numbers)
                if np.all(np.isin(atomic_numbers, ALLOWED_ATOMIC_NUMBERS)):
                    db_all_count += 1
                    out_db.write(row.toatoms(),
                                data=row.data,
                                **row.key_value_pairs)
    
    print(f"Database: {db_file}")
    print(f"\tAllowed molecules (all): {db_all_count} - {db_all_count / db_total_count * 100:.2f}%")
    
    return db_all_count, db_total_count


if __name__ == "__main__":
    db_files = sorted(glob(DATASET_WILDCARD))
    makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use all available CPU cores (or specify a number, e.g., Pool(4))
    num_processes = cpu_count()
    print(f"Processing {len(db_files)} database files using {num_processes} processes...")
    
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_db_file, db_files), total=len(db_files),))
    
    # Aggregate results
    all_count = sum(r[0] for r in results)
    total_count = sum(r[1] for r in results)
    
    print(f"\n{'='*60}")
    print(f"Total molecules processed: {total_count}")
    print(f"Total molecules with allowed atoms: {all_count} ({all_count / total_count * 100:.2f}%)")
    print(f"{'='*60}")

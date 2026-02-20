from os import makedirs
import numpy as np
from ase.db import connect
from tqdm import tqdm 
from glob import glob

ALLOWED_ATOMIC_NUMBERS = np.array([1, 6, 7, 8, 9, 17], dtype=np.int64)
HAS_ONLY_ALLOWED_ATOMS = False
DATASET_PATH = "/home/karella/Projects/hippynn/dataset/omol_25_neutral_train"
OUTPUT_DIR = "/home/karella/Projects/hippynn/dataset/omol_25_bens_neutral_train"
DATASET_WILDCARD = f"{DATASET_PATH}/*.aselmdb"

MAX_ASELMBD_FILE_ENTRIES = 429198  # How big should be the export files
makedirs(OUTPUT_DIR, exist_ok=True)

all_count = 0 
outdir = 0
out_entries = 0

total_count = 0
out_db = connect(f"{OUTPUT_DIR}/data{outdir:04d}.aselmdb", readonly=False)

for db_file in tqdm(sorted(glob(DATASET_WILDCARD))):
    with connect(db_file, readonly=True) as db:
        db_all_count = 0
        db_any_count = 0
        for row in db.select(): 
            atomic_numbers = np.unique(row.numbers)
            if np.all(np.isin(atomic_numbers, ALLOWED_ATOMIC_NUMBERS)):
                db_all_count += 1
                out_db.write(row.toatoms(),
                             data=row.data,
                             **row.key_value_pairs)
                out_entries += 1
                if out_entries >= MAX_ASELMBD_FILE_ENTRIES:
                    out_db.flush()
                    outdir += 1
                    out_db = connect(f"{OUTPUT_DIR}/data{outdir:04d}.aselmdb", readonly=False)
                    out_entries = 0
                    print(f"Created new output database: {OUTPUT_DIR}/data{outdir:04d}.aselmdb")
            else: 
                pass
        print(f"Database: {db_file}")
        print(f"\tAllowed molecules (all): {db_all_count} - {db_all_count / db.count() * 100:.2f}%")
        total_count += db.count()
        all_count += db_all_count
        any_count += db_any_count


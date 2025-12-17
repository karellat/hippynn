import os
# TODO: Remove
import torch
from torch.nn.functional import pad
from tqdm import tqdm
from ase.db import connect
from typing import Optional
from torch.utils.data import DataLoader
from fairchem.core.datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from fairchem.core.datasets.atomic_data import AtomicData

from hippynn.databases import _Database


class Omol25Database(_Database):
    """
    Database for the Omol25 dataset.

    Expects the data to be in a directory structure compatible with
    hippynn.databases.DirectoryDatabase, with files named
    omol25_data_*.db

    Inputs:
        - atomic_numbers
        - pos

    Targets:
        - energy
        - forces
    """
    def _hippynn_collate_fn(self, data_list: list[AtomicData], exclude_keys: Optional[list] = None) -> list:   
        """Convert a list of AtomicData objects into a batched AtomicData object."""
        # Using fairchem utility function
        atomic_numbers = []
        positions = []
        energies = []
        forces = []

        for mol in data_list:
            padding_length = self.n_atoms_max - mol.natoms
            atomic_numbers.append(pad(mol.atomic_numbers, (0, padding_length), value=0))
            positions.append(pad(mol.pos, (0, 0, 0, padding_length), value=0.0))
            energies.append(mol.energy)
            forces.append(pad(mol.forces, (0, 0, 0, padding_length), value=0.0))
        atomic_numbers = torch.stack(atomic_numbers).to(torch.int32)
        positions = torch.stack(positions).to(torch.get_default_dtype())
        energies = torch.stack(energies).to(torch.get_default_dtype())
        forces = torch.stack(forces).to(torch.get_default_dtype())
        return [atomic_numbers, positions, energies, forces]

    @property
    def is_in_memory(self) -> bool:
        return False

    @property
    def inputs(self) -> list[str]:
        return ["atomic_numbers", "pos"]

    @property
    def targets(self) -> list[str]:
        return ["energy", "forces"]

    @property
    def var_list(self) -> list[str]:
        return self.inputs + self.targets
    
    @property
    def splits(self) -> list[str]:
        return ["train", "valid", "test"]

    def __init__(self,
                 training_asedb_path: str, 
                 validation_asedb_path: str,
                 test_asedb_path: str,
                 n_atoms_max: Optional[int] = None,
                 dataloader_kwargs: dict = {}):
        super().__init__()
        self.AseDBDatasets = dict()

        _labels = dict(train=training_asedb_path,
                       valid=validation_asedb_path, 
                       test=test_asedb_path)

        assert os.path.exists(training_asedb_path), f"Training database path does not exist: {training_asedb_path}"
        assert os.path.exists(validation_asedb_path), f"Validation database path does not exist: {validation_asedb_path}"
        assert os.path.exists(test_asedb_path), f"Test database path does not exist: {test_asedb_path}"
        
        # Connect to db and count the maximum natoms
        self.dataloader_kwargs = dataloader_kwargs
        self.n_atoms_max = 0 
        for split_name, path in _labels.items():
            if not os.path.exists(path):
                raise ValueError(f"Path for split {split_name} does not exist: {path}")
            else: 
                db = AseDBDataset(
                    config=dict(
                        src=path,
                        a2g_args=dict(
                            task_name="omol",   
                            r_energy=True,
                            r_forces=True,
                            r_stress=False,
                        ),
                    )
                )
                self.AseDBDatasets[split_name] = db
        if n_atoms_max is not None:
            self.n_atoms_max = n_atoms_max
        else: 
            for split_name in self.splits:
                # Determine n_atoms_max if not provided
                if n_atoms_max is None:
                    dl = self._get_loader(split_name, batch_size=16384, shuffle=False, collate_fn=atomicdata_list_to_batch, num_workers=12)
                    for batch in tqdm(dl):
                        batch = batch.to(torch.device('cuda'))
                        self.n_atoms_max = max(self.n_atoms_max, torch.max(batch.natoms))
        print(f"Determined n_atoms_max: {self.n_atoms_max}")
    
    def _get_loader(self, split_name: str, batch_size: int, shuffle: bool, collate_fn=None, **dataloader_kwargs) -> DataLoader:
        dataset = self.AseDBDatasets[split_name]
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )


    def make_generator(self, 
                       split_name: str, 
                       evaluation_mode: str,
                       batch_size: int):
        assert evaluation_mode in ["train", "eval"], "evaluation_mode must be 'train' or 'eval'"
        if split_name not in self.splits:
            raise ValueError(f"Split {split_name} Invalid. Current splits:{list(self.splits)}")
        shuffle = evaluation_mode == "train"
        return self._get_loader(split_name, batch_size, shuffle, self._hippynn_collate_fn, **self.dataloader_kwargs)
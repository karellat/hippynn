import os
import yaml
import torch
from torch.nn.functional import pad
from tqdm import tqdm
from typing import Optional
from torch.utils.data import DataLoader
from fairchem.ase_datasets import AseDBDataset
from fairchem.atomic_data import atomicdata_list_to_batch, AtomicData

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
        out_batch = {
            self._atomic_numbers: [],
            self._pos: [],
            self._energy: [],
            self._forces: []
        }
        for mol in data_list:
            padding_length = self.n_atoms_max - mol.natoms
            out_batch[self._atomic_numbers].append(pad(mol.atomic_numbers, (0, padding_length), value=0))
            out_batch[self._pos].append(pad(mol.pos, (0, 0, 0, padding_length), value=0.0))
            out_batch[self._energy].append(mol.energy)
            out_batch[self._forces].append(pad(mol.forces, (0, 0, 0, padding_length), value=0.0))
        
        # Stack tensors (before retyping)
        out_batch[self._atomic_numbers] = torch.stack(out_batch[self._atomic_numbers])
        out_batch[self._pos] = torch.stack(out_batch[self._pos])
        out_batch[self._energy] = torch.stack(out_batch[self._energy])
        out_batch[self._forces] = torch.stack(out_batch[self._forces])
        
        # Apply normalization (if available)
        if hasattr(self, 'normalization_params') and self.normalization_params is not None:
            out_batch[self._energy] = self._apply_energy_normalization(out_batch[self._atomic_numbers], out_batch[self._energy])
        
        # Retype after normalization
        out_batch[self._atomic_numbers] = out_batch[self._atomic_numbers].to(torch.long)
        out_batch[self._pos] = out_batch[self._pos].to(torch.get_default_dtype())
        out_batch[self._energy] = out_batch[self._energy].to(torch.get_default_dtype())
        out_batch[self._forces] = out_batch[self._forces].to(torch.get_default_dtype())
        
        return [out_batch[var_name] for var_name in self.var_list]
    
    def _apply_energy_normalization(self, species: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """Apply E0 normalization to energies using reference values."""
        # Ignore zeros and check that all species are in the normalization parameters
        assert torch.isin(torch.unique(species)[1:], self.species).all(), "All species in the batch must be in the normalization parameters"
        atom_counts = torch.nn.functional.one_hot(species, num_classes=self._species_max+1).sum(dim=1).double()
        atom_counts = atom_counts[:, self.species]  # Filter to only include specified species
        energy_e0 = self.normalization_params @ atom_counts.T
        normalized_energy = energy[:, 0] - energy_e0
        return normalized_energy.unsqueeze(1)

    @property
    def is_in_memory(self) -> bool:
        return False

    @property
    def splits(self) -> list[str]:
        return ["train", "valid", "test"]

    @property
    def inputs(self) -> list:
        return self._inputs
    
    @inputs.setter
    def inputs(self, value: list) -> None:
        valid_names = {"atomic_numbers", "pos", "energy", "forces"}
        if not all(name in valid_names for name in value):
            raise ValueError(f"Invalid input names. Valid names are: {valid_names}")
        self._inputs = value
    
    @property
    def targets(self) -> list:
        return self._targets
    
    @targets.setter
    def targets(self, value: list) -> None:
        valid_names = {"atomic_numbers", "pos", "energy", "forces"}
        if not all(name in valid_names for name in value):
            raise ValueError(f"Invalid target names. Valid names are: {valid_names}")
        self._targets = value

    def __init__(self,
                 db_inputs: list[str],
                 db_targets: list[str],
                 training_asedb_path: str | list[str], 
                 validation_asedb_path: str | list[str],
                 test_asedb_path: str | list[str],
                 n_atoms_max: Optional[int] = None,
                 species: Optional[list[int]] = None,
                 dataloader_kwargs: dict = {},
                 normalization_yaml_path: Optional[str] = None):

        # Set default inputs and targets 
        self._atomic_numbers = "atomic_numbers"
        self._pos = "pos"
        self._energy = "energy"
        self._forces = "forces"

        self._inputs = []
        self._targets = []
        super().__init__()
        self.AseDBDatasets = dict()
        self.inputs = db_inputs
        self.targets = db_targets
        self.species = torch.tensor(species, dtype=torch.int) if species is not None else torch.tensor([range(1, 100)], dtype=torch.int)
        self._species_max = torch.max(self.species).item()
        
        # Load normalization parameters if provided
        self.normalization_params = None
        if normalization_yaml_path is not None:
            self._load_normalization_params(normalization_yaml_path, species)

        # order the inpus and targets
        training_asedb_path = training_asedb_path if isinstance(training_asedb_path, list) else [training_asedb_path] 
        validation_asedb_path = validation_asedb_path if isinstance(validation_asedb_path, list) else [validation_asedb_path]
        test_asedb_path = test_asedb_path if isinstance(test_asedb_path, list) else [test_asedb_path]
        _labels = dict(train=training_asedb_path,
                       valid=validation_asedb_path, 
                       test=test_asedb_path)
        # Connect to db and count the maximum natoms
        self.dataloader_kwargs = dataloader_kwargs
        self.n_atoms_max = 0 
        for split_name, paths in _labels.items():
            for path in paths: 
                if not os.path.exists(path):
                    raise ValueError(f"Path for split {split_name} does not exist: {path}")
            db = AseDBDataset(
                config=dict(
                    src=paths,
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
        # We load splits independently. 
        self.splitting_completed = True
    
    def _load_normalization_params(self, yaml_path: str, species: Optional[list[int]] = None) -> None:
        """Load normalization reference values from YAML file."""
        if not os.path.exists(yaml_path):
            raise ValueError(f"Normalization YAML file does not exist: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        
        assert 'omol_elem_refs' in yaml_data, f"YAML file must contain 'omol_elem_refs' key. Found keys: {list(yaml_data.keys())}"
        # TODO: We should try the OC22 normalization, it seems UMA doesn't bother with spins and charges.
        omol_ref = yaml_data['omol_elem_refs']
        # Safe tensor creation with proper device handling
        self.normalization_params = torch.as_tensor(omol_ref, dtype=torch.float64)
        print(f"Loaded omol pre-normalization parameters from {yaml_path}")
        if species is not None:
            # Filter normalization parameters to only include specified species
            for  s in species:
                assert s < len(self.normalization_params), f"Species index {s} out of bounds for normalization parameters with length {len(self.normalization_params)}"
            self.normalization_params = self.normalization_params[species]
            print(f"\tSpecies provided: {species}. Filtered normalization parameters to these species.")
    
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
        assert split_name != 'test', "Test generator is not supported. Omol25 test set does not have energies."
        assert evaluation_mode in ["train", "eval"], "evaluation_mode must be 'train' or 'eval'"
        if split_name not in self.splits:
            raise ValueError(f"Split {split_name} Invalid. Current splits:{list(self.splits)}")
        shuffle = evaluation_mode == "train"
        return self._get_loader(split_name, batch_size, shuffle, self._hippynn_collate_fn, **self.dataloader_kwargs)
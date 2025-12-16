import os
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
    @staticmethod
    def _hippynn_collate_fn(data_list: list[AtomicData]):   
        """Convert a list of AtomicData objects into a batched AtomicData object."""
        # Using fairchem utility function
        atomic_data_batch = atomicdata_list_to_batch(data_list)
        # Extract only the required fields for HippyNN batch[:n_inputs] batch[-n_targets:]
        return [atomic_data_batch.atomic_numbers, atomic_data_batch.pos, 
                atomic_data_batch.energy, atomic_data_batch.forces]

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
        return ["train", "val", "test"]


    def __init__(self,
                 training_asedb_path: str, 
                 validation_asedb_path: str,
                 test_asedb_path: str,
                 dataloader_kwargs: dict = {}):
        super().__init__()
        self.AseDBDatasets = dict()

        _labels = dict(train=training_asedb_path,
                       val=validation_asedb_path, 
                       test=test_asedb_path)

        self.dataloader_kwargs = dataloader_kwargs
        for split_name, path in _labels.items():
            if not os.path.exists(path):
                raise ValueError(f"Path for split {split_name} does not exist: {path}")
            else: 
                self.AseDBDatasets[split_name] = AseDBDataset(
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

    def make_generator(self, 
                       split_name: str, 
                       batch_size: int):
        if split_name not in self.splits:
            raise ValueError(f"Split {split_name} Invalid. Current splits:{list(self.splits)}")
        dataset = self.AseDBDatasets[split_name]
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=Omol25Database._hippynn_collate_fn,  # turns list[AtomicData] -> Batched HippyNN format
            **self.dataloader_kwargs,
    )
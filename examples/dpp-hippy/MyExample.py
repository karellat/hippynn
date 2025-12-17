import os
import torch
import pytorch_lightning as pl

from hippynn.experiment.routines import SetupParams
from hippynn.graphs import inputs, networks, targets, physics
from hippynn.graphs.nodes.loss import MAELoss
from hippynn.experiment.assembly import assemble_for_training
from hippynn.tools import active_directory, log_terminal
from hippynn.experiment import setup_and_train
from hippynn.experiment import HippynnLightningModule
from hippynn.pretraining import hierarchical_energy_initialization
# TODO: Source code 
from Omol25_database import Omol25Database

# Parameters
TEST_DIR = "/home/karella/Projects/hippynn/examples/dpp-hippy/test_folder"
TRAIN_PATH = "/home/karella/Projects/hippynn/train_4M"
VAL_PATH = "/home/karella/Projects/hippynn/val"
TEST_PATH = "/home/karella/Projects/hippynn/test"
HENERGY_INIT_PATH = "/home/karella/Projects/examples/dpp-hippy/test_folder/omol4M_hierarchical_energy_init.pt"

MAX_EPOCHS = 50
BATCH_SIZE = 16
LR_RATE = 0.001

N_ATOM_MAX = 350

NETWORK_PARAMS = {
    "possible_species": list(range(90)),
    "n_features": 128,
    "n_sensitivities": 20,
    "dist_soft_min": 0.8,
    "dist_soft_max": 5.5,
    "dist_hard_max": 6.5,
    "n_interaction_layers": 2,
    "n_atom_layers": 5,
    "n_max":3,
    "l_max":2,
    "sensitivity_type": "inverse",
    "resnet": True,
}


# 1. Setup the model graph
species = inputs.SpeciesNode(db_name="atomic_numbers")
positions = inputs.PositionsNode(db_name="pos")
# TODO: Ask Michael about cell input
# cell = inputs.CellNode(db_name="cell")

network = networks.HipHopnn("HipHopnn", 
                            (species, positions),
                             module_kwargs=NETWORK_PARAMS,
                             periodic=False)

henergy = targets.HEnergyNode("HEnergy", network)
sys_energy = henergy.mol_energy
sys_energy.db_name = "energy"
hierarchicality = henergy.hierarchicality
hierarchicality = physics.PerAtom("RperAtom", hierarchicality)
force = physics.GradientNode("forces", (sys_energy, positions), sign=-1)
force.db_name = "forces"

validation_losses = {
"T-MAE":MAELoss.of_node(sys_energy),
"F-MAE":MAELoss.of_node(force),
}
validation_losses['train'] =  validation_losses['T-MAE'] + validation_losses['F-MAE']

train_loss = validation_losses['train']

training_modules, db_info = assemble_for_training(train_loss, validation_losses)

# 2. Setup Lightning training environment
with active_directory(TEST_DIR):
    # Log the output of python to `training_log.txt`
    with log_terminal("training_log.txt", "wt"):
        database = Omol25Database(
            training_asedb_path=TRAIN_PATH,
            validation_asedb_path=VAL_PATH,
            test_asedb_path=TEST_PATH,
            n_atoms_max=N_ATOM_MAX,
            dataloader_kwargs={'num_workers': 12},
        )

        # TODO: This must be made sequential
        # TODO: Test this on some dataset that can be fitted into memory and compare
        # TODO: This seems to be called in the trainer too. 
        if not os.path.exists(HENERGY_INIT_PATH):
            print("Initializing hierarchical energy...")
            hierarchical_energy_initialization(henergy, database, trainable_after=False)
            torch.save(henergy, HENERGY_INIT_PATH)
            print(f"Hierarchical energy initialization saved to {HENERGY_INIT_PATH}")
        else:
            print(f"Loading hierarchical energy from {HENERGY_INIT_PATH}")
            henergy_loaded = torch.load(HENERGY_INIT_PATH)
            henergy.load_state_dict(henergy_loaded.state_dict())
            print("Hierarchical energy loaded successfully")
        

        # Parameters describing the training procedure.
        experiment_params = SetupParams(
            stopping_key="T-MAE",  # The name in the validation_losses dictionary.
            # TODO: This should be automaticall find
            batch_size=BATCH_SIZE,
            # TODO: Is there a good reason for Adam instead AdamW?
            optimizer=torch.optim.Adam,
            max_epochs=MAX_EPOCHS,
            learning_rate=LR_RATE,
        )

        # (*) For debubging purposes try to run directly 
        #setup_and_train(
        #    training_modules=training_modules,
        #    database=database,
        #    experiment_params=experiment_params)
# Run the Lightning Module
# Try out the lightning callbacks and parallel logging

# lightning needs to run exactly where the script is located in distributed modes.
lightmod, datamodule = HippynnLightningModule.from_experiment_setup(training_modules, 
                                                                    database, 
                                                                    experiment_params)
# Init ModelCheckpoint callback, monitoring 'val_loss'
# checkpoint_callback = ModelCheckpoint(monitor="val_loss")
# TODO: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html


trainer = pl.Trainer(accelerator='gpu',
                     devices=2, 
                     strategy="ddp",
                     callbacks=[], # Callbacks in WanDB 
                     use_distributed_sampler=True)

trainer.fit(model=lightmod, datamodule=datamodule)
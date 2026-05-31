"""
Michael Omol example 
"""

import os
import time
import torch
import wandb
import logging
from fairchem.core.datasets import AseDBDataset

import hippynn
from hippynn.experiment import SetupParams, setup_and_train, test_model
from hippynn.interfaces.ase_interface import AseDatabaseIterable

from hippynn.graphs import inputs, networks, targets, physics, loss
from hippynn.graphs.nodes.loss import MSELoss, MAELoss, Rsq, Mean
from hippynn.experiment.assembly import assemble_for_training
from hippynn.pretraining import hierarchical_energy_initialization
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController

torch.set_default_dtype(torch.float32)
hippynn.settings.WARN_LOW_DISTANCES = True
# TODO: ask michael about this setting
# 

class TomasWandbLogger():
    def __init__(self, metric_tracker, wandb_run):
        self.metric_tracker = metric_tracker
        self.run = wandb_run

    def __call__(self, epoch, better_model):
        tracker = self.metric_tracker
        assert epoch == (tracker.current_epoch - 1)
        current_metrics = tracker.epoch_metric_values[epoch]
        for split, metrics in current_metrics.items():
            for key, value in metrics.items():
                self.run.log({f"{split}-{key}": value}, step=epoch)
        for key, value in tracker.best_metric_values.items():
            self.run.summary[f"Best-{key}"] = value


max_epochs = 500

first_n = int(1e5)

network_params = {
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

early_stopping_key = "train"
test_size = 0.1
valid_size = 0.1

training_path = os.path.abspath('.') + '/'

def stream_conversion_generator(fairchem_db):

    for atom_data in fairchem_db:
        atoms = atom_data.to_ase()[0]
        arrays = atoms.arrays
        #arrays['atomic_numbers'] = atom_data.atomic_numbers.numpy()
        arrays['forces'] = atom_data.forces.numpy()
        arrays['positions'] = atom_data.pos.numpy()
        
        atoms.info['energy'] = atom_data.energy.numpy()
        #atoms.info['charge'] = atom_data.charge
        #atoms.info['spin'] = atom_data.spin
        yield atoms

    return



def setup_network(network_params):

    # Hyperparameters for the network
    logging.debug("Network hyperparameters:")
    logging.debug(network_params)

    

    species = inputs.SpeciesNode(db_name="numbers")
    positions = inputs.PositionsNode(db_name="positions")
    cell = inputs.CellNode(db_name="cell")

    network = networks.HipHopnn("HipHopnn", (species, positions), module_kwargs=network_params, periodic=False)

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
    

    # Factors of 1e3 for meV

    training_modules, db_info = assemble_for_training(train_loss, validation_losses)
    
    return henergy, training_modules, db_info

def fit_model(training_modules,database):

    model, loss_module, model_evaluator = training_modules

    hierarchical_energy_initialization(henergy, database, peratom=False, energy_name="energy", decay_factor=1e-2)

    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=1e-3)

    scheduler = RaiseBatchSizeOnPlateau(
        optimizer=optimizer,
        max_batch_size=32,
        patience=25,
        factor=0.5,
    )

    controller = PatienceController(
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=16,
        eval_batch_size=16,
        max_epochs=max_epochs,
        termination_patience=50,
        stopping_key=early_stopping_key,
    )
    experiment_params = SetupParams(controller=controller)

    logging.info("Experiment Params set")
    logging.debug("%s", experiment_params)

    # Parameters describing the training procedure.
    logging.debug("Current controller epoch: %s", controller.current_epoch)
    sTime = time.time()

    setup_and_train(
        training_modules=training_modules,
        database=database,
        setup_params=experiment_params,
    )

    elTime = time.time()-sTime
    EpTime = elTime/controller.current_epoch


    with hippynn.tools.log_terminal("model_results.txt",'wt'):
        test_model(database, training_modules.evaluator, 128, "Final Training")
        logging.info("FOM Average Epoch time: %12.8f", EpTime)

if __name__=="__main__":
    logging.info("Set-Up Wandb connection.")
    wandb_settings = wandb.Settings(_disable_stats=True, save_code=False, _disable_meta=True,x_save_requirements=False)

    with wandb.init(project="hiphop-omol", settings=wandb_settings, entity="karella", config=network_params) as wandb_run:
        logging.info("Setting up model.")
        henergy, training_modules, db_info = setup_network(network_params)
        # TODO: Watch the model.

        ### Read in the dataset you wish to submit predictions to
        dataset = AseDBDataset({"src": "/home/karella/Projects/hippynn/train_4M"})  
        logging.debug("%s", db_info)
        logging.info("Preparing dataset.")

        gen = stream_conversion_generator(dataset[0:first_n])

        database = AseDatabaseIterable(
            iterable=gen,
            seed=1001,  # Random seed for splitting data
            quiet=False,
            pin_memory=False,
            test_size=test_size,
            valid_size=valid_size,
            **db_info)
        
        database.send_to_device() # Send to GPU if available

        logging.info("Training model")
        with hippynn.tools.active_directory("model_files"):
            fit_model(training_modules,database)
        
        logging.info("Writing test results")
        with hippynn.tools.log_terminal("model_results.txt",'wt'):
            test_model(database, training_modules.evaluator, 128, "Final Training")
        
        logging.info("All done.")

# TODO: Ask Michael about deleted code. 
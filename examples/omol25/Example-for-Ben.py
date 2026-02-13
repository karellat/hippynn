BATCH_SIZE = 512
network_params = {
    "possible_species": list(range(90)),   
    "n_features": 128,                     
    "n_sensitivities": 20,                 
    "dist_soft_min": 0.8,                  
    "dist_soft_max": 5.5,                  
    "dist_hard_max": 6.5,                  
    "n_interaction_layers": 2,             
    "n_atom_layers": 5,                    
    "n_max": 3,                            
    "l_max": 2,                            
    "sensitivity_type": "inverse",         
    "resnet": True,                        
}

# Database parameters
training_asedb_path = "/home/karella/Projects/hippynn/train_4M"
validation_asedb_path = "/home/karella/Projects/hippynn/val"
test_asedb_path = "/home/karella/Projects/hippynn/test"
n_atoms_max = 350
dataloader_kwargs = {'num_workers': 1}
# NOTE: This can be downloaded: https://raw.githubusercontent.com/facebookresearch/fairchem/refs/heads/main/configs/uma/training_release/element_refs/uma_v1_hof_lin_refs.yaml
normalization_yaml_path = "/home/karella/Projects/hippynn/examples/omol25/uma_v1_hof_lin_refs.yaml"

import torch 
from tqdm import tqdm 
torch.set_default_dtype(torch.float32)

from hippynn.graphs import inputs, networks, targets, physics
from hippynn.graphs.nodes.loss import MAELoss
from hippynn.experiment.assembly import assemble_for_training
from Omol25_database import Omol25Database

# 1. Setup the model graph
species = inputs.SpeciesNode(db_name="atomic_numbers")
positions = inputs.PositionsNode(db_name="pos")

network = networks.HipHopnn("HipHopnn", 
                        (species, positions),
                        module_kwargs=network_params,
                        periodic=False)

henergy = targets.HEnergyNode("HEnergy", network)
sys_energy = henergy.mol_energy
sys_energy.db_name = "energy"
force = physics.GradientNode("forces", (sys_energy, positions), sign=-1)
force.db_name = "forces"

validation_losses = {
"T-MAE":MAELoss.of_node(sys_energy),
"F-MAE":MAELoss.of_node(force),
}
validation_losses['loss'] =  validation_losses['T-MAE'] + validation_losses['F-MAE']

train_loss = validation_losses['loss']

training_modules, db_info = assemble_for_training(train_loss, validation_losses)

model, loss_module, model_evaluator = training_modules


database = Omol25Database(
    db_inputs=db_info['inputs'],
    db_targets=db_info['targets'],
    training_asedb_path=training_asedb_path,
    validation_asedb_path=validation_asedb_path,
    test_asedb_path=test_asedb_path,
    n_atoms_max=n_atoms_max,
    dataloader_kwargs=dataloader_kwargs,
    # Comment this line to disable energy normalization
    normalization_yaml_path=normalization_yaml_path
)

loader = database.make_generator(split_name="train", batch_size=BATCH_SIZE, evaluation_mode='train')

num_samples = 0
target_energy = 0.0
pbar = tqdm(loader)
for batch in pbar:
    species = batch[database.inputs.index('atomic_numbers')]
    energy = batch[database.targets.index('energy') + len(database.inputs)]
    atomic_numbers = batch[database.targets.index('atomic_numbers') + len(database.inputs)]

    target_energy += energy.sum().item()  # Update target energy with the mean of the batch
    num_samples += energy.numel()
    # Update progress bar with current mean and variance
    pbar.set_description(f"Mean Energy: {target_energy / num_samples:.4f}")
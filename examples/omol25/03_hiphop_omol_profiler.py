import os
import torch

import hippynn
from hippynn.experiment.controllers import PatienceController, RaiseBatchSizeOnPlateau
from hippynn.graphs import inputs, networks, targets, physics
from hippynn.graphs.nodes.loss import MAELoss
from hippynn.experiment.assembly import assemble_for_training
from hippynn.tools import active_directory, log_terminal
from hippynn.experiment import HippynnLightningModule, setup_and_profile
from hippynn.pretraining import hierarchical_energy_initialization
from hippynn.plotting import SensitivityPlot, PlotMaker
from Omol25_database import Omol25Database
from common import (
    configure_torch_memory_allocator,
    get_network_params,
    get_torch_backend,
    get_trainer_params,
    parse_args,
)
    

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Set CUDA memory allocator configuration to combat fragmentation
    configure_torch_memory_allocator()
    
    if args.wandb:
        import wandb
    
    # Debug settings
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('high')
    hippynn.settings.WARN_LOW_DISTANCES = True
    print(f"Detected torch backend: {get_torch_backend()}")

    # Build parameters from args
    network_params = get_network_params(args)
    trainer_params = get_trainer_params(args)

    # 1. Setup the model graph
    species = inputs.SpeciesNode(db_name="atomic_numbers")
    positions = inputs.PositionsNode(db_name="pos")

    network = networks.HipHopnn("HipHopnn", 
                                (species, positions),
                                module_kwargs=network_params,
                                periodic=False)

    henergy = targets.HEnergyNode("HEnergy", network)
    sys_energy = henergy.mol_energy # TODO: What mol energy means in this? 
    sys_energy.db_name = "energy"
    force = physics.GradientNode("forces",
                                 (sys_energy, positions),
                                  sign=-1)
    force.db_name = "forces"

    validation_losses = {
    "T-MAE":MAELoss.of_node(sys_energy),
    "F-MAE":MAELoss.of_node(force),
    }
    # TODO: We should introduce some weights here
    validation_losses['loss'] =  validation_losses['T-MAE'] + validation_losses['F-MAE']

    train_loss = validation_losses['loss']

    training_modules, db_info = assemble_for_training(train_loss, validation_losses)

    model, loss_module, model_evaluator = training_modules

    # 2. Setup Lightning training environment
    # Log the output of python to `training_log.txt`
    with active_directory(args.test_dir):
        with log_terminal("training_log.txt", "wt"):
            database = Omol25Database(
                db_inputs=db_info['inputs'],
                db_targets=db_info['targets'],
                training_asedb_path=args.train_path,
                validation_asedb_path=args.val_path,
                test_asedb_path=args.test_path,
                species=network_params['possible_species'],
                n_atoms_max=args.n_atom_max,
                dataloader_kwargs={
                    'num_workers': args.profile_workers},
                normalization_yaml_path=args.normalization_yaml_path
            )

            # Calculating energies 
            # TODO: This might be a bit redundant, we can just calculate the energies since energies are UMA normalized.
            if not os.path.exists(args.henergy_init_path):
                print("Initializing hierarchical energy...")
                hierarchical_energy_initialization(henergy, database, energy_name="energy", decay_factor=1e-2)
                #TODO: Check the training dataset path and count MD5 hash to check the training data consistency 
                torch.save(henergy.torch_module.state_dict(), args.henergy_init_path)
                print(f"Hierarchical energy initialization saved to {args.henergy_init_path}")
            else:
                print(f"Loading hierarchical energy from {args.henergy_init_path}")
                result = henergy.torch_module.load_state_dict(torch.load(args.henergy_init_path, weights_only=True))
                print("Hierarchical energy loaded successfully")
            
            # 
            optimizer = torch.optim.AdamW(training_modules.model.parameters(),
                                        lr=args.lr_rate)

            scheduler = RaiseBatchSizeOnPlateau(
                optimizer=optimizer,
                max_batch_size=args.batch_size,
                patience=args.patience,
                factor=0.5,
            )
            controller = PatienceController(
                optimizer=optimizer,
                scheduler=scheduler,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                max_epochs=args.max_epochs,
                termination_patience=args.termination_patience,
                stopping_key='loss'
            )
            experiment_params = hippynn.experiment.SetupParams(
                controller=controller
            )
        

            _ = setup_and_profile(
                training_modules=training_modules,
                database=database,
                batches_per_epoch=args.debug_batches_per_epoch,
                profile_epochs=args.profile_epochs,
                setup_params=experiment_params)
import os
import yaml
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
from matplotlib import pyplot as plt

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

# TODO: List 
# - Add test evaluation after training https://fair-chem.github.io/molecules/leaderboard.html#s2ef 
# - Fix indexers to work without knowing n_max_atoms in advance
# - When run after calculating hierarchical energy init it fails
# - Support multiple nodes training
# - Fix the stride warning in gradients (permute, contiguous on grads?) 
# - Testdataloader does not have energies 
# - Validation loss explodes for large datasets, we should change eval_step to accumulate loss rather than accumulate all predictions - quick fixed this with averaging eval loss
    # * TODO:  FIX THIS: 
# - Try to replace Lightning with TorchTNT 
# - Ask Nick about 
# - Move the init energies on the dataset



def parse_args():
    parser = argparse.ArgumentParser(description="HipHopNN Training Script")
    
    # Config file option
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI args override config file values.")
    
    # Directories
    parser.add_argument("--test-dir", type=str, default="/home/karella/Projects/hippynn/examples/dpp-hippy/test-4M",
                        help="Directory for test outputs and logs")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints (default: {test_dir}/checkpoints)")
    
    # Dataset paths
    parser.add_argument("--train-path", type=str, default=["/home/karella/Projects/hippynn/dataset/omol_4M_train"],
                        help="Path to training dataset")
    parser.add_argument("--val-path", type=str, default=["/home/karella/Projects/hippynn/dataset/omol_val"],
                        help="Path to validation dataset")
    parser.add_argument("--val-neutral-path", type=str, default="",
                        help="Path to neutral validation dataset")
    parser.add_argument("--test-path", type=str, default="/home/karella/Projects/hippynn/test",
                        help="Path to test dataset")
    parser.add_argument("--henergy-init-path", type=str, default=None,
                        help="Path to hierarchical energy init file (default: {test_dir}/hierarchical_energy_init.pt)")
    parser.add_argument("--normalization-yaml-path", type=str, default="/home/karella/Projects/hippynn/examples/omol25/uma_v1_hof_lin_refs.yaml",
                        help="Path to YAML file with normalization reference values")
    
    # Training parameters
    parser.add_argument("--dl-num-workers", type=int, default=10,
                        help="Number of dataloader workers")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode with limited batches")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPU devices")
    parser.add_argument("--profiler", action="store_true", default=False,
                        help="Enable PyTorch Lightning profiler")
    parser.add_argument("--nodes", type=int, default=1,
                        help="Number of compute nodes")
    parser.add_argument("--max-epochs", type=int, default=1,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1,
                        help="Number of batches to accumulate gradients (for effective larger batch size)")
    parser.add_argument("--lr-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=25,
                        help="Patience for learning rate scheduler")
    parser.add_argument("--termination-patience", type=int, default=50,
                        help="Patience for early stopping")
    parser.add_argument("--n-atom-max", type=int, default=350,
                        help="Maximum number of atoms per molecule")
    
    # Network parameters
    parser.add_argument("--n-features", type=int, default=128,
                        help="Number of features in network")
    parser.add_argument("--n-sensitivities", type=int, default=20,
                        help="Number of sensitivities")
    parser.add_argument("--dist-soft-min", type=float, default=0.8,
                        help="Soft minimum distance cutoff")
    parser.add_argument("--dist-soft-max", type=float, default=5.5,
                        help="Soft maximum distance cutoff")
    parser.add_argument("--dist-hard-max", type=float, default=6.5,
                        help="Hard maximum distance cutoff")
    parser.add_argument("--n-interaction-layers", type=int, default=2,
                        help="Number of interaction layers")
    parser.add_argument("--n-atom-layers", type=int, default=5,
                        help="Number of atom layers")
    parser.add_argument("--n-max", type=int, default=3,
                        help="Maximum n for spherical harmonics")
    parser.add_argument("--l-max", type=int, default=2,
                        help="Maximum l for spherical harmonics")
    parser.add_argument("--sensitivity-type", type=str, default="inverse",
                        help="Type of sensitivity function")
    parser.add_argument("--no-resnet", action="store_true", default=False,
                        help="Disable ResNet connections")
    parser.add_argument("--max-species", type=int, default=84,
                        help="Maximum number of species (elements), +1 for the zero padding.")
    parser.add_argument("--species", type=list, default=None,
                        help="List of atomic numbers to include as species")
    
    # WandB parameters
    parser.add_argument("--wandb", action="store_true", default=False,  
                        help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="hippynn",
                        help="WandB project name")
    parser.add_argument("--wandb-name", type=str, default="HipHopNN-omol4m-20-epoch-64-batch-fix",
                        help="WandB run name")
    parser.add_argument("--wandb-entity", type=str, default="karella",
                        help="WandB entity")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update args with config values (CLI args take precedence)
        for key, value in config.items():
            # Convert key from YAML format (snake_case or kebab-case) to argparse format
            arg_key = key.replace('-', '_')
            # Only set if not explicitly provided on command line
            if hasattr(args, arg_key) and getattr(args, arg_key) == parser.get_default(arg_key):
                setattr(args, arg_key, value)
    
    # Set derived paths
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"{args.test_dir}/checkpoints"
    if args.henergy_init_path is None:
        args.henergy_init_path = f"{args.test_dir}/hierarchical_energy_init.pt"
    
    return args


def get_network_params(args):
    """Build network parameters dictionary from args."""
    assert args.species is not None or args.max_species is not None, "Either --species or --max-species must be provided"
    assert not (args.species is not None and args.max_species is not None), "Only one of --species or --max-species can be provided"
    return {
        "possible_species": list(range(args.max_species)) if args.max_species is not None else args.species,
        "n_features": args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.dist_soft_min,
        "dist_soft_max": args.dist_soft_max,
        "dist_hard_max": args.dist_hard_max,
        "n_interaction_layers": args.n_interaction_layers,
        "n_atom_layers": args.n_atom_layers,
        "n_max": args.n_max,
        "l_max": args.l_max,
        "sensitivity_type": args.sensitivity_type,
        "resnet": not args.no_resnet,
    }


def get_trainer_params(args):
    """Build trainer parameters based on debug mode."""
    params = {
        'accumulate_grad_batches': args.accumulate_grad_batches,
    }
    
    if args.debug:
        params.update({
            'limit_train_batches': 100,
            'limit_val_batches': 100,
            'detect_anomaly': True,
            'deterministic': True
        })
    
    return params


class PlotCallback(pl.Callback):
    """Custom callback to generate plots and log them to wandb."""
    
    def __init__(self, plot_maker, plot_every=10):
        super().__init__()
        self.plot_maker = plot_maker
        self.plot_every = plot_every
    
    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        # Only plot every N epochs
        if current_epoch % self.plot_every != 0:
            return
        
        # Only run on rank 0
        if trainer.global_rank != 0:
            return
        
        
        for plotter in self.plot_maker.plotters:
            try:
                fig = plotter.make_plot(())
                
                if plotter.saved and trainer.logger:
                    plot_name = plotter.saved.replace('.pdf', '').replace('.png', '')
                    trainer.logger.experiment.log({
                        f"plots/{plot_name}": wandb.Image(fig),
                        "epoch": current_epoch
                    })
                
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not generate plot {plotter.saved}: {e}")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Set CUDA memory allocator configuration to combat fragmentation
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        #NOTE: We might need something similar for AMDs, I didn't find the equivalent setting.  
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    if args.wandb:
        import wandb
    
    # Debug settings
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('high')
    hippynn.settings.WARN_LOW_DISTANCES = False

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
                    'num_workers': args.dl_num_workers,
                    'pin_memory': True,
                    'persistent_workers': True,
                    'prefetch_factor': 4},
                normalization_yaml_path=args.normalization_yaml_path
            )

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
        
            # TODO: Original code uses mostly Adam optimizer 
            # https://arxiv.org/abs/1711.05101 
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

            # Setup plotters for visualization
            #plotters = [
            #    SensitivityPlot(
            #        network.torch_module.sensitivity_layers[0],
            #        saved="sensitivity",
            #        shown=False,
            #    ),
            #]

            #plot_maker = PlotMaker(
            #    *plotters,
            #    plot_every=10,
            #)

            # Setup plot callback for wandb logging
            # plot_callback = PlotCallback(plot_maker, plot_every=10)

            # Setup ModelCheckpoint callback for restart functionality
            checkpoint_callback = ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename="model-{epoch:02d}-{valid_loss:.4f}",
                monitor="valid_loss",
                mode="min",
                save_top_k=3,
                save_last=True,  # Always save last checkpoint for restart
            )

            # Create checkpoint directory if it doesn't exist
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            
            # Check for existing checkpoint to restart from
            checkpoint_path = f"{args.checkpoint_dir}/last.ckpt"
            ckpt_path = checkpoint_path if os.path.exists(checkpoint_path) else None
            
            # WanDB logging setup with checkpoint resumption support
            if args.wandb:
                wandb_run_id_file = f"{args.checkpoint_dir}/wandb_run_id.txt"
                if ckpt_path and os.path.exists(wandb_run_id_file):
                    with open(wandb_run_id_file, "r") as f:
                        wandb_run_id = f.read().strip()
                    print(f"Restarting training from checkpoint: {ckpt_path}")
                    print(f"Resuming wandb run: {wandb_run_id}")
                    resume_mode = "must"
                else:
                    wandb_run_id = wandb.util.generate_id()
                    resume_mode = "allow"
                    print("Starting fresh training")
                # Setup WandB logger
                wandb_logger = WandbLogger(
                    project=args.wandb_project,
                    name=args.wandb_name,
                    entity=args.wandb_entity,
                    save_dir=args.test_dir,
                    log_model=True,
                    id=wandb_run_id,
                    resume=resume_mode,
                    
                    config={
                        "lr_rate": args.lr_rate,
                        "batch_size": args.batch_size,
                        "eval_batch_size": args.eval_batch_size,
                        "max_epochs": args.max_epochs,
                        "patience": args.patience,
                        **network_params
                    }
                )
                # Save wandb run ID for future resumption
                with open(wandb_run_id_file, "w") as f:
                    f.write(wandb_run_id)
                trainer_params['logger'] = wandb_logger

            experiment_params = hippynn.experiment.SetupParams(
                controller=controller
            )

            
            # NOTE: For testing purposes, you can skip the lightning training
            #setup_and_profile(
            #    training_modules=training_modules,
            #    database=database,
            #    setup_params=experiment_params)
            #exit(0) 

            # lightning needs to run exactly where the script is located in distributed modes.
            lightmod, datamodule = HippynnLightningModule.from_experiment_setup(training_modules, 
                                                                                database, 
                                                                                experiment_params)
            if args.profiler:
                trainer_params['profiler'] = AdvancedProfiler(dirpath=args.test_dir,
                                                              filename="profiler_report.prof",
                                                              dump_stats=True)
            
            trainer = pl.Trainer(accelerator='gpu',
                                max_epochs=args.max_epochs,
                                devices=args.devices,
                                strategy="ddp" if args.devices > 1 else "auto",
                                num_nodes=args.nodes,
                                callbacks=[checkpoint_callback], # Add plot_callback if needed
                                use_distributed_sampler=True if args.devices > 1 else False,
                                **trainer_params,
                                )

            # TODO: Skip the gradients logging for now, it is very slow and we are not sure if it is working correctly with the current setup. We can add it back later with proper testing.
            # wandb_logger.watch(lightmod, log="gradients", log_freq=10000, log_graph=False)

            if args.wandb:
                wandb_run = wandb_logger.experiment
                if wandb_run is not None:
                    wandb_run.define_metric("train_loss", summary="min")
            try: 
                _wandb_status = "failure"
                trainer.fit(model=lightmod, 
                            datamodule=datamodule,
                            ckpt_path=ckpt_path)
                _wandb_status = "success"
            finally:
                if args.wandb:
                    wandb_logger.finalize(status=_wandb_status)

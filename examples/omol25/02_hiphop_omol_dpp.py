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
from common import parse_args, get_network_params, get_trainer_params

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
    hippynn.settings.WARN_LOW_DISTANCES = True

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
                    'pin_memory': False,
                    'persistent_workers': False,
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
                                #TODO: Remove
                                detect_anomaly=True,
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

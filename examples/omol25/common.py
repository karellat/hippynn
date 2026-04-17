
import argparse
import yaml
# Parsing args
def parse_args():
    parser = argparse.ArgumentParser(description="HipHopNN Training Script")
    
    # Config file option
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI args override config file values.")
    
    # Directories
    parser.add_argument("--test-dir", type=str, default="/home/karella/Projects/hippynn/examples/ben-subset/test",
                        help="Directory for test outputs and logs")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints (default: {test_dir}/checkpoints)")
    
    # Dataset paths
    parser.add_argument("--train-path", type=str, default=["/home/karella/Projects/hippynn/dataset/omol_bensneutral_train", "/home/karella/Projects/hippynn/dataset/opoly_bens_train"],
                        help="Path to training dataset")
    parser.add_argument("--val-path", type=str, default=["/home/karella/Projects/hippynn/dataset/omol_bensneutral_val", "/home/karella/Projects/hippynn/dataset/opoly_bens_val"],
                        help="Path to validation dataset")
    parser.add_argument("--test-path", type=str, default="/home/karella/Projects/hippynn/test",
                        help="Path to test dataset")
    parser.add_argument("--henergy-init-path", type=str, default=None,
                        help="Path to hierarchical energy init file (default: {test_dir}/hierarchical_energy_init.pt)")
    parser.add_argument("--normalization-yaml-path", type=str, default="/home/karella/Projects/hippynn/examples/omol25/uma_v1_hof_lin_refs.yaml",
                        help="Path to YAML file with normalization reference values")
    
    # Training parameters
    parser.add_argument("--dl-num-workers", type=int, default=10,
                        help="Number of dataloader workers")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPU devices")
    parser.add_argument("--profiler", action="store_true", default=False,
                        help="Enable PyTorch Lightning profiler")
    parser.add_argument("--nodes", type=int, default=1,
                        help="Number of compute nodes")
    parser.add_argument("--max-epochs", type=int, default=80,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=256,
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
    # Debugging parameters
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode with limited batches")
    parser.add_argument("--debug-batches-per-epoch", type=int, default=10,
                        help="Number of batches to use for debugging per epoch")
    parser.add_argument("--profile-epochs", type=int, default=10,
                        help="Number of epochs to profile")
    parser.add_argument("--profile-workers", type=int, default=0,
                        help="Number of dataloader workers to use during profiling (set to 0 for main process only)")   

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
    parser.add_argument("--max-species", type=int, default=None,
                        help="Maximum number of species (elements), +1 for the zero padding.")
    parser.add_argument("--species", type=list, default=[1, 6, 7, 8, 9, 17],
                        help="List of atomic numbers to include as species")
    
    # WandB parameters
    parser.add_argument("--wandb", action="store_true", default=True,  
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


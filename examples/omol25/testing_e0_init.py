

import torch

from hippynn.graphs.nodes.base.node_functions import _BaseNode

from hippynn.graphs.nodes.base import _BaseNode
from hippynn.graphs.nodes.tags import Encoder
from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode, ForceNode
from hippynn.graphs import find_unique_relative, Predictor

from hippynn.networks.hipnn import compute_hipnn_e0, compute_hipnn_e0_sequentially

torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Don't try this if you want CPU training!

import hippynn

### Put ourselves in a working directory for the model
netname = "TEST_MY_FIRST_QM7_MODEL"

# Log the output of python to `training_log.txt`
with hippynn.tools.active_directory(netname):

    with hippynn.tools.log_terminal("training_log.txt", "wt"):

        # Hyperparameters for the network

        network_params = {
            # Note: First element is the blank species
            "possible_species": [0, 1, 6, 7, 8, 16],  # Z values of the elements
            "n_features": 20,  # Number of neurons at each layer
            "n_sensitivities": 20,  # Number of sensitivity functions in an interaction layer
            "dist_soft_min": 1.6,  #
            "dist_soft_max": 10.0,
            "dist_hard_max": 12.5,
            "n_interaction_layers": 2,  # Number of interaction blocks
            "n_atom_layers": 3,  # Number of atom layers in an interaction block
        }

        # Define a model

        from hippynn.graphs import inputs, networks, targets, physics

        # model inputs
        species = inputs.SpeciesNode(db_name="Z")
        positions = inputs.PositionsNode(db_name="R")

        # Model computations
        network = networks.Hipnn("HIPNN", (species, positions), module_kwargs=network_params)
        henergy = targets.HEnergyNode("HEnergy", network)
        molecule_energy = henergy.mol_energy
        molecule_energy.db_name = "T"
        hierarchicality = henergy.hierarchicality

        # define loss quantities
        from hippynn.graphs import loss

        rmse_energy = loss.MSELoss.of_node(molecule_energy) ** (1 / 2)
        mae_energy = loss.MAELoss.of_node(molecule_energy)
        rsq_energy = loss.Rsq.of_node(molecule_energy)

        ### More advanced usage of loss graph

        pred_per_atom = physics.PerAtom("PeratomPredicted", (molecule_energy, species)).pred
        true_per_atom = physics.PerAtom("PeratomTrue", (molecule_energy.true, species.true))
        mae_per_atom = loss.MAELoss(pred_per_atom, true_per_atom)

        ### End more advanced usage of loss graph

        loss_error = rmse_energy + mae_energy

        rbar = loss.Mean.of_node(hierarchicality)
        l2_reg = loss.l2reg(network)
        loss_regularization = 1e-6 * l2_reg + rbar  # L2 regularization and hierarchicality regularization

        train_loss = loss_error + loss_regularization

        # Validation losses are what we check on the data between epochs -- we can only train to
        # a single loss, but we can check other metrics too to better understand how the model is training.
        # There will also be plots of these things over time when training completes.
        validation_losses = {
            "T-RMSE": rmse_energy,
            "T-MAE": mae_energy,
            "T-RSQ": rsq_energy,
            "TperAtom MAE": mae_per_atom,
            "T-Hier": rbar,
            "L2Reg": l2_reg,
            "Loss-Err": loss_error,
            "Loss-Reg": loss_regularization,
            "Loss": train_loss,
        }
        early_stopping_key = "Loss-Err"

        from hippynn.experiment import assemble_for_training

        # This piece of code glues the stuff together as a pytorch model,
        # dropping things that are irrelevant for the losses defined.
        training_modules, db_info = assemble_for_training(train_loss, validation_losses)

        max_batch_size = 12
        database_params = {
            "name": "qm7",  # Prefix for arrays in folder
            "directory": "/home/karella/Projects/hippynn/dataset",
            "quiet": False,
            "test_size": 0.1,
            "valid_size": 0.1,
            "seed": 2001,
            # How many samples from the training set to use during evaluation
            **db_info,  # Adds the inputs and targets names from the model as things to load
        }

        from hippynn.databases import DirectoryDatabase

        database = DirectoryDatabase(**database_params)

        # Now that we have a database and a model, we can
        # Fit the non-interacting energies by examining the database.

        # Cut the hierarchical energy term out for testing
        energy_module = henergy
        encoder = None
        species_name = None
        energy_name = None
        for peratom in [False, True]:
            if isinstance(energy_module, _BaseNode):
                if encoder is None:
                    encoder = find_unique_relative(energy_module, Encoder, "Constructing E0 Values")
                if species_name is None:
                    species_name = find_unique_relative(energy_module, SpeciesNode, "Constructing E0 Values").db_name
                if energy_name is None:
                    energy_name = energy_module.main_output.db_name

                energy_module = energy_module.torch_module

            if isinstance(encoder, _BaseNode):
                encoder = encoder.torch_module

            # If model has E0 term, set its initial value using the database provided
            if not energy_module.first_is_interacting:
                if database is None:
                    raise ValueError("Database must be provided if model includes E0 energy term.")
            
                # TODO: Switch this and test on known datasets
                if database.is_in_memory: 
                    # Load all training data at once
                    train_data = database.splits["train"]

                    z_vals = train_data[species_name]
                    t_vals = train_data[energy_name]

                    encoder.to(t_vals.device)
                    eovals_memory = compute_hipnn_e0(encoder, z_vals, t_vals, peratom=peratom)
                    eovals_sequential = compute_hipnn_e0_sequentially(encoder,
                                                        database,
                                                        species_name=species_name,
                                                        energy_name=energy_name,
                                                        peratom=peratom)
                    # Check that both methods give the same result
                    torch.testing.assert_close(eovals_memory, eovals_sequential, msg="E0 values computed from in-memory and sequential methods do not match peratom={peratom}.")

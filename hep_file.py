def _find_h5_files(base_string: str ) -> tuple:
    egam1_str = base_string + "user.amoranch.361106.e3601_s3681_r13167_p4940.egd.electrons-cell_trial.EGAM1.25_0_63.3afd5d9_output.h5"
    egam7_str = base_string + "user.amoranch.423300.e3848_s3681_r13167_p4940.egd.electrons-cell_trial.EGAM7.25_0_63.3afd5d9_output.h5"
    data_str = base_string + "user.amoranch.data15_13TeV.data15_13TeV.egd.electrons-cells-data.EGAM1.25_0_63.44b1f75_output.h5"
    name_strings = ["MC_EGAM1", "MC_EGAM7", "DATA_EGAM", "MC_EGAM"]
    full_strings = [egam1_str, egam7_str, data_str]
    h5_files_dir = data_loading.create_path_lists(full_strings, outputs=name_strings, mixing_list = [(0, 1)], mixing_lengths = [(12, 6)], seed=0)
    h5_files_egam1 = h5_files_dir[name_strings[0]]
    h5_files_egam7 = h5_files_dir[name_strings[1]]
    h5_files_data = h5_files_dir[name_strings[-2]]
    h5_files_egam = h5_files_dir[name_strings[-1]]

    return h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam

def _train():
    # Choose two folders from the same particle type and simulation category.
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    base_string = "/groups/hep/morancho/ATLAS_Open_Data/Zee_Analysis/h5/"
    h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam = _find_h5_files(base_string)
    h5_files_data = h5_files_data[:20]

    valid_fields = {"egammas": ["pt", "eta", "phi", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"], "truth": ["pt", "eta", "phi"]}
    variables = ["pt"]

    for variable in variables:
        dataset_class = dataset_classes.ZPairDataset if variable in ["z", "origin", "origin-all"] else dataset_classes.H5EgammaDataset
        nn_model = neural_models.ParticlePairNetwork if variable in ["z", "origin", "origin-all"] else neural_models.ParticleOriginNetwork
        loss_class = torch.nn.BCEWithLogitsLoss() if variable in ["origin", "origin-all", "type", "isolation"] else training_classes.AngularMSELoss() if variable in ["phi"] else training_classes.LogCoshLoss() if variable in ["pt", "z"] else torch.nn.MSELoss()
        if variable in ["pt", "eta", "phi"]:
            variance_loss_class = training_classes.AngularGaussianVarianceNLLLoss() if variable in ["phi"] else training_classes.RobustGaussianVarianceNLLLoss() if variable in ["pt"] else training_classes.GaussianVarianceNLLLoss()

        field = [["truthOrigin", "pt", "eta", "phi"], ["passid_Loose", "pt", "eta", "phi"]] if variable in ["origin", "origin-all"] else [["truthType", "passid_VeryLoose"], ["passid_Medium", "passid_VeryLoose"]] if variable == "type" else ["passiso_Loose_VarRad", "passiso_Loose_VarRad"] if variable == "isolation" else [["pt", "eta", "phi"], ["pt", "eta", "phi"]] if variable == "z" else [[variable, "truthType"], [variable, "passid_Tight"]]
        output_size = 1 #if variable in ["origin", "pt", "z"] else 2

        binary_target = 13 if variable in ["origin", "origin-all"] else 2 if variable == "type" else 1 if variable == "isolation" else False
        norm_target = True if variable in ["origin"] else False
        z_target = True if variable == "z" else False
        log_target = False
        use_mask = "../Masters_project/models_library/model_type.pth" if variable in ["pt", "eta", "phi"] else False

        variable_name = "pair_" + variable if variable in ["z", "origin", "origin-all"] else variable
        if variable != "origin":
            model, train_losses, val_losses, _, _, _ = model_training.compressed_training_function(
                h5_files = h5_files_egam,
                mix_files = h5_files_data,
                mixture_ratio = 0.66,
                dataset_class = dataset_class,
                nn_model = nn_model,
                model_kwargs = {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 8], "event_hidden_sizes" : [64, 64, 16, 4], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : output_size},
                loss_function = loss_class,
                optimizer_class = torch.optim.Adam,
                batch_size = 4096,
                num_workers = 7,
                exclude_features = ["truth"],
                exclude_fields = valid_fields,
                y_source = "truth",
                y_field = field[0],
                fallback_source = "egammas",
                fallback_field = field[1],
                valid_fields = valid_fields,
                log_target = log_target,
                binary_target = binary_target,
                z_target = z_target,
                norm_target = norm_target,
                pin_memory = True,
                num_epochs = 5,
                sample_size = 1_000_000,
                #domain_adversary = True,
                #domain_loss_function = torch.nn.BCEWithLogitsLoss(),
                #domain_loss_weight = 0.05,
                #gradient_reversal_strength = 0.50,
                mixed_target_loss_weight = 0.80,
                domain_input = True,
                domain = "bool",
                periodic_save_interval = 2,
                periodic_save_path = f"../Masters_project/models_library/model_{variable_name}.pth",
                use_mask = use_mask,
                verbosity = 0
            )
            torch.save(model.state_dict(), f"../Masters_project/models_library/model_{variable_name}.pth")

            if variable in ["pt", "eta", "phi"]:
                model_var, train_losses_var, val_losses_var, _, _, _ = model_training.compressed_training_function(
                    h5_files = h5_files_egam,
                    mix_files = h5_files_data,
                    mixture_ratio = 0.66,
                    dataset_class = dataset_class,
                    nn_model = nn_model,
                    model_kwargs = {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 8], "event_hidden_sizes" : [64, 64, 16, 4], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : output_size},
                    loss_function = variance_loss_class,
                    optimizer_class = torch.optim.Adam,
                    batch_size = 4096,
                    num_workers = 7,
                    exclude_features = ["truth"],
                    exclude_fields = valid_fields,
                    y_source = "truth",
                    y_field = field[0],
                    fallback_source = "egammas",
                    fallback_field = field[1],
                    valid_fields = valid_fields,
                    log_target = log_target,
                    binary_target = binary_target,
                    z_target = z_target,
                    norm_target = norm_target,
                    pin_memory = True,
                    num_epochs = 5,
                    sample_size = 1_000_000,
                    #domain_adversary = True,
                    #domain_loss_function = torch.nn.BCEWithLogitsLoss(),
                    #domain_loss_weight = 0.05,
                    #gradient_reversal_strength = 0.50,
                    mixed_target_loss_weight = 0.80,
                    domain_input = True,
                    domain = "bool",
                    periodic_save_interval = 2,
                    periodic_save_path = f"../Masters_project/models_library/model_{variable_name}_logvar.pth",
                    use_mask = use_mask,
                    variance_target = model,
                    verbosity = 0
                )
                torch.save(model_var.state_dict(), f"../Masters_project/models_library/model_{variable_name}_logvar.pth")

                fig = plt.figure(figsize=(10, 6))
                plt.plot(np.arange(len(train_losses_var)), train_losses_var, label="Training Loss")
                plt.plot(np.arange(len(val_losses_var)), val_losses_var, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Training and Validation Loss for {variable_name}")
                plt.legend()
                misc_functions.save_plot(f"../Masters_project/plots/{date.today()}/loss_plot_{variable_name}_logvar.png", fig)
                plt.close()
        else:
            model, train_losses, val_losses, _, _, _ = model_training.compressed_training_function(
                h5_files = h5_files_egam,
                mix_files = h5_files_data,
                mixture_ratio = 0.66,
                dataset_class = dataset_class,
                nn_model = nn_model,
                model_kwargs = {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 8], "event_hidden_sizes" : [64, 64, 16, 4], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : output_size},
                loss_function = loss_class,
                optimizer_class = torch.optim.Adam,
                batch_size = 4096,
                num_workers = 7,
                exclude_features = ["truth"],
                exclude_fields = valid_fields,
                y_source = "egammas",
                y_field = ["truthOrigin", "pt", "eta", "phi", "z0", "d0", "sig_z0", "truthType"],
                fallback_field = ["passid_Loose", "pt", "eta", "phi", "z0", "d0", "sig_z0", "passid_Tight"],
                valid_fields = valid_fields,
                log_target = log_target,
                binary_target = binary_target,
                z_target = z_target,
                norm_target = norm_target,
                pin_memory = True,
                num_epochs = 5,
                sample_size = 1_000_000,
                #domain_adversary = True,
                #domain_loss_function = torch.nn.BCEWithLogitsLoss(),
                #domain_loss_weight = 0.05,
                #gradient_reversal_strength = 0.50,
                #mixed_target_loss_weight = 0.10,
                domain_input = True,
                periodic_save_interval = 2,
                periodic_save_path = f"../Masters_project/models_library/model_{variable_name}.pth",
                z_pid = (True, f"../Masters_project/models_library/model_type.pth", f"../Masters_project/models_library/model_isolation.pth"),
                verbosity = 0
            )            

            torch.save(model.state_dict(), f"../Masters_project/models_library/model_{variable_name}.pth")

        fig = plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(train_losses)), train_losses, label="Training Loss")
        plt.plot(np.arange(len(val_losses)), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss for {variable_name}")
        plt.legend()
        misc_functions.save_plot(f"../Masters_project/plots/{date.today()}/loss_plot_{variable_name}.png", fig)
        plt.close()

def _write(a, b, c):
    base_string = "/groups/hep/morancho/ATLAS_Open_Data/Zee_Analysis/h5/"
    h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam = _find_h5_files(base_string)

    validation_split = 0.25
    train_size_egam = int((1 - validation_split) * len(h5_files_egam))
    train_size_data = int((1 - validation_split) * len(h5_files_data))

    models_path = Path("./models_library")
    csv_path = Path("./csv_library")
    csv_path.mkdir(parents=True, exist_ok=True)

    valid_fields = {"egammas": ["pt", "eta", "phi", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"], "truth": ["pt", "eta", "phi"]}
    model_configs = {"z" : [False, False, False], "pt" : [False, False, False], "eta" : [False, False, False], "phi" : [False, False, False], "origin" : [False, False, False], "type" : [False, False, False], "isolation" : [False, False, False]}
    if a:
        data_loading.models_to_csv_files(
            models_path,
            csv_path,
            h5_files_egam[train_size_egam:],
            h5_files_data[train_size_data:],
            valid_fields = valid_fields,
            batch_size = 4096,
            num_workers = 7,
            pin_memory = False,
            max_rows = 1_000_000,
            model_type = "egamma",
            model_configs = model_configs,
        )
    if b:
        data_loading.models_to_csv_files(
            models_path,
            csv_path,
            h5_files_egam[train_size_egam:],
            h5_files_data[train_size_data:],
            valid_fields = valid_fields,
            batch_size = 4096,
            num_workers = 7,
            pin_memory = False,
            max_rows = 1_000_000,
            model_type = "pair",
            model_configs = model_configs,
        )
    if c:
        data_loading.datasets_to_csv_files(
            h5_files_egam[train_size_egam:],
            h5_files_data[train_size_data:],
            csv_path,
            valid_fields = valid_fields,
            batch_size = 4096,
            num_workers = 7,
            pin_memory = False,
            max_rows = 1_000_000,
            )

def _plot():
    base_string = f"/groups/hep/wrt770/Master_Project/git_shared/Masters_project"
    csv_string = f"{base_string}/csv_library"
    atlas_mass_paths = [f"{csv_string}/z_pair_atlas_mc.csv", f"{csv_string}/z_pair_atlas_data.csv"]
    atlas_origin_paths = [f"{csv_string}/origin_pair_atlas_mc.csv", f"{csv_string}/origin_pair_atlas_data.csv"]
    predictions_paths = [f"{csv_string}/origin_pair_predictions_mc.csv", f"{csv_string}/origin_pair_predictions_data.csv"]
    origin_truth_path = f"{csv_string}/origin_pair_truth_mc.csv"
    z_mass_truth_path = f"{csv_string}/z_pair_truth_mc.csv"
    save_namespace = ["mc", "data"]
    plotting_path = f"{base_string}/plots/{date.today()}/"

    N_pred_true, N_cut_true, N_pred_fit_mc, N_cut_fit_mc, N_pred_fit_data, N_cut_fit_data = misc_functions.create_pseudo_roc_curve(
        predictions_paths, atlas_mass_paths, atlas_origin_paths, origin_truth_path, z_mass_truth_path, plotting_path, 
        save_namespace=save_namespace, bins = np.arange(54.75, 125.25, 0.5), n_total = 100)
    
    print(f"N_tp and N_fp for the cut threshold: {N_cut_true}, {N_cut_fit_mc}, {N_cut_fit_data}")
    print(f"N_tp and N_fp for the pseudo-roc curve: {N_pred_true}, {N_pred_fit_mc}, {N_pred_fit_data}")


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from functions_library import dataset_classes, neural_models, training_classes, model_training, data_loading, misc_functions
    from pathlib import Path
    from datetime import date
    import time

    time_start = time.perf_counter()

    _train()
    #_write(0, 0, 1)
    #_plot()
    time_end = time.perf_counter()
    print(f"Total execution time: {time_end - time_start:.2f} seconds")
    print("Training completed successfully.")
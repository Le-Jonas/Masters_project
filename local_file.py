def _find_h5_files(base_string: str ) -> tuple:
    egam1_str = base_string + "MC_EGAM1"
    egam7_str = base_string + "MC_EGAM7"
    data_str = base_string + "DATA_EGAM"
    name_strings = ["MC_EGAM1", "MC_EGAM7", "DATA_EGAM", "MC_EGAM"]
    full_strings = [egam1_str, egam7_str, data_str]
    h5_files_dir = data_loading.create_path_lists(full_strings, outputs=name_strings, mixing_list = [(0, 1)], seed=42)
    h5_files_egam1 = h5_files_dir[name_strings[0]]
    h5_files_egam7 = h5_files_dir[name_strings[1]]
    h5_files_data = h5_files_dir[name_strings[-2]]
    h5_files_egam = h5_files_dir[name_strings[-1]]

    return h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam


def _train():
    # Choose two folders from the same particle type and simulation category.
    base_string = f"C:/Users/jonas/{system_string}/Main/h5/"
    h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam = _find_h5_files(base_string)

    valid_fields = {"egammas": ["pt", "eta", "phi", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"], "truth": ["pt", "eta", "phi"]}
    variables = ["type"]

    for variable in variables:
        dataset_class = dataset_classes.ZPairDataset if variable in ["z", "origin", "origin-all"] else dataset_classes.H5EgammaDataset
        nn_model = neural_models.ParticlePairNetwork if variable in ["z", "origin", "origin-all"] else neural_models.ParticleOriginNetwork
        loss_class = torch.nn.BCEWithLogitsLoss() if variable in ["origin", "origin-all", "type", "isolation"] else training_classes.AngularMSELoss() if variable in ["phi"] else training_classes.LogCoshLoss() if variable in ["z", "pt"] else torch.nn.MSELoss()
        if variable in ["pt", "eta", "phi"]:
            variance_loss_class = training_classes.AngularGaussianVarianceNLLLoss() if variable in ["phi"] else training_classes.RobustGaussianVarianceNLLLoss() if variable in ["pt"] else training_classes.GaussianVarianceNLLLoss()

        field = [["truthOrigin", "pt", "eta", "phi"], ["passid_Loose", "pt", "eta", "phi"]] if variable in ["origin", "origin-all"] else [["truthType", "passid_VeryLoose"], ["passid_Medium", "passid_VeryLoose"]] if variable == "type" else ["passiso_Loose_VarRad", "passiso_Loose_VarRad"] if variable == "isolation" else [["pt", "eta", "phi"], ["pt", "eta", "phi"]] if variable == "z" else [[variable, "truthType"], [variable, "passid_Tight"]]

        binary_target = 13 if variable in ["origin", "origin-all"] else 2 if variable == "type" else 1 if variable == "isolation" else False
        norm_target = True if variable in ["origin"] else False
        gev_target = True if variable in ["z", "pt"] else False
        z_target = True if variable == "z" else False
        use_mask = "./models_library/model_type.pth" if variable in ["pt", "eta", "phi"] else False
        model_kwargs = {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 64, 64], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [64, 32, 32, 8], "output_size" : 1}


        variable_name = "pair_" + variable if variable in ["z", "origin", "origin-all"] else variable
        if variable != "origin":
            model, train_losses, val_losses, _, _, _ = model_training.compressed_training_function(
                h5_files = h5_files_egam,
                mix_files = h5_files_data,
                mixture_ratio = 0.66,
                dataset_class = dataset_class,
                nn_model = nn_model,
                model_kwargs = model_kwargs,
                loss_function = loss_class,
                optimizer_class = torch.optim.Adam,
                batch_size = 1024,
                num_workers = 7,
                exclude_features = ["truth"],
                exclude_fields = valid_fields,
                y_source = "truth",
                y_field = field[0],
                fallback_source = "egammas",
                fallback_field = field[1],
                valid_fields = valid_fields,
                binary_target = binary_target,
                z_target = z_target,
                norm_target = norm_target,
                gev_target = gev_target,
                pin_memory = False,
                num_epochs = 15,
                #domain_adversary = True,
                #domain_loss_function = torch.nn.BCEWithLogitsLoss(),
                #domain_loss_weight = 0.05,
                #gradient_reversal_strength = 0.50,
                mixed_target_loss_weight = 1.0,
                domain_input = True,
                domain = "bool",
                periodic_save_interval = 2,
                periodic_save_path = f"./models_library/model_{variable_name}.pth",
                means_library_path = f"./means_library",
                use_mask = use_mask,
                npy_training = "./subset_library",
                verbosity = 1,
            )
            torch.save(model.state_dict(), f"./models_library/model_{variable_name}.pth")

            if variable in ["pt", "eta", "phi"]:
                model_var, train_losses_var, val_losses_var, _, _, _ = model_training.compressed_training_function(
                    h5_files = h5_files_egam,
                    mix_files = h5_files_data,
                    mixture_ratio = 0.66,
                    dataset_class = dataset_class,
                    nn_model = nn_model,
                    model_kwargs = model_kwargs,
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
                    binary_target = binary_target,
                    z_target = z_target,
                    norm_target = norm_target,
                    gev_target = gev_target,
                    pin_memory = False,
                    num_epochs = 3,
                    #domain_adversary = True,
                    #domain_loss_function = torch.nn.BCEWithLogitsLoss(),
                    #domain_loss_weight = 0.05,
                    #gradient_reversal_strength = 0.50,
                    mixed_target_loss_weight = 1.0,
                    domain_input = True,
                    domain = "bool",
                    periodic_save_interval = 2,
                    periodic_save_path = f"./models_library/model_{variable_name}_logvar.pth",
                    use_mask = use_mask,
                    means_library_path = f"./means_library",
                    variance_target = model
                )
                torch.save(model_var.state_dict(), f"./models_library/model_{variable_name}_logvar.pth")

                fig = plt.figure(figsize=(10, 6))
                plt.plot(np.arange(len(train_losses_var)), train_losses_var, label="Training Loss")
                plt.plot(np.arange(len(val_losses_var)), val_losses_var, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Training and Validation Loss for {variable_name}")
                plt.legend()
                plotting_funcs.save_plot(f"./plots/{date.today()}/loss_plot_{variable_name}_logvar.png", fig)
                plt.close()
        else:
            model, train_losses, val_losses, _, _, _ = model_training.compressed_training_function(
                h5_files = h5_files_egam,
                mix_files = h5_files_data,
                mixture_ratio = 0.66,
                dataset_class = dataset_class,
                nn_model = nn_model,
                model_kwargs = model_kwargs,
                loss_function = loss_class,
                optimizer_class = torch.optim.Adam,
                batch_size = 4096,
                num_workers = 7,
                exclude_features = ["truth"],
                exclude_fields = valid_fields,
                y_source = "egammas",
                y_field = ["truthOrigin", "pt", "eta", "phi", "z0", "sig_z0"],
                fallback_field = ["passid_Loose", "pt", "eta", "phi", "z0", "sig_z0"],
                valid_fields = valid_fields,
                binary_target = binary_target,
                z_target = z_target,
                norm_target = norm_target,
                pin_memory = False,
                num_epochs = 5,
                #domain_adversary = True,
                #domain_loss_function = torch.nn.BCEWithLogitsLoss(),
                #domain_loss_weight = 0.05,
                #gradient_reversal_strength = 0.50,
                mixed_target_loss_weight = 0.80,
                domain_input = True,
                periodic_save_interval = 2,
                periodic_save_path = f"./models_library/model_{variable_name}.pth",
                means_library_path = f"./means_library",
                z_pid = (True, f"./models_library/model_type.pth", f"./models_library/model_isolation.pth")
            )            

            torch.save(model.state_dict(), f"./models_library/model_{variable_name}.pth")

        fig = plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(train_losses)), train_losses, label="Training Loss")
        plt.plot(np.arange(len(val_losses)), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss for {variable_name}")
        plt.legend()
        plotting_funcs.save_plot(f"./plots/{date.today()}/loss_plot_{variable_name}.png", fig)
        plt.close()
    print("Training completed successfully.")

def _write(write_string):
    base_string = f"C:/Users/jonas/{system_string}/Main/h5/"
    h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam = _find_h5_files(base_string)

    validation_split = 0.25
    train_size_egam = int((1 - validation_split) * len(h5_files_egam))
    train_size_data = int((1 - validation_split) * len(h5_files_data))

    models_path = Path("./models_library")
    csv_path = Path("./csv_library")
    csv_path.mkdir(parents=True, exist_ok=True)
    subset_path = Path("./subset_library")
    subset_path.mkdir(parents=True, exist_ok=True)
    
    valid_fields = {"egammas": ["pt", "eta", "phi", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"], "truth": ["pt", "eta", "phi"]}
    model_configs = {"z" : [False, True], "pt" : [False, True], "eta" : [False, False], "phi" : [False, False], "origin" : [False, False], "type" : [False, False], "isolation" : [False, False]}
    if write_string == "model_to_csv":
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
            domain_input = True,
        )
    if write_string == "model_to_csv":
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
            domain_input = True,
        )
    if write_string == "data_to_csv":
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
    
    if write_string == "data_to_subset":
        file_lists = [[h5_files_egam1, "sig"], [h5_files_egam7, "bkg"], [h5_files_data, "data"]]
        for list in file_lists:
            if list[1] != "data":
                y_source = "egammas"
                y_field = ["truthType", "passid_VeryLoose"]
            else:
                y_source = "egammas"
                y_field = ["passid_Medium", "passid_VeryLoose"]
            data_loading.save_data_subset(
                list[0],
                subset_path,
                list[1],
                valid_fields = valid_fields,
                exclude_features = ["truth"],
                exclude_fields = valid_fields,
                sample_size = 20_000,
                y_source = y_source,
                y_field = y_field,
            )
    print("Writing completed successfully.")

def _plot():
    base_string = f"C:/Users/jonas/{system_string}/Main/Masters_project"
    csv_string = f"{base_string}/csv_library"
    atlas_mass_paths = [f"{csv_string}/z_pair_atlas_mc.csv", f"{csv_string}/z_pair_atlas_data.csv"]
    atlas_origin_paths = [f"{csv_string}/origin_pair_atlas_mc.csv", f"{csv_string}/origin_pair_atlas_data.csv"]
    predictions_paths = [f"{csv_string}/origin_pair_predictions_mc.csv", f"{csv_string}/origin_pair_predictions_data.csv"]
    origin_truth_path = f"{csv_string}/origin_pair_truth_mc.csv"
    z_mass_truth_path = f"{csv_string}/z_pair_truth_mc.csv"
    save_namespace = ["mc", "data"]
    plotting_path = f"{base_string}/plots/{date.today()}/"

    N_pred_true, N_cut_true, N_pred_fit_mc, N_cut_fit_mc, N_pred_fit_data, N_cut_fit_data = plotting_funcs.create_pseudo_roc_curve(
        predictions_paths, atlas_mass_paths, atlas_origin_paths, origin_truth_path, z_mass_truth_path, plotting_path, 
        save_namespace=save_namespace, bins = np.arange(54.75, 125.25, 0.5), n_total = 200, mc_signal_limit = 15_000)
    
    #print(f"N_tp and N_fp for the cut threshold: {N_cut_true}, {N_cut_fit_mc}, {N_cut_fit_data}")
    #print(f"N_tp and N_fp for the pseudo-roc curve: {N_pred_true}, {N_pred_fit_mc}, {N_pred_fit_data}")

    print("Plotting completed successfully.")

def _test(test_length: int):
    base_string = f"C:/Users/jonas/{system_string}/Main/h5/"
    h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam = _find_h5_files(base_string)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


    batch_size = 4096
    num_workers = 7
    pin_memory = False
    persistent_workers = True
    model_kwargs = {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 8], "event_hidden_sizes" : [64, 64, 16, 4], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1}
    valid_fields = {"egammas": ["pt", "eta", "phi", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"], "truth": ["pt", "eta", "phi"]}
    mask_model_path = "./models_library/model_type.pth"
    mixed_target_loss_weight = 0.80

    test_dataset = dataset_classes.H5EgammaDataset(
        h5_files_egam,
        mix_files = h5_files_data,
        mixture_ratio = 0.66,
        exclude_features = ["truth"],
        exclude_fields = valid_fields,
        y_source = "truth",
        y_field = ["pt", "truthType"],
        fallback_source = "egammas",
        fallback_field = ["pt", "passid_Tight"],
        valid_fields = valid_fields,
    )

    means, stds = misc_functions.compute_mean_std(test_dataset, sample_size=100_000, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, verbosity=0)
    sampler = data_loading.ShuffledContiguousBatchSampler(len(test_dataset), batch_size=batch_size, dataset=test_dataset, discard_remainder=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler = sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    particle_fields = [field for field in test_dataset.fields if not field.startswith("eventwise_")]
    event_fields = [field for field in test_dataset.fields if field.startswith("eventwise_")]

    event_size = len(event_fields) + 1
    particle_size = len(particle_fields)
    embedding_size = model_kwargs["particle_hidden_sizes"][-1] + model_kwargs["event_hidden_sizes"][-1]

    model = neural_models.ParticleOriginNetwork(
        particle_size,
        event_size,
        **model_kwargs
    ).to(device)

    mask_model = neural_models.ParticleOriginNetwork(
        particle_size,
        event_size,
        **model_kwargs
    ).to(device)
    mask_model.load_state_dict(torch.load(mask_model_path, map_location=device))

    loss_function = training_classes.LogCoshLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


    means_i, means_e = model_training._move_to_device(means, device)
    stds_i, stds_e = model_training._move_to_device(stds, device)

    mask_model.eval()
    for parameter in mask_model.parameters():
        parameter.requires_grad_(False)

    print(f"Warming up the loader and data files {test_length} batches")
    for batch, data in enumerate(test_loader):
        if batch >= test_length:
            break

    print(f"Starting time evaluation of loading only for {test_length} batches")
    time_start = time.perf_counter()
    for batch, data in enumerate(test_loader):
        if batch >= test_length:
            break
    time_end = time.perf_counter()
    print(f"Time taken to load {test_length} batches: {time_end - time_start:.2f} seconds")

    print(f"Starting time evaluation of loading and preprocessing for {test_length} batches")
    time_start = time.perf_counter()
    for batch, data in enumerate(test_loader):
        inputs, eventwise_features, targets, domain_targets, _ = model_training._prepare_training_inputs(data, device, (means_i, means_e), (stds_i, stds_e), mask_model=mask_model, verbosity=0, domain_input=True)
        if batch >= test_length:
            break
    time_end = time.perf_counter()
    print(f"Time taken to load and preprocess {test_length} batches: {time_end - time_start:.2f} seconds")

    print(f"Starting time evaluation of full training process for {test_length} batches")
    time_start = time.perf_counter()
    running_loss = 0.0
    samples_seen = 0
    for batch, data in enumerate(test_loader):
        inputs, eventwise_features, targets, domain_targets, _ = model_training._prepare_training_inputs(data, device, (means_i, means_e), (stds_i, stds_e), mask_model=mask_model, verbosity=0, domain_input=True)
        optimizer.zero_grad()
        outputs = model_training._call_with_inputs(model, inputs, eventwise_features)
        targets = targets.to(dtype=outputs.dtype)

        loss = model_training._weighted_task_loss(
            loss_function,
            outputs,
            targets,
            domain_targets,
            mixed_target_loss_weight,
        )

        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        samples_seen += batch_size
        average_loss = running_loss / samples_seen

        if batch >= test_length:
            break
    time_end = time.perf_counter()
    print(f"Time taken to train on {test_length} batches: {time_end - time_start:.2f} seconds")
    print(f"Testing completed successfully.")

if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from functions_library import dataset_classes, neural_models, training_classes, model_training, data_loading, misc_functions, plotting_funcs
    from pathlib import Path
    from datetime import date
    import time

    #system_string = "Desktop/Masters" #Laptop path
    system_string = "Coding/Master" #Desktop path

    time_start = time.perf_counter()
    _train()
    #_write("model_to_csv")
    #_plot()
    #_test(100)
    time_end = time.perf_counter()
    print(f"Total execution time: {time_end - time_start:.2f} seconds")


#TESTING OUTPUT
#Warming up the loader and data files 100 batches
#Starting time evaluation of loading only for 100 batches
#Time taken to load 100 batches: 79.58 seconds
#Starting time evaluation of loading and preprocessing for 100 batches
#Time taken to load and preprocess 100 batches: 82.74 seconds
#Starting time evaluation of full training process for 100 batches
#Time taken to train on 100 batches: 85.04 seconds
#Testing completed successfully.
#Total execution time: 432.70 seconds
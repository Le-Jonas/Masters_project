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



def _train(n_epoch : int, save_model : bool = True, trial = None):
    global npy_lr_modify
    # Choose two folders from the same particle type and simulation category.
    base_string = f"C:/Users/jonas/{system_string}/Main/h5/"
    h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam = _find_h5_files(base_string)

    valid_fields = {"egammas": ["pt", "eta", "phi", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"], "truth": ["pt", "eta", "phi"]}

    epoch_callback = None
    if trial is not None:
        def epoch_callback(epoch, best_val_loss):
            trial.report(best_val_loss, step=epoch)
            if trial.should_prune():
                raise opt.TrialPruned()

    for variable in variables:
        dataset_class = variable_kwargs_dict[variable]["dataset"]
        nn_model = variable_kwargs_dict[variable]["nn_model"]
        loss_class = variable_kwargs_dict[variable]["loss_function"]
        binary_target = variable_kwargs_dict[variable]["binary_target"]
        norm_target = variable_kwargs_dict[variable]["norm_target"]
        gev_target = variable_kwargs_dict[variable]["gev_target"]
        field = variable_kwargs_dict[variable]["y_field"]
        try:
            logvar_name = variable.split("_")[1]
            if logvar_name == "logvar":
                variance_target = "./models_library/model_pt.pth"
        except IndexError:
            variance_target = False


        model_kwargs = model_kwargs_dict[variable]
        optimizer_kwargs = optimizer_kwargs_dict[variable]
        z_target = True if variable == "z" else False
        use_mask = "./models_library/model_type.pth" if variable in ["pt", "eta", "phi", "pt_logvar"] else False

        if npy_training is not None:
            if npy_lr_modify:
                optimizer_kwargs["lr"] *= 100
                npy_lr_modify = False

    
        variable_name = "pair_" + variable if variable in ["z", "origin", "origin-all"] else variable
        periodic_save_interval = 4 if save_model else None
        periodic_save_path = f"./models_library/model_{variable_name}.pth" if save_model else None
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
                optimizer_kwargs = optimizer_kwargs,
                batch_size = 4096,
                num_workers = 0,
                exclude_features = ["truth"],
                exclude_fields = valid_fields,
                y_field = field[0],
                fallback_field = field[1],
                valid_fields = valid_fields,
                binary_target = binary_target,
                z_target = z_target,
                norm_target = norm_target,
                gev_target = gev_target,
                pin_memory = False,
                num_epochs = n_epoch,
                #domain_adversary = True,
                #domain_loss_function = torch.nn.BCEWithLogitsLoss(),
                #domain_loss_weight = 0.05,
                #gradient_reversal_strength = 0.50,
                mixed_target_loss_weight = 1.0,
                domain_input = True,
                domain = "bool",
                periodic_save_interval = periodic_save_interval,
                periodic_save_path = periodic_save_path,
                means_library_path = f"./means_library",
                use_mask = use_mask,
                npy_training = npy_training,
                verbosity = 1,
                variance_target = variance_target,
                prefetch_factor = None,
                persistent_workers = False,
                epoch_callback = epoch_callback,
            )
            if save_model:
                torch.save(model.state_dict(), f"./models_library/model_{variable_name}.pth")

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
                optimizer_kwargs = optimizer_kwargs,
                batch_size = 2048,
                num_workers = 6,
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
                num_epochs = n_epoch,
                #domain_adversary = True,
                #domain_loss_function = torch.nn.BCEWithLogitsLoss(),
                #domain_loss_weight = 0.05,
                #gradient_reversal_strength = 0.50,
                mixed_target_loss_weight = 1.0,
                domain_input = True,
                periodic_save_interval = periodic_save_interval,
                periodic_save_path = periodic_save_path,
                means_library_path = f"./means_library",
                z_pid = (True, f"./models_library/model_type.pth", f"./models_library/model_isolation.pth"),
                epoch_callback = epoch_callback,
            )            

            if save_model:
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
    return min(val_losses)

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
    model_configs = {"z" : [False, True], "pt" : [False, True], "logvar" : [False, True], "eta" : [False, False], "phi" : [False, False], "origin" : [False, False], "type" : [False, False], "isolation" : [False, False]}
    if write_string == "model_to_csv":
        data_loading.models_to_csv_files(
            models_path,
            csv_path,
            model_kwargs_dict,
            h5_files_egam[train_size_egam:],
            h5_files_data[train_size_data:],
            valid_fields = valid_fields,
            batch_size = 2048,
            num_workers = 7,
            pin_memory = False,
            max_rows = 1_000_000,
            model_type = "egamma",
            model_configs = model_configs,
            domain_input = True,
            npy_training = npy_training,
            prediction_list = variables,
        )
    if write_string == "model_to_csv":
        data_loading.models_to_csv_files(
            models_path,
            csv_path,
            model_kwargs_dict,
            h5_files_egam[train_size_egam:],
            h5_files_data[train_size_data:],
            valid_fields = valid_fields,
            batch_size = 2048,
            num_workers = 7,
            pin_memory = False,
            max_rows = 1_000_000,
            model_type = "pair",
            model_configs = model_configs,
            npy_training = npy_training,
            domain_input = True,
            prediction_list = variables,
        )
    if write_string == "data_to_csv":
        data_loading.datasets_to_csv_files(
            h5_files_egam[train_size_egam:],
            h5_files_data[train_size_data:],
            csv_path,
            valid_fields = valid_fields,
            batch_size = 2048,
            num_workers = 7,
            pin_memory = False,
            max_rows = 1_000_000,
            )
    
    if write_string == "data_to_subset":
        file_lists = [[h5_files_egam1, "sig"], [h5_files_egam7, "bkg"], [h5_files_data, "data"]]
        for list in file_lists:
            y_field = ["passiso_Loose_VarRad"]
            fallback_field = ["passiso_Loose_VarRad"]
            data_loading.save_data_subset(
                list[0],
                subset_path,
                list[1],
                valid_fields = valid_fields,
                exclude_features = ["truth"],
                exclude_fields = valid_fields,
                sample_size = 50_000,
                y_field = y_field,
                fallback_field = fallback_field,
            )
    print("Writing completed successfully.")
    return 1

def _plot(write_string : str):
    if write_string == "pseudo_roc_plot":
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
    return 1

def _optimize(n_trials: int):

    if len(variables) != 1:
        raise ValueError("Hyperparameter optimization is only supported for a single variable at a time.")
    global npy_lr_modify

    def objective(trial):
        global npy_lr_modify
        # Define the hyperparameter search space
        if True:
            learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-3, log=True)
            #beta1 = trial.suggest_float('beta1', 0.25, 1.0)
            #beta2 = trial.suggest_float('beta2', 0.50, 1.0)
            #weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-1, log=True)

            optimizer_kwargs_dict[variables[0]]["lr"] = learning_rate
            #optimizer_kwargs_dict[variables[0]]["betas"] = (beta1, beta2)
            #optimizer_kwargs_dict[variables[0]]["weight_decay"] = weight_decay

            npy_lr_modify = True

        if False:
            particle_hidden_sizes = trial.suggest_int('particle_hidden_sizes', 64, 512, step=64)
            event_hidden_sizes = trial.suggest_int('event_hidden_sizes', 8, 64, step=8)
            consolidation_hidden_sizes = trial.suggest_int('consolidation_hidden_sizes', 32, min(256, particle_hidden_sizes//2), step=32)
            particle_hidden_length = trial.suggest_int('particle_hidden_length', 1, 6)
            event_hidden_length = trial.suggest_int('event_hidden_length', 1, 3)
            consolidation_hidden_length = trial.suggest_int('consolidation_hidden_length', 1, 5)

            model_kwargs_dict[variables[0]]["particle_hidden_sizes"] = [1024] + [particle_hidden_sizes] * particle_hidden_length + [particle_hidden_sizes // 2] * 2
            model_kwargs_dict[variables[0]]["event_hidden_sizes"] = [64] + [event_hidden_sizes] * event_hidden_length + [event_hidden_sizes // 4]
            model_kwargs_dict[variables[0]]["consolidation_hidden_sizes"] = [consolidation_hidden_sizes] * consolidation_hidden_length + [8]

        
        return _train(15, save_model = False, trial=trial)

    study = opt.create_study(
        direction='minimize',
        pruner=opt.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
    )
    study.optimize(objective, n_trials=n_trials)

    completed_trials = [
        trial for trial in study.trials
        if trial.state == opt.trial.TrialState.COMPLETE and trial.value is not None
    ]
    top_trials = sorted(completed_trials, key=lambda trial: trial.value)[:3]
    if not top_trials:
        raise RuntimeError("No Optuna trials completed successfully.")

    top_results = [
        {
            "rank": rank,
            "trial_number": trial.number,
            "value": trial.value,
            "params": trial.params,
        }
        for rank, trial in enumerate(top_trials, start=1)
    ]
    results_path = Path("./optuna_results") / f"top3_{variables[0]}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(top_results, results_file, indent=2)

    print(f"Best loss achieved: {study.best_value}")
    print(f"Best parameter values: {study.best_params}")
    print(f"Top {len(top_results)} trial results saved to {results_path}")
    print("Hyperparameter optimization completed successfully.")

    return 1

def _test_optimization(optuna_res : str | Path):
    global npy_lr_modify
    
    if len(variables) != 1:
        raise ValueError("Hyperparameter optimization testing is only supported for a single variable at a time.")

    optuna_res_path = Path(optuna_res) / f"top3_{variables[0]}.json"
    if not optuna_res_path.is_file():
        raise FileNotFoundError(f"Optuna results file not found: {optuna_res_path}")

    with optuna_res_path.open("r", encoding="utf-8") as results_file:
        top_results = json.load(results_file)

    if not top_results:
        raise ValueError("No trial results found in the Optuna results file.")

    for trial in top_results:
        params = trial["params"]
        #model_kwargs_dict[variables[0]]["particle_hidden_sizes"] = [1024] + [params['particle_hidden_sizes']] * params['particle_hidden_length'] + [params['particle_hidden_sizes'] // 2] * 2
        #model_kwargs_dict[variables[0]]["event_hidden_sizes"] = [64] + [params['event_hidden_sizes']] * params['event_hidden_length'] + [params['event_hidden_sizes'] // 4]
        #model_kwargs_dict[variables[0]]["consolidation_hidden_sizes"] = [params['consolidation_hidden_sizes']] * params['consolidation_hidden_length'] + [8]
        optimizer_kwargs_dict[variables[0]]["lr"] = params['learning_rate']
        optimizer_kwargs_dict[variables[0]]["betas"] = (params['beta1'], params['beta2'])
        optimizer_kwargs_dict[variables[0]]["weight_decay"] = params['weight_decay']
        npy_lr_modify = True

        losses = []
        for _ in range(3):
            loss = _train(20, save_model=False)
            losses.append(loss)
        avg_loss = sum(losses) / len(losses)
        min_loss = min(losses)
        print(f"Trial {trial['trial_number']} - Average Loss over 3 runs: {avg_loss:.6f}, Minimum Loss: {min_loss:.6f}")

if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from functions_library import dataset_classes, neural_models, training_classes, model_training, data_loading, misc_functions, plotting_funcs
    from functions_library.models_dicts import model_kwargs_dict, fields_dict, optimizer_kwargs_dict, variable_kwargs_dict
    from pathlib import Path
    from datetime import date
    import time
    import optuna as opt
    import json

    #system_string = "Desktop/Masters" #Laptop path
    system_string = "Coding/Master" #Desktop path

    npy_training = "./subset_library" #Method to use preprocessed npy files for training and writing. Much faster but only small subset
    npy_lr_modify = True
    variables = ["isolation"]

    time_start = time.perf_counter()
    #for i in range(3):
    #    loss = _train(20, save_model=False)
    #    print(f"Run {i+1} - Loss: {loss:.6f}")
    #_write("data_to_subset")
    #_write("model_to_csv")
    #_plot("pseudo_roc_plot")
    #_test(100)
    _optimize(10)
    #_test_optimization("./optuna_results")
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
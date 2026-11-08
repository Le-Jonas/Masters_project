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
    base_string = "C:/Users/jonas/Coding/Master/Main/h5/"
    h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam = _find_h5_files(base_string)

    valid_fields = {"egammas": ["pt", "eta", "phi", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight"], "truth": ["pt", "eta", "phi"]}
    variables = ["pt", "eta", "phi"]

    for variable in variables:
        dataset_class = dataset_classes.ZPairDataset if variable in ["z", "origin"] else dataset_classes.H5EgammaDataset
        nn_model = neural_models.ParticlePairNetwork if variable in ["z", "origin"] else neural_models.ParticleOriginNetwork
        loss_class = torch.nn.BCEWithLogitsLoss() if variable == "origin" else training_classes.AngularMSELoss() if variable in ["phi"] else torch.nn.MSELoss()

        field = ["truthOrigin", "passid_Medium"] if variable == "origin" else [["pt", "eta", "phi"], ["pt", "eta", "phi"]] if variable == "z" else [variable, variable]
        output_size = 1 #if variable in ["origin", "pt", "z"] else 2

        binary_target = 13 if variable == "origin" else False
        norm_target = True if variable in ["z", "pt"] else False
        z_target = True if variable == "z" else False
        log_target = False
        use_mask = False# = "./models_library/model_pair_origin.pth" if variable == "z" else False

        variable_name = "pair_" + variable if variable in ["z", "origin"] else variable

        model, train_losses, val_losses, means, stds, norm_target = model_training.compressed_training_function(
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
            pin_memory = False,
            num_epochs = 5,
            sample_size = 1_000_000,
            domain_adversary = True,
            domain_loss_function = torch.nn.BCEWithLogitsLoss(),
            domain_loss_weight = 0.05,
            gradient_reversal_strength = 0.50,
            mixed_target_loss_weight = 0.20,
            domain = "bool",
            periodic_save_interval = 2,
            periodic_save_path = f"./models_library/model_{variable_name}.pth",
            use_mask = use_mask
        )

        torch.save(model.state_dict(), f"./models_library/model_{variable_name}.pth")

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(train_losses)), train_losses, label="Training Loss")
        plt.plot(np.arange(len(val_losses)), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss for {variable_name}")
        plt.legend()
        plt.savefig(f"./plots/loss_plot_{variable_name}.png")
        plt.close()

def _write():
    base_string = "C:/Users/jonas/Coding/Master/Main/h5/"
    h5_files_egam1, h5_files_egam7, h5_files_data, h5_files_egam = _find_h5_files(base_string)

    validation_split = 0.25
    train_size_egam = int((1 - validation_split) * len(h5_files_egam))
    train_size_data = int((1 - validation_split) * len(h5_files_data))

    models_path = Path("./models_library")
    csv_path = Path("./csv_library")
    csv_path.mkdir(parents=True, exist_ok=True)

    valid_fields = {"egammas": ["pt", "eta", "phi", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight"], "truth": ["pt", "eta", "phi"]}
    model_configs = {"z" : [False, True, False], "pt" : [False, True, False], "eta" : [False, False, False], "phi" : [False, False, False], "origin" : [False, False, False]}
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
    #data_loading.models_to_csv_files(
    #    models_path,
    #    csv_path,
    #    h5_files_egam[train_size_egam:],
    #    h5_files_data[train_size_data:],
    #    valid_fields = valid_fields,
    #    batch_size = 4096,
    #    num_workers = 7,
    #    pin_memory = False,
    #    max_rows = 1_000_000,
    #    model_type = "pair",
    #    model_configs = model_configs,
    #)
    #data_loading.datasets_to_csv_files(
    #    h5_files_egam[train_size_egam:],
    #    h5_files_data[train_size_data:],
    #    csv_path,
    #    valid_fields = valid_fields,
    #    batch_size = 4096,
    #    num_workers = 7,
    #    pin_memory = False,
    #    max_rows = 1_000_000,
    #    )


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from functions_library import dataset_classes, neural_models, training_classes, model_training, data_loading
    from pathlib import Path

    #_train()
    _write()
    print("Training and writing completed successfully.")
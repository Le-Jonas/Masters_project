def main():
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from functions_library import dataset_classes, neural_models, training_classes, model_training, data_loading

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    base_string = "/eos/user/j/jjeppese/h5/"
    data_strings = ["MC_EGAM1", "MC_EGAM7", "DATA_EGAM"]
    name_strings = data_strings + ["MC_EGAM"]
    full_strings = [base_string + data_string for data_string in data_strings]
    h5_files_dir = data_loading.create_path_lists(full_strings, outputs=name_strings, mixing_list = [(0, 1)], seed=0)
    h5_files_egam1 = h5_files_dir[name_strings[0]]
    h5_files_egam7 = h5_files_dir[name_strings[1]]
    h5_files_egam = h5_files_dir[name_strings[-1]]
    h5_files_data = h5_files_dir[name_strings[-2]]

    valid_fields = {"egammas": ["pt", "eta", "phi", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight"], "truth": ["pt", "eta", "phi", "e"]}
    variables = ["pt", "eta", "phi", "z", "origin"]

    for variable in variables:
        dataset_class = dataset_classes.ZPairDataset if variable in ["z", "origin"] else dataset_classes.H5EgammaDataset
        nn_model = neural_models.ParticlePairNetwork if variable in ["z", "origin"] else neural_models.ParticleOriginNetwork
        loss_class = torch.nn.BCEWithLogitsLoss() if variable == "origin" else training_classes.AngularGaussianNLLLoss() if variable in ["phi"] else training_classes.GaussianNLLLoss()

        field = ["truthOrigin", "passid_Medium"] if variable == "origin" else [["pt", "eta", "phi"], ["pt", "eta", "phi"]] if variable == "z" else [variable, variable]
        output_size = 1 if variable == "origin" else 2

        binary_target = 13 if variable == "origin" else False
        norm_target = True if variable in ["z", "pt"] else False
        z_target = True if variable == "z" else False
        log_target = False

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
            num_workers = 15,
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
            num_epochs = 20,
            sample_size = 1_000_000,
            domain_adversary = True,
            domain_loss_function = torch.nn.BCEWithLogitsLoss(),
            domain_loss_weight = 0.05,
            gradient_reversal_weight = 0.50,
            mixed_target_loss_weight = 0.20,
            domain = "bool",
            periodic_save_interval = 5,
            periodic_save_path = f"./models_library/model_{variable_name}.pth",
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

if __name__ == "__main__":
    main()
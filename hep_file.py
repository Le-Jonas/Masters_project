def main():
    from pathlib import Path
    from torch.utils.data import Sampler, DataLoader, random_split
    import torch
    import h5py
    from functions_library import dataset_classes, data_loading, neural_models, model_training, misc_functions

    notebook_dir = Path.cwd()
    group_dir = notebook_dir.parent.parent.parent.parent
    h5_dirs_path = group_dir / "morancho/ATLAS_Open_Data/Zee_Analysis/h5"


    # Choose two folders from the same particle type and simulation category.
    source_1_dir = h5_dirs_path / "user.amoranch.361106.e3601_s3681_r13167_p4940.egd.electrons-cell_trial.EGAM1.25_0_63.3afd5d9_output.h5"
    source_2_dir = h5_dirs_path / "user.amoranch.423300.e3848_s3681_r13167_p4940.egd.electrons-cell_trial.EGAM7.25_0_63.3afd5d9_output.h5"
    source_3_dir = h5_dirs_path / "user.amoranch.data15_13TeV.data15_13TeV.egd.electrons-cells-data.EGAM1.25_0_63.44b1f75_output.h5"

    h5_files_egam1 = sorted(
        path for path in source_1_dir.iterdir() if path.is_file() and path.suffix == ".h5"
    )
    h5_files_egam7 = sorted(
        path for path in source_2_dir.iterdir() if path.is_file() and path.suffix == ".h5"
    )
    h5_files_data = sorted(
        path for path in source_3_dir.iterdir() if path.is_file() and path.suffix == ".h5"
    )

    if not h5_files_egam1:
        raise FileNotFoundError(f"No files found in {source_1_dir}")
    if not h5_files_egam7:
        raise FileNotFoundError(f"No files found in {source_2_dir}")
    if not h5_files_data:
        raise FileNotFoundError(f"No files found in {source_3_dir}")

    len_egam1 = len(h5_files_egam1)
    len_egam7 = len(h5_files_egam7)
    len_data = len(h5_files_data)

    egam1_split = int(len_egam1 * 0.8)
    egam7_split = int(len_egam7 * 0.8)

    mean_dataset = dataset_classes.H5EgammaDataset(
        h5_files_egam1[:egam1_split],
        mix_files = h5_files_egam7[:egam7_split],
        exclude_features = ["truth"],
        exclude_fields = ["truthOrigin", "truthType"],
        y_source = "egammas",
        y_field = "truthOrigin",
        mixture_ratio = 0.5
    )

    means, stds = misc_functions.compute_mean_std(mean_dataset, sample_size=1000000, batch_size=4096)

    train_dataset = dataset_classes.ZPairDataset(
        h5_files_egam1[:egam1_split],
        mix_files = h5_files_egam7[:egam7_split],
        exclude_features = ["truth"],
        exclude_fields = ["truthOrigin", "truthType"],
        y_source = "egammas",
        y_field = "truthOrigin",
        mixture_ratio = 0.5
    )

    val_dataset = dataset_classes.ZPairDataset(
        h5_files_egam1[egam1_split:],
        mix_files = h5_files_egam7[egam7_split:],
        exclude_features = ["truth"],
        exclude_fields = ["truthOrigin", "truthType"],
        y_source = "egammas",
        y_field = "truthOrigin",
        mixture_ratio = 0.5
    )

    test_dataset = dataset_classes.ZPairDataset(
        h5_files_data,
        exclude_features = ["truth"],
        exclude_fields = ["truthOrigin", "truthType"],
        y_source = None,
        y_field = None,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    train_sampler = data_loading.ShuffledContiguousBatchSampler(len(train_dataset), batch_size=4096, dataset=train_dataset)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=16, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=16, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False, num_workers=16, prefetch_factor=2, persistent_workers=True, pin_memory=True)

    particle_fields = [field for field in train_dataset.fields if not field.startswith("eventwise_")]
    event_fields = [field for field in train_dataset.fields if field.startswith("eventwise_")]
    particle_size = len(particle_fields) // 2
    if len(particle_fields) % 2 != 0:
        raise ValueError("Pair particle fields are not evenly split between the two particles.")

    model_origin = neural_models.ParticlePairNetwork(
        particle_size,
        len(event_fields),
        particle_hidden_sizes=[1024, 256, 256, 256, 256, 256, 64, 8],
        event_hidden_sizes=[64, 64, 16, 4],
        consolidation_hidden_sizes=[32, 32, 32, 8],
        output_size=1,
    ).to(device)
    optimizer = torch.optim.Adam(model_origin.parameters(), lr=0.001)
    loss_function = torch.nn.BCEWithLogitsLoss()

    model_origin, train_losses, val_losses = model_training.train_model(model_origin, optimizer, loss_function, train_loader, val_loader, num_epochs=10, device=device, means=means, stds=stds, binary_target=13)
    torch.save(model_origin.state_dict(), "model_origin_pair_full.pth")

    predictions_data = model_training.predict(model_origin, test_loader, device=device, means=means, stds=stds)
    mask = predictions_data > 0
    misc_functions.find_Z_peak_pairs(
        source_3_dir,
        csv_output_path="z_masses_full.csv",
        global_mask=mask,
    )

if __name__ == "__main__":
    main()
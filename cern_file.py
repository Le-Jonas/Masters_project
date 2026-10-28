import torch
from functions_library import *

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

model, train_losses, val_losses = model_training.compressed_training_function(
    h5_files = h5_files_egam,
    mix_files = h5_files_data,
    mixture_ratio = 0.75,
    dataset_class = dataset_classes.ZPairDataset,
    nn_model = neural_models.ParticlePairNetwork,
    model_kwargs = {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 8], "event_hidden_sizes" : [64, 64, 16, 4], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 2,},
    loss_function = training_classes.GaussianNLLLoss(),
    optimizer_class = torch.optim.Adam,
    batch_size = 4096,
    num_workers = 15,
    exclude_features = ["truth"],
    exclude_fields = {"egammas" : ["truthOrigin", "truthType", "pt", "eta", "phi", "e"]},
    fallback_source = "egammas",
    fallback_field = ["pt", "eta", "phi", "e"],
    y_source = "truth",
    y_field = ["pt", "eta", "phi", "e"],
    valid_fields = {"egammas": ["pt", "eta", "phi", "e"], "truth": ["pt", "eta", "phi", "e"]},
    log_target = False,
    binary_target = False,
    z_target = True,
    norm_target = True,
    pin_memory = False,
    num_epochs = 20,
    sample_size = 1_000_000,
    domain_adversary = True,
    domain_loss_function = torch.nn.BCEWithLogitsLoss(),
    domain_loss_weight = 0.05,
    gradient_reversal_weight = 0.5,
    mixed_target_loss_weight = 0.25,
    domain = "bool",
    periodic_save_interval = 5,
    periodic_save_path = "./models_library/model_z_pair.pth",
)

torch.save(model.state_dict(), "./models_library/model_z_pair.pth")
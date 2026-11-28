import torch.nn as tnn
from functions_library import neural_models, training_classes, dataset_classes

model_kwargs_dict = {"type" : {"particle_hidden_sizes": [1024, 64, 64, 32, 32], "event_hidden_sizes" : [64, 64, 64, 16], "consolidation_hidden_sizes" : [32, 8], "output_size" : 1},
                     "isolation" : {"particle_hidden_sizes": [1024, 64, 32, 32], "event_hidden_sizes" : [64, 48, 48, 8], "consolidation_hidden_sizes" : [32, 32, 8], "output_size" : 1},
                     "pt" : {"particle_hidden_sizes": [1024, 256, 256, 128, 128], "event_hidden_sizes" : [16, 16, 4], "consolidation_hidden_sizes" : [128, 128, 128, 8], "output_size" : 1},
                     "pt_logvar" : {"particle_hidden_sizes": [1024, 256, 256, 128, 128], "event_hidden_sizes" : [16, 16, 4], "consolidation_hidden_sizes" : [128, 128, 128, 8], "output_size" : 1},
                     "eta" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1},
                     "phi" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1},
                     "origin" : {"particle_hidden_sizes": [64, 64, 16, 16, 8], "event_hidden_sizes" : [16, 8], "consolidation_hidden_sizes" : [16, 8], "output_size" : 1},
                     "origin-all" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1},
                     "z" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1}
}


fields_dict = {"type" : {"egammas" : ["pt", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight"]}, 
               "isolation" : {"egammas" : ["truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"]}, 
               "pt" : {"egammas" : ["pt", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight"]}, 
               "pt_logvar" : {"egammas" : ["pt", "e", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight"]},
               "eta" : {"egammas" : ["eta", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"]}, 
               "phi" : {"egammas" : ["phi", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"]}, 
               "origin" : {"egammas" : ["z0"]},
               "origin-all" : {"egammas" : ["truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"]},
               "z" : {"egammas" : ["z0"]},
}

optimizer_kwargs_dict = {"type" : {"lr" : 5e-5, "betas" : (0.65, 0.65), "eps" : 1e-07, "weight_decay" : 5e-4},
                         "isolation" : {"lr" : 2e-5, "betas" : (0.75, 0.88), "eps" : 1e-07, "weight_decay" : 1e-3},
                         "pt" : {"lr" : 1e-5, "betas" : (0.85, 0.95), "eps" : 1e-07, "weight_decay" : 1e-5},
                         "pt_logvar" : {"lr" : 1e-2, "betas" : (0.50, 0.95), "eps" : 1e-07, "weight_decay" : 1e-7},
                         "eta" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "phi" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "origin" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "origin-all" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "z" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
}

variable_kwargs_dict = {"type" : {"dataset": dataset_classes.H5EgammaDataset, "nn_model" : neural_models.ParticleOriginNetwork, "loss_function" : tnn.BCEWithLogitsLoss(), "binary_target" : 2, "norm_target" : False, "gev_target" : False, "y_field": [["truthType", "passid_VeryLoose"], ["passid_Medium", "passid_VeryLoose"]]},
                        "isolation" : {"dataset": dataset_classes.H5EgammaDataset, "nn_model" : neural_models.ParticleOriginNetwork, "loss_function" : tnn.BCEWithLogitsLoss(), "binary_target" : 1, "norm_target" : False, "gev_target" : False, "y_field": [["passiso_Loose_VarRad"], ["passiso_Loose_VarRad"]]},
                        "pt" : {"dataset": dataset_classes.H5EgammaDataset, "nn_model" : neural_models.ParticleOriginNetwork, "loss_function" : training_classes.LogCoshLoss(), "binary_target" : False, "norm_target" : False, "gev_target" : True, "y_field": [["pt", "truthType"], ["pt", "passid_Tight"]]},
                        "pt_logvar" : {"dataset": dataset_classes.H5EgammaDataset, "nn_model" : neural_models.ParticleOriginNetwork, "loss_function" : training_classes.RobustGaussianVarianceNLLLoss(), "binary_target" : False, "norm_target" : False, "gev_target" : True, "y_field": [["pt", "truthType"], ["pt", "passid_Tight"]]},
                        "eta" : {"dataset": dataset_classes.H5EgammaDataset, "nn_model" : neural_models.ParticleOriginNetwork, "loss_function" : tnn.MSELoss(), "binary_target" : False, "norm_target" : False, "gev_target" : False, "y_field": [["eta", "truthType"], ["eta", "passid_Tight"]]},
                        "phi" : {"dataset": dataset_classes.H5EgammaDataset, "nn_model" : neural_models.ParticleOriginNetwork, "loss_function" : training_classes.AngularMSELoss(), "binary_target" : False, "norm_target" : False, "gev_target" : False, "y_field": [["phi", "truthType"], ["phi", "passid_Tight"]]},
                        "origin" : {"dataset": dataset_classes.ZPairDataset, "nn_model" : neural_models.ParticlePairNetwork, "loss_function" : tnn.BCEWithLogitsLoss(), "binary_target" : 13, "norm_target" : True, "gev_target" : False, "y_field": [["truthOrigin", "pt", "eta", "phi"], ["passid_Loose", "pt", "eta", "phi"]]},
                        "origin-all" : {"dataset": dataset_classes.ZPairDataset, "nn_model" : neural_models.ParticlePairNetwork, "loss_function" : tnn.BCEWithLogitsLoss(), "binary_target" : 13, "norm_target" : False, "gev_target" : False, "y_field": [["truthOrigin", "pt", "eta", "phi"], ["passid_Loose", "pt", "eta", "phi"]]},
                        "z" : {"dataset": dataset_classes.ZPairDataset, "nn_model" : neural_models.ParticlePairNetwork, "loss_function" : training_classes.LogCoshLoss(), "binary_target" : False, "norm_target" : False, "gev_target" : True, "y_field": [["pt", "eta", "phi"], ["pt", "eta", "phi"]]}
}
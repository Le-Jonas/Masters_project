model_kwargs_dict = {"type" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1},
                     "isolation" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1},
                     "pt" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1},
                     "eta" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1},
                     "phi" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1},
                     "origin" : {"particle_hidden_sizes": [64, 64, 16, 16, 8], "event_hidden_sizes" : [16, 8], "consolidation_hidden_sizes" : [16, 8], "output_size" : 1},
                     "origin-all" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1},
                     "z" : {"particle_hidden_sizes": [1024, 256, 256, 256, 256, 256, 64, 32], "event_hidden_sizes" : [64, 64, 16, 8], "consolidation_hidden_sizes" : [32, 32, 32, 8], "output_size" : 1}
}


fields_dict = {"type" : {"egammas" : ["pt", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"]}, 
               "isolation" : {"egammas" : ["pt", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"]}, 
               "pt" : {"egammas" : ["pt", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"]}, 
               "eta" : {"egammas" : ["eta", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"]}, 
               "phi" : {"egammas" : ["phi", "truthOrigin", "truthType", "passid_VeryLoose", "passid_Loose", "passid_Medium", "passid_Tight", "passiso_Loose_VarRad"]}, 
               "origin" : {"egammas" : ["z0"]},
               "origin-all" : {"egammas" : ["z0", "truthOrigin", "truthType"]},
               "z" : {"egammas" : ["z0"]},
}

optimizer_kwargs_dict = {"type" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "isolation" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "pt" : {"lr" : 1e-3, "betas" : (0.85, 0.95), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "eta" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "phi" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "origin" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "origin-all" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
                         "z" : {"lr" : 1e-4, "betas" : (0.9, 0.999), "eps" : 1e-07, "weight_decay" : 1e-4},
}
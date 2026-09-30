from torch import nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ParticleOriginNetwork(nn.Module):
    def __init__(
        self,
        particle_size,
        event_size,
        particle_hidden_sizes,
        event_hidden_sizes,
        output_size,
    ):
        super().__init__()

        def make_branch(input_size, hidden_sizes):
            layers = []
            last_size = input_size
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(last_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(0.10),
                ])
                last_size = hidden_size
            return nn.Sequential(*layers)

        if not particle_hidden_sizes or not event_hidden_sizes:
            raise ValueError("Both hidden-size lists must contain at least one layer.")

        self.particle_branch = make_branch(particle_size, particle_hidden_sizes)
        self.event_branch = make_branch(event_size, event_hidden_sizes)

        self.classifier = nn.Sequential(
            nn.Linear(particle_hidden_sizes[-1] + event_hidden_sizes[-1], 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, particle_features, event_features):
        particle_embedding = self.particle_branch(particle_features)
        event_embedding = self.event_branch(event_features)
        combined = torch.cat([particle_embedding, event_embedding], dim=-1)
        return self.classifier(combined).squeeze(-1)

class ParticlePairNetwork(nn.Module):
    def __init__(
        self,
        particle_size,
        event_size,
        particle_hidden_sizes,
        event_hidden_sizes,
        output_size,
    ):
        super().__init__()

        def make_branch(input_size, hidden_sizes):
            layers = []
            last_size = input_size
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(last_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(0.10),
                ])
                last_size = hidden_size
            return nn.Sequential(*layers)

        if not particle_hidden_sizes or not event_hidden_sizes:
            raise ValueError("Both hidden-size lists must contain at least one layer.")

        self.particle1_branch = make_branch(particle_size, particle_hidden_sizes)
        self.particle2_branch = make_branch(particle_size, particle_hidden_sizes)
        self.event_branch = make_branch(event_size, event_hidden_sizes)

        self.classifier = nn.Sequential(
            nn.Linear(2 * particle_hidden_sizes[-1] + event_hidden_sizes[-1], 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, particle1_features, particle2_features, event_features):
        particle1_embedding = self.particle1_branch(particle1_features)
        particle2_embedding = self.particle2_branch(particle2_features)
        event_embedding = self.event_branch(event_features)
        combined = torch.cat([particle1_embedding, particle2_embedding, event_embedding], dim=-1)
        return self.classifier(combined).squeeze(-1)
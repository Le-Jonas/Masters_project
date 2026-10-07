from torch import nn
import torch

class NeuralNetwork(nn.Module):
    """
    A simple flat feedforward neural network with customizable hidden layers and activation functions.
    Activation functions used are SiLU for hidden layers and no activation for the output layer.

    Arguments:
    - input_size (int): The number of input features.
    - hidden_sizes (list of int): A list containing the sizes of each hidden layer.
    - output_size (int): The number of output features.
    """
    def __init__(self, input_size : int, hidden_sizes : list[int], output_size : int):
        super().__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.SiLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Arguments:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.model(x)

class ParticleOriginNetwork(nn.Module):
    """
    A neural network designed to process particle and event features separately, then combine them for classification.
    Activation functions used are SiLU for hidden layers and no activation for the output layer.
    Layer normalization and dropout are applied after each hidden layer.

    Arguments:
    - particle_size (int): The number of features for each particle.
    - event_size (int): The number of features for each event.
    - particle_hidden_sizes (list of int): A list containing the sizes of each hidden layer for the particle branch.
    - event_hidden_sizes (list of int): A list containing the sizes of each hidden layer for the event branch.
    - output_size (int): The number of output features.
    """
    def __init__(
        self,
        particle_size : int,
        event_size : int,
        particle_hidden_sizes : list[int],
        event_hidden_sizes : list[int],
        output_size : int,
    ):
        super().__init__()

        def _make_branch(input_size : int, hidden_sizes : list[int]) -> nn.Sequential:
            """
            Creates a sequential branch of layers for either particle or event features.

            Arguments:
            - input_size (int): The number of input features for the branch.
            - hidden_sizes (list of int): A list containing the sizes of each hidden layer for the branch.

            Returns:
            - nn.Sequential: The sequential branch of layers.
            """
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

        self.particle_branch = _make_branch(particle_size, particle_hidden_sizes)
        self.event_branch = _make_branch(event_size, event_hidden_sizes)

        self.classifier = nn.Sequential(
            nn.Linear(particle_hidden_sizes[-1] + event_hidden_sizes[-1], 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, particle_features : torch.Tensor, event_features : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network. Combines the outputs of the particle and event branches and passes them through a combination classifier network.

        Arguments:
        - particle_features (torch.Tensor): The input features for the particle branch.
        - event_features (torch.Tensor): The input features for the event branch.

        Returns:
        - torch.Tensor: The output of the network.
        """
        particle_embedding = self.particle_branch(particle_features)
        event_embedding = self.event_branch(event_features)
        combined = torch.cat([particle_embedding, event_embedding], dim=-1)
        return self.classifier(combined).squeeze(-1)

class ParticlePairNetwork(nn.Module):
    """
    A neural network designed to process features from particle pairs and their corresponding event features separately, then combine them for classification.
    Activation functions used are SiLU for hidden layers and no activation for the output layer.
    Layer normalization and dropout are applied after each hidden layer.

    Arguments:
    - particle_size (int): The number of features for each particle.
    - event_size (int): The number of features for each event.
    - particle_hidden_sizes (list of int): A list containing the sizes of each hidden layer for the particle branches.
    - event_hidden_sizes (list of int): A list containing the sizes of each hidden layer for the event branch.
    - consolidation_hidden_sizes (list of int): A list containing the sizes of each hidden layer for the combined classifier network.
    - output_size (int): The number of output features.
    """
    def __init__(
        self,
        particle_size : int,
        event_size : int,
        particle_hidden_sizes : list[int],
        event_hidden_sizes : list[int],
        consolidation_hidden_sizes : list[int],
        output_size : int,
    ):
        super().__init__()

        def _make_branch(input_size : int, hidden_sizes : list[int]) -> nn.Sequential:
            """
            Creates a sequential branch of layers for either particle or event features.

            Arguments:
            - input_size (int): The number of input features for the branch.
            - hidden_sizes (list of int): A list containing the sizes of each hidden layer for the branch.

            Returns:
            - nn.Sequential: The sequential branch of layers.
            """
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

        self.particle1_branch = _make_branch(particle_size, particle_hidden_sizes)
        self.particle2_branch = _make_branch(particle_size, particle_hidden_sizes)
        self.event_branch = _make_branch(event_size, event_hidden_sizes)
    
        self.classifier = nn.Sequential(
            nn.Linear(2 * particle_hidden_sizes[-1] + event_hidden_sizes[-1], consolidation_hidden_sizes[0]),
            nn.LayerNorm(consolidation_hidden_sizes[0]),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(consolidation_hidden_sizes[0], consolidation_hidden_sizes[1]),
            nn.LayerNorm(consolidation_hidden_sizes[1]),
            nn.SiLU(),
            nn.Linear(consolidation_hidden_sizes[1], output_size),
        )

    def forward(self, particle1_features : torch.Tensor, particle2_features : torch.Tensor, event_features : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network. Combines the outputs of the particle and event branches and passes them through a combination classifier network.

        Arguments:
        - particle1_features (torch.Tensor): The features for the first particle.
        - particle2_features (torch.Tensor): The features for the second particle.
        - event_features (torch.Tensor): The features for the event.

        Returns:
        - torch.Tensor: The output of the classifier.
        """
        particle1_embedding = self.particle1_branch(particle1_features)
        particle2_embedding = self.particle2_branch(particle2_features)
        event_embedding = self.event_branch(event_features)
        combined = torch.cat([particle1_embedding, particle2_embedding, event_embedding], dim=-1)
        return self.classifier(combined).squeeze(-1)
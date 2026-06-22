


import torch

from CompleteState import CompleteState
from QNetwork import QNetwork
import torch.nn as nn
import torch.optim as optim

class QCheaterNetwork(QNetwork):
    """
    A specialized Q-Network for the Cheater variant of the game.
    It extends the base QNetwork to accommodate the specific input dimensions and action space of the Cheater game.
    """
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__(1, 1)
        # Additional initialization for the Cheater variant can be added here if needed
        self.step_count = 1  # Counter to track the number of training steps
        self.OUTPUT_DIM = num_actions  # Number of possible actions
        INTERNAL_DIM = 256  # Dimension of the internal layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, INTERNAL_DIM),
            nn.ReLU(),
            nn.Linear(INTERNAL_DIM, INTERNAL_DIM),
            nn.ReLU(),
            nn.Linear(INTERNAL_DIM, self.OUTPUT_DIM)
        )
        self.errors = []  # List to store estimation errors for analysis
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)  # Optimizer for training the network


    ####################### WATCH OUT: ILLEGAL OVERRIDE (?) #######################
    def forward(self, state: CompleteState | torch.Tensor) -> torch.Tensor: # type: ignore
        """
        Esegue il passaggio in avanti (forward pass) della rete neurale.
        Accetta sia un oggetto di classe CompleteState (singolo) sia un torch.Tensor (già convertito o in batch).
        
        Args:
            state: Oggetto CompleteState oppure torch.Tensor di shape (batch_size, input_dim)
        Returns:
            q_values: Tensore di shape (batch_size, num_actions)
        """
        # 1. Se riceve un oggetto CompleteState, lo trasforma in tensore PyTorch
        if isinstance(state, CompleteState):
            state_bytes = state.to_flat_network_input()  # Lista di float
            # Creiamo il tensore e aggiungiamo la dimensione del batch -> shape (1, input_dim)
            state_tensor = torch.tensor(state_bytes, dtype=torch.float32).unsqueeze(0)
            # Lo spostiamo sullo stesso device della rete (e.g., cuda o cpu)
            state_tensor = state_tensor.to(next(self.parameters()).device)
        
        # 2. Se è già un Tensore, lo usa direttamente
        elif isinstance(state, torch.Tensor):
            state_tensor = state
            
        else:
            raise TypeError(f"Tipo di input non supportato dal forward: {type(state)}")

        # 3. Passa il tensore finale attraverso la rete neurale
        return self.net(state_tensor)
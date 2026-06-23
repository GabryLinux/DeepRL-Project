from bitarray import bitarray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from State import State
from ReplayBuffer import ReplayBuffer


class QNetwork(nn.Module):
    """
        It is a Deep Q-Network implementation using PyTorch.
        It take as input the state of the environment and outputs the Q-values for each possible action.
    """
    def __init__(self, num_cards: int, num_actions: int):
        """
        Initializes the QNetwork with the given number of cards and actions.
        It builds a feedforward neural network with two hidden layers and ReLU activations.
        The input dimension is calculated as (num_cards * 2) + 1, where:
            - num_cards * 2: Represents the hand and played history bitarrays.
            - +1: Represents the number of cards played in the current round.
        The output dimension is equal to the number of possible actions.
        Args:
            num_cards (int): The total number of cards in the deck.
            num_actions (int): The total number of possible actions (2^hand_size).
        """
        super().__init__()
        self.step_count = 1  # Counter to track the number of training steps
        input_dim = (num_cards * 2) + 1
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

    def forward(self, state: State | torch.Tensor) -> torch.Tensor:
        """
        It runs the forward pass of the neural network.
        It accepts either a State object (single) or a torch.Tensor (already converted or in batch).
        
        Args:
            state: State object or torch.Tensor of shape (batch_size, input_dim)
        Returns:
            q_values: Tensor of shape (batch_size, num_actions)
        """
        if isinstance(state, State):
            state_bytes = state.to_flat_network_input()
            state_tensor = torch.tensor(state_bytes, dtype=torch.float32).unsqueeze(0)
            state_tensor = state_tensor.to(next(self.parameters()).device)
        
        elif isinstance(state, torch.Tensor):
            state_tensor = state
            
        else:
            raise TypeError(f"Tipo di input non supportato dal forward: {type(state)}")

        return self.net(state_tensor)
    

class TrainingUtils:
    @staticmethod
    def train_dqn_step(
        policy_net: QNetwork, 
        target_net: QNetwork, 
        replay_buffer, 
        batch_size: int, 
        gamma: float,
        SYNC_EVERY: int = 50,
        criterion: nn.Module = nn.MSELoss(),
        device: torch.device = torch.device("cuda")
    ):
        """
        Perform a single training step for the DQN using experiences from the replay buffer.
        It implements the Double DQN algorithm to reduce overestimation bias.
        """
        policy_net.to(device)
        target_net.to(device)
            
        if not hasattr(policy_net, 'optimizer') or policy_net.optimizer is None:
            policy_net.optimizer = optim.Adam(policy_net.parameters(), lr=1e-1)
        
        # Target network step count incremented to track when to sync with the policy network
        target_net.step_count += 1
        
        # Sync the target network with the policy network every SYNC_EVERY steps
        if target_net.step_count % SYNC_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 1. Sample a batch of experiences from the replay buffer
        states_np, actions_np, rewards_np, next_states_np = replay_buffer.sample(batch_size)

        # 2. Convert the sampled experiences into PyTorch tensors and move them to the specified device (CPU or GPU)
        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=device)

        # 3. Ensure that the actions tensor has the correct shape for gathering Q-values
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        # 4. Compute the Q-values for the current states using the policy network and gather the Q-values corresponding to the taken actions
        policy_q_values = policy_net(states).gather(1, actions).squeeze(1)


        with torch.no_grad():
            # ACTION: Use the policy network to select the best action for the next states
            best_actions = policy_net(next_states).argmax(dim=1, keepdim=True) # shape (batch_size, 1)
            
            # Q-VALUES: Use the target network to compute the Q-values for the next states and gather the Q-values corresponding to the best actions selected by the policy network
            next_q_values = target_net(next_states).gather(1, best_actions).squeeze(1) # shape (batch_size,)
            
            # C. TARGET: Compute the target Q-values using the Bellman equation, which combines the immediate rewards and the discounted future rewards
            target_q_values = rewards + (gamma * next_q_values)


        # 5. Loss Calculation: Compute the loss between the predicted Q-values and the target Q-values using the specified criterion (e.g., Mean Squared Error)
        loss = criterion(policy_q_values, target_q_values)

        # 6. Backpropagation and Optimization: Perform backpropagation to compute gradients and update the policy network's weights using the optimizer
        policy_net.optimizer.zero_grad()
        loss.backward()
        policy_net.optimizer.step()
        
        # 7. Taking the first element of the batch for learning analysis
        estimate_0 = policy_q_values[0].detach().cpu().item()
        target_0 = target_q_values[0].detach().cpu().item()

        return estimate_0, target_0
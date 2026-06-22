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
        Esegue il passaggio in avanti (forward pass) della rete neurale.
        Accetta sia un oggetto di classe State (singolo) sia un torch.Tensor (già convertito o in batch).
        
        Args:
            state: Oggetto State oppure torch.Tensor di shape (batch_size, input_dim)
        Returns:
            q_values: Tensore di shape (batch_size, num_actions)
        """
        # 1. Se riceve un oggetto State, lo trasforma in tensore PyTorch
        if isinstance(state, State):
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
        Esegue un singolo passo di ottimizzazione usando l'algoritmo Double DQN (DDQN).
        Garantisce il calcolo interamente su GPU CUDA.
        """
        # Spostiamo di sicurezza le reti sul device hardware (GPU/CUDA)
        print("UPDATE!")
        policy_net.to(device)
        target_net.to(device)
        
        # Inizializzazione di default della loss function se non passata
        if criterion is None:
            criterion = nn.MSELoss()
            
        # Controllo di sicurezza per l'ottimizzatore integrato
        if not hasattr(policy_net, 'optimizer') or policy_net.optimizer is None:
            # Fallback se la rete non ha un attributo .optimizer configurato
            policy_net.optimizer = optim.Adam(policy_net.parameters(), lr=1e-1)
        
        # Incrementiamo il contatore dei passi sulla target network
        target_net.step_count += 1
        
        # Sincronizzazione periodica dei pesi tra Policy e Target Network
        if target_net.step_count % SYNC_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 1. Campionamento dal Replay Buffer
        states_np, actions_np, rewards_np, next_states_np = replay_buffer.sample(batch_size)

        # 2. Trasferimento immediato dei dati NumPy su Tensori PyTorch allocati su CUDA
        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=device)

        # Allineamento shape delle azioni per l'operazione .gather()
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        # 3. Valutazione Q(s, a) corrente tramite la Policy Network
        policy_q_values = policy_net(states).gather(1, actions).squeeze(1)

        # =========================================================================
        # CORREZIONE DOUBLE DQN (DDQN): Scollegamento Selezione/Valutazione
        # =========================================================================
        with torch.no_grad():
            # A. SELEZIONE: Usiamo la Policy Network per scegliere l'azione migliore nello stato s'
            # argmax(dim=1) restituisce l'indice dell'azione con il valore Q più alto.
            best_actions = policy_net(next_states).argmax(dim=1, keepdim=True) # shape (batch_size, 1)
            
            # B. VALUTAZIONE: Usiamo la Target Network per stimare il valore di quell'azione specifica
            # .gather(1, best_actions) estrae il valore Q stimato dalla target_net per l'azione selezionata al punto A.
            next_q_values = target_net(next_states).gather(1, best_actions).squeeze(1) # shape (batch_size,)
            
            # C. TARGET: Applichiamo l'equazione di Bellman modificata per DDQN
            target_q_values = rewards + (gamma * next_q_values)
        # =========================================================================

        # 5. Calcolo della Loss (eseguita su CUDA)
        loss = criterion(policy_q_values, target_q_values)

        # 6. Backpropagation e aggiornamento dei pesi tramite l'ottimizzatore interno della Policy
        policy_net.optimizer.zero_grad()
        loss.backward()
        policy_net.optimizer.step()
        
        # 7. Conversione sicura dei dati in scalari Python per monitorare l'andamento
        estimate_0 = policy_q_values[0].detach().cpu().item()
        target_0 = target_q_values[0].detach().cpu().item()

        return estimate_0, target_0
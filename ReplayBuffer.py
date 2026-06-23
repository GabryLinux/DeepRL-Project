from collections import deque
import random

from bitarray import bitarray
import numpy as np

from State import State

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    @staticmethod
    def pad_msb_fixed(action: bitarray) -> bitarray:
        """
        Pad the action bitarray with leading zeros to ensure its length is a multiple of 8.
        This is necessary for consistent storage and processing of actions in the replay buffer.
        """
        remaining_bits = (8 - (len(action) % 8)) % 8 
        padding = bitarray('0' * remaining_bits) 
        return padding + action
    
    def push(self, state: State, action: bitarray, reward, next_state: State):
        """Saves a transition in the buffer."""
        int_action = int.from_bytes(self.pad_msb_fixed(action).tobytes(), byteorder='big')
        self.buffer.append((state, int_action, reward, next_state))
    
    def sample(self, batch_size: int):
        """Extracts a batch and converts the State objects to flat arrays of floats for the DQN."""
        states, actions, rewards, next_states = zip(*random.sample(self.buffer, batch_size))
        
        # The neural network needs the individual bits flattened as floats
        states_network = [s.to_flat_network_input() for s in states]
        next_states_network = [ns.to_flat_network_input() for ns in next_states]
        
        # Convert the lists to numpy arrays for efficient processing
        
        return (np.array(states_network, dtype=np.float32), 
                np.array(actions, dtype=np.int64), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states_network, dtype=np.float32))
                
    def __len__(self):
        return len(self.buffer)
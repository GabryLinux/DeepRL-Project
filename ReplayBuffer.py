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
        Aggiunge esattamente 2 bit a 0 all'inizio (MSB) 
        per portare un bitarray di 14 bit a 16 bit (2 byte).
        """
        # Creiamo un bitarray di 2 zeri e lo concateniamo a sinistra
        remaining_bits = (8 - (len(action) % 8)) % 8  # Calcola quanti bit di padding sono necessari
        padding = bitarray('0' * remaining_bits)  # Crea un bitarray di zeri per il padding
        return padding + action  # Concatena il padding a sinistra dell'azione originale
    
    def push(self, state: State, action: bitarray, reward, next_state: State):
        """Salva una transizione nel buffer."""
        # Calcola quanti bit di padding sono stati aggiunti
        int_action = int.from_bytes(self.pad_msb_fixed(action).tobytes(), byteorder='big')  # Converti l'azione in un intero dopo il padding
        self.buffer.append((state, int_action, reward, next_state))
    
    def sample(self, batch_size: int):
        """Estrae un batch e converte gli oggetti State in vettori piatti di float per il DQN."""
        states, actions, rewards, next_states = zip(*random.sample(self.buffer, batch_size))
        
        # La rete neurale ha bisogno dei singoli bit srotolati come float
        states_network = [s.to_flat_network_input() for s in states]
        next_states_network = [ns.to_flat_network_input() for ns in next_states]
        
        # Se anche l'azione è un oggetto/byte, convertila in un indice o valore numerico qui
        # Se l'azione nel buffer è già un array di byte, la convertiamo in float/long a seconda del tipo di output della tua rete
        
        return (np.array(states_network, dtype=np.float32), 
                np.array(actions, dtype=np.int64), # O np.float32 se usi output continui
                np.array(rewards, dtype=np.float32), 
                np.array(next_states_network, dtype=np.float32))
                
    def __len__(self):
        return len(self.buffer)
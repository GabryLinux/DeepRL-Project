import numpy as np
from bitarray import bitarray

class CardMapper:
    """
    Class used for the implementation of the compressed-space action agent.
    It maps the absolute action space (TOTAL_CARDS) to a relative action space (HAND_SIZE),
    where HAND_SIZE is the number of cards in the initial hand.
    """
    def __init__(self, initial_hand_bitmap: bitarray, total_cards_dim: int):
        """
        Initializes the CardMapper with the initial hand bitmap and the total number of cards.

        :param initial_hand_bitmap: A bitarray of size TOTAL_CARDS where 1 indicates the presence of a card in the initial hand.
        :param total_cards_dim: The total number of cards (TOTAL_CARDS).
        """
        self.initial_hand_bitmap = initial_hand_bitmap
        self.total_cards_dim = total_cards_dim
        
        # Memorizziamo lo stato del mapping: gli indici assoluti delle carte della mano iniziale (dove c'è 1)
        self.indices_in_hand = [i for i, bit in enumerate(initial_hand_bitmap) if bit]
        
        # La dimensione fissa dello spazio delle azioni relativo (HAND_SIZE)
        self.hand_size = len(self.indices_in_hand)

    def compress_to_hand_size(self, bitmap_total: bitarray) -> bitarray:
        """
        Given a bitarray of size TOTAL_CARDS, returns a compressed bitarray of size HAND_SIZE
        """
        # Crea un nuovo bitarray compresso prendendo solo i bit indicizzati
        compressed = bitarray()
        for idx in self.indices_in_hand:
            compressed.append(bitmap_total[idx])
        return compressed

    def expand_to_total_size(self, bitmap_hand: bitarray) -> bitarray:
        """
        Given a bitarray of size HAND_SIZE, returns an expanded bitarray of size TOTAL_CARDS 
        """
        # Crea un bitarray vuoto (tutti zeri) di dimensione assoluta
        bitmap_total = bitarray(self.total_cards_dim)
        bitmap_total.setall(0)
        
        # Mappa i bit corti sulle posizioni originali salvate nello stato del mapper
        for short_idx, original_idx in enumerate(self.indices_in_hand):
            if short_idx < len(bitmap_hand):
                bitmap_total[original_idx] = bitmap_hand[short_idx]
            
        return bitmap_total
import numpy as np
from bitarray import bitarray

class CardMapper:
    """
    Classe istanziabile che memorizza lo stato della mano iniziale 
    per mappare in modo bidirezionale i bitarray tra TOTAL_CARDS e HAND_SIZE.
    """
    def __init__(self, initial_hand_bitmap: bitarray, total_cards_dim: int):
        """
        Inizializza il mapper con lo stato della mano iniziale espresso come bitarray.
        
        Args:
            initial_hand_bitmap: bitarray della mano iniziale (dimensione TOTAL_CARDS)
            total_cards_dim: Numero totale di carte nel mazzo (es. 40, 52, 54)
        """
        self.initial_hand_bitmap = initial_hand_bitmap
        self.total_cards_dim = total_cards_dim
        
        # Memorizziamo lo stato del mapping: gli indici assoluti delle carte della mano iniziale (dove c'è 1)
        self.indices_in_hand = [i for i, bit in enumerate(initial_hand_bitmap) if bit]
        
        # La dimensione fissa dello spazio delle azioni relativo (HAND_SIZE)
        self.hand_size = len(self.indices_in_hand)

    def compress_to_hand_size(self, bitmap_total: bitarray) -> bitarray:
        """
        Prende un bitarray di dimensione TOTAL_CARDS e lo restringe a dimensione HAND_SIZE,
        estraendo solo i bit corrispondenti alle carte della mano iniziale.
        """
        # Crea un nuovo bitarray compresso prendendo solo i bit indicizzati
        compressed = bitarray()
        for idx in self.indices_in_hand:
            compressed.append(bitmap_total[idx])
        return compressed

    def expand_to_total_size(self, bitmap_hand: bitarray) -> bitarray:
        """
        Prende un bitarray relativo di dimensione HAND_SIZE (es. l'azione del DQN decodificata)
        e lo espande alla dimensione originale TOTAL_CARDS, riposizionando i bit sulle carte della mano iniziale.
        """
        # Crea un bitarray vuoto (tutti zeri) di dimensione assoluta
        bitmap_total = bitarray(self.total_cards_dim)
        bitmap_total.setall(0)
        
        # Mappa i bit corti sulle posizioni originali salvate nello stato del mapper
        for short_idx, original_idx in enumerate(self.indices_in_hand):
            if short_idx < len(bitmap_hand):
                bitmap_total[original_idx] = bitmap_hand[short_idx]
            
        return bitmap_total
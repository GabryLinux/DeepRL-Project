import numpy as np
from bitarray import bitarray

class State:
    def __init__(self, hand: bitarray, played_history: bitarray, cards_played_before_in_round: int):
        # Manteniamo i bitarray internamente per la massima efficienza di memoria
        self.hand = hand
        self.played_history = played_history
        self.cards_played_before_in_round = cards_played_before_in_round

    def to_gym_observation(self) -> dict:
        """
        Restituisce lo stato formattato in byte per Gymnasium (risparmio RAM).
        """
        return {
            "hand": np.frombuffer(self.hand.tobytes(), dtype=np.uint8),
            "played_history": np.frombuffer(self.played_history.tobytes(), dtype=np.uint8),
            "cards_placed": int(self.cards_played_before_in_round)
        }

    def to_flat_network_input(self) -> list[float]:
        """
        Trasforma lo stato in un vettore piatto di float (0.0 o 1.0) per PyTorch.
        Ogni singola carta del mazzo avrà il suo neurone dedicato a 0.0 o 1.0.
        """
        # Ottimizzazione: map(float, ...) è scritto in C sotto il cofano di Python 
        # ed è leggermente più veloce della list comprehension quando lavori con migliaia di elementi.
        hand_bits = list(map(float, self.hand.tolist()))
        history_bits = list(map(float, self.played_history.tolist()))
        
        # Uniamo i vettori e aggiungiamo la singola variabile numerica alla fine
        return hand_bits + history_bits + [float(self.cards_played_before_in_round)]



from bitarray import bitarray
import numpy as np

from State import State


class CompleteState(State):
    def __init__(self, hand: bitarray, played_history: bitarray, played_cards_in_round: bitarray):
        super().__init__(hand, played_history, 0)  # Initialize with 0 cards played before in the round
        self.played_cards_in_round = played_cards_in_round  # Bitarray to track the cards played in the current round

    def to_gym_observation(self) -> dict:
        """
        Restituisce lo stato formattato in byte per Gymnasium (risparmio RAM).
        """
        return {
            "hand": np.frombuffer(self.hand.tobytes(), dtype=np.uint8),
            "played_history": np.frombuffer(self.played_history.tobytes(), dtype=np.uint8),
            "played_cards_in_round": np.frombuffer(self.played_cards_in_round.tobytes(), dtype=np.uint8),
        }

    def to_flat_network_input(self) -> list[float]:
        """
        Trasforma lo stato in un vettore piatto di float (0.0 o 1.0) per PyTorch.
        Ogni singola carta del mazzo avrà il suo neurone dedicato a 0.0 o 1.0.
        """
        hand_bits = list(map(float, self.hand.tolist()))
        history_bits = list(map(float, self.played_history.tolist()))
        round_bits = list(map(float, self.played_cards_in_round.tolist()))

        # Uniamo i vettori e aggiungiamo la singola variabile numerica alla fine
        return hand_bits + history_bits + round_bits
import numpy as np
from bitarray import bitarray

class State:
    """
        Implementation of the State class, which represents the current state of the game.
    """
    def __init__(self, hand: bitarray, played_history: bitarray, cards_played_before_in_round: int):
        self.hand = hand
        self.played_history = played_history
        self.cards_played_before_in_round = cards_played_before_in_round

    def to_gym_observation(self) -> dict:
        """
        Converts the state to a format suitable for Gym environments.
        """
        return {
            "hand": np.frombuffer(self.hand.tobytes(), dtype=np.uint8),
            "played_history": np.frombuffer(self.played_history.tobytes(), dtype=np.uint8),
            "cards_placed": int(self.cards_played_before_in_round)
        }

    def to_flat_network_input(self) -> list[float]:
        """
        Transforms the state into a flat list of floats (0.0 or 1.0) for PyTorch.
        Each card in the deck will have its dedicated neuron set to 0.0 or 1.0, and the number of cards played before in the round will be added as a single float at the end.
        """
        hand_bits = list(map(float, self.hand.tolist()))
        history_bits = list(map(float, self.played_history.tolist()))
        
        return hand_bits + history_bits + [float(self.cards_played_before_in_round)]
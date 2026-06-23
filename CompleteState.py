


from bitarray import bitarray
import numpy as np

from State import State


class CompleteState(State):
    """
        Class used for the implementation of the agents in the 4th experiment.
        It extends the State class by adding a bitarray to track the cards played in the current round.
    """


    def __init__(self, hand: bitarray, played_history: bitarray, played_cards_in_round: bitarray):
        """
        Initializes the CompleteState with the hand, played history, and played cards in the current round.
        Args:
            hand (bitarray): A bitarray representing the cards in the player's hand.
            played_history (bitarray): A bitarray representing all the cards already played in previous rounds.
            played_cards_in_round (bitarray): A bitarray representing the cards played in the current round.
        """
        super().__init__(hand, played_history, 0)  # Initialize with 0 cards played before in the round
        self.played_cards_in_round = played_cards_in_round  # Bitarray to track the cards played in the current round

    def to_gym_observation(self) -> dict:
        """
        Converts the state to a format suitable for Gym environments.
        """
        return {
            "hand": np.frombuffer(self.hand.tobytes(), dtype=np.uint8),
            "played_history": np.frombuffer(self.played_history.tobytes(), dtype=np.uint8),
            "played_cards_in_round": np.frombuffer(self.played_cards_in_round.tobytes(), dtype=np.uint8),
        }

    def to_flat_network_input(self) -> list[float]:
        """
        Transforms the state into a flat list of floats (0.0 or 1.0) for PyTorch.
        Each card in the deck will have its dedicated neuron set to 0.0 or 1.0.
        """
        hand_bits = list(map(float, self.hand.tolist()))
        history_bits = list(map(float, self.played_history.tolist()))
        round_bits = list(map(float, self.played_cards_in_round.tolist()))

        # Merge the vectors and return the combined list
        return hand_bits + history_bits + round_bits
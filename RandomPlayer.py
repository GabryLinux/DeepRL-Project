

import random

from bitarray import bitarray

from Deck import DeckHelper
from Player import Player


class RandomPlayer(Player):
    """
        Implementation of the RandomPlayer class, which extends the Player class.
        It always plays a random card from its hand, without any strategy or learning.
    """
    def __init__(self, player_id: int, total_cards: int):
        super().__init__(player_id)
        self.total_cards = total_cards
        self._hand = bitarray(total_cards)  # Bitarray to represent the player's hand
        self._hand.setall(0)  # Initialize all cards in hand as not present

    def play_cards(self):
        """Play a random card from the player's hand and update the player's state."""
        random_played_cards = self._random_play(self._hand)
        self._hand &= ~random_played_cards  # Remove the played cards from the player's hand
        return random_played_cards
    
    def reset(self):
        """
        Reset the player's state at the beginning of each round.
        """
        self._hand.setall(0)  # Clear the player's hand
        self._score = 0  # Reset the player's score to 0

    def update_state(self, action: bitarray, played_cards_in_round: bitarray, reward: int):
        """
        Update the player's state based on the cards played by other players in the current round.
        This method can be used to implement strategies based on the observed plays.
        """
        # For a RandomPlayer, we might not need to update any state based on played cards,
        # but we can still return the current hand and played history for consistency.
        return
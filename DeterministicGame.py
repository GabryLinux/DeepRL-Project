

from collections import OrderedDict
import math
from typing import Any

from bitarray import bitarray
import numpy as np

from Deck import Deck, DeckHelper
from Game import GameEnv
from Player import Player

from gymnasium import spaces

class DeterministicGame(GameEnv):
    """
        Class used in the implementation of the environment in the 4th experiment.
        It extends the Game class by disabling the changing of the playing order and
        by changing the state representation to include the cards played in the current round.
    """
    def __init__(self, deck: Deck):
        super().__init__(deck)
        self.deck = deck
        self.players: list[Player] = []
        self.old_hands: OrderedDict[Player, bitarray] = OrderedDict()  # Store the old hands of players before they play
        self.played_cards_in_round: OrderedDict[Player, bitarray] = OrderedDict()



    def set_next_first_player(self, scores: OrderedDict[Player, int]):
        """
        It doesn't change the position of players.
        """
        return
    
    def start_game(self):
        """Start the game by dealing cards to players."""
        super().start_game()
        num_cards = self.deck.num_cards
        self.observation_space = spaces.Dict(
            {
                # Hand of the player: 1 float per every card in the deck, 0.0 or 1.0
                "hand": spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(num_cards,), 
                    dtype=np.uint8
                ),
                
                # Cards history: 1 float per every card in the deck, 0.0 or 1.0
                "played_history": spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(num_cards,), 
                    dtype=np.uint8
                ),
                # Cards played in the current round: 1 float per every card in the deck, 0.0 or 1.0
                "played_cards_in_round": spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(num_cards,), 
                    dtype=np.uint8
                ),
            }
        )
        self.action_space = spaces.Discrete(2 ** self.deck.num_cards) # every possible combination of cards to play
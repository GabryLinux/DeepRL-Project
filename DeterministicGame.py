

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
    def __init__(self, deck: Deck):
        super().__init__(deck)
        self.deck = deck
        self.players: list[Player] = []
        self.old_hands: OrderedDict[Player, bitarray] = OrderedDict()  # Store the old hands of players before they play
        self.played_cards_in_round: OrderedDict[Player, bitarray] = OrderedDict()


    """
    A deterministic game where the outcome is determined by the actions taken by the players.
    """
    def set_next_first_player(self, scores: OrderedDict[Player, int]):
        """
        Set the next first player based on the scores of the current round.
        The player with the highest score becomes the first player in the next round.
        In case of a tie, the player with the lowest ID among those tied becomes the first player.
        """
        return
    
    def start_game(self):
        """Start the game by dealing cards to players."""
        super().start_game()
        num_cards = self.deck.num_cards
        self.observation_space = spaces.Dict(
            {
                # La mano del giocatore: un vettore di 7 byte (valori da 0 a 255)
                "hand": spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(num_cards,), 
                    dtype=np.uint8
                ),
                
                # Lo storico delle carte giocate nella partita: altri 7 byte
                "played_history": spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(num_cards,), 
                    dtype=np.uint8
                ),
                # Le carte giocate nel round corrente: altri 7 byte
                "played_cards_in_round": spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(num_cards,), 
                    dtype=np.uint8
                ),
            }
        )
        self.action_space = spaces.Discrete(2 ** self.deck.num_cards) # every possible combination of cards to play
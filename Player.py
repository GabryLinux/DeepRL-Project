# Tangentopolis implementation

from abc import ABC, abstractmethod

from bitarray import bitarray
import numpy as np

from Deck import DeckHelper


class Player(ABC): 
    def __init__(self, player_id: int):
        """Initialize a player with a unique ID, an empty hand, and a score of 0."""
        self._player_id = player_id
        self._hand = bitarray()  # Player's hand represented as a bitarray
        self._score = 0  # Player's score

    def receive_cards(self, cards: bitarray):
        """
        Player receives a list of cards and adds them to their hand.
        The hand is represented as a bitarray where each bit corresponds to a card in the deck.
        """
        print(f"Player {self._player_id} received cards: {cards}")  # Debug: Print the received cards
        self._hand = cards
        
    @abstractmethod
    def play_cards(self) -> bitarray:
        """
        Abstract method to play cards from the player's hand.
        This method should be implemented by subclasses to define how a player plays cards.
        """
        pass

    @abstractmethod
    def update_state(self, action: bitarray, played_cards_in_round: bitarray, reward: int):
        """
        Update the player's state based on the cards played by other players in the current round.
        This method can be used to implement strategies based on the observed plays.
        """
        # This method can be implemented to allow players to observe the played cards
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the player's state for a new game.
        This method should clear the player's hand and reset their score.
        """
        pass

    def update_score(self, score):
        """
        Update the player's score based on the cards in their hand.
        The scoring rules can be defined based on the game requirements.
        """
        self._score += score
        return
    
    def get_score(self):
        """
        Get the player's current score.
        """
        return self._score
    

    def get_id(self):
        """
        Get the player's ID.
        """
        return self._player_id
    
    
    def has_cards(self):
        """Check if the player has any cards left in their hand.
        """
        return sum(self._hand) > 0  # Check if there are any cards left in the player's hand
    
    def get_hand(self):
        """Get the player's current hand as a bitarray."""
        return self._hand
    
    
    @staticmethod
    def _random_play(available_cards_bitarray: bitarray) -> bitarray:
        """
        Randomly selects a combination of cards to play from the available cards in hand.
        The selection is made by randomly choosing a number of cards to play (at least one)
        and then randomly selecting that many cards from the available ones.
        
        Args:
            available_cards_bitarray: A bitarray representing the cards available in the player's hand (1 for available, 0 for not available).
            
        Returns:
            A bitarray representing the cards chosen to play (1 for played, 0 for not played).
        """
        num_cards = len(available_cards_bitarray)
        
        available_indices = [i for i, bit in enumerate(available_cards_bitarray) if bit]
        num_available = len(available_indices)
        
        played_cards = bitarray(num_cards)
        played_cards.setall(0)
        
        if num_available == 0:
            return played_cards
            
        num_to_play = np.random.randint(1, num_available + 1)
        
        chosen_indices = np.random.choice(available_indices, size=num_to_play, replace=False)
        
        for idx in chosen_indices:
            played_cards[idx] = True
        
        if not played_cards.any():
            raise ValueError("Random play generated an empty action, which should not happen.")
        return played_cards
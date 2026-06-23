from bitarray import bitarray
import numpy as np

from DQNPlayer import DQNPlayer
from Player import Player
from QNetwork import QNetwork
from State import State
from card_mapper import CardMapper



class DQNPlayer_HandBitmap(DQNPlayer):
    """
       Implementation of the compressed-space action agent.
    """
    def __init__(self, player_id: int, total_cards: int, cards_per_player: int, 
                 discount_factor: float = 0.99, epsilon_decay: float = 0.995, 
                 replay_buffer_capacity: int = 10000, batch_size: int = 32,
                 SYNC_TARGET_EVERY: int = 10):
        """
        Initializes the DQNPlayer_HandBitmap with the given parameters.
        It also build the target and policy networks with the appropriate 
        input and output dimensions based on the total number of cards and the number of cards per player.
        Args:
            player_id (int): The ID of the player.
            total_cards (int): The total number of cards in the deck.
            cards_per_player (int): The number of cards each player starts with.
            discount_factor (float): The discount factor for future rewards.
            epsilon_decay (float): The decay rate for the exploration rate.
            replay_buffer_capacity (int): The capacity of the replay buffer.
            batch_size (int): The batch size for training the Q-network.
            SYNC_TARGET_EVERY (int): Frequency to sync the target network with the policy network.
        """
        super().__init__(player_id, total_cards, cards_per_player, discount_factor, 
                         epsilon_decay, replay_buffer_capacity, batch_size,
                         SYNC_TARGET_EVERY, disable_network_building=True)  # Disable network building in the parent class
        self.total_cards = total_cards                      # Number of cards in the deck
        self.cards_per_player = cards_per_player  # Number of cards each player starts with

        self.cards_played = bitarray(total_cards)  # Bitarray to track played cards
        self.cards_played.setall(0)  # Initialize all cards as not played

        self._hand = bitarray(total_cards)  # Bitarray to represent the player's hand
        self._hand.setall(0)  # Initialize all cards in hand as not present

        self._action = bitarray(total_cards)  # Bitarray to represent the action (cards to play)
        self._action.setall(0)  # Initialize all actions as not taken

        self.num_cards_played = 0

        hand_bitmap_size = self.cards_per_player  # Each card is represented by a bit in the bitmap
        self.policy_network = QNetwork(num_cards=total_cards, num_actions=2**hand_bitmap_size)  # Initialize the Q-network with bitmap size
        self.target_network = QNetwork(num_cards=total_cards, num_actions=2**hand_bitmap_size)  # Initialize the target network with bitmap size
        self.card_mapper = CardMapper(initial_hand_bitmap=self._hand, total_cards_dim=self.total_cards)  # Initialize the card mapper with the player's hand and total cards

    def receive_cards(self, cards: bitarray):
        """
        Player receives a list of cards and adds them to their hand.
        The hand is represented as a bitarray where each bit corresponds to a card in the deck.
        """
        self.card_mapper = CardMapper(initial_hand_bitmap=cards, total_cards_dim=self.total_cards) # Mapper initialization
        super().receive_cards(cards)  # Call the parent class method to set the hand

    def q_play_cards(self) -> bitarray:
        _action = super().q_play_cards()  # Get the action from the parent class
        expanded_action = self.card_mapper.expand_to_total_size(_action)  # Expand the action to the total card size
        return expanded_action
    
    def _experience_push(self, new_hand, new_cards_played, action, new_num_cards_played, reward):
        """
        Push the experience into the replay buffer.
        The action is first compressed to its bitmap representation before being stored.
        """
        compressed_action = self.card_mapper.compress_to_hand_size(action)  # Compress the action to bitmap
        self.replay_buffer.push(
            State(self._hand, self.cards_played, self.num_cards_played), # s
            compressed_action,                                           # a
            reward,                                                      # r
            State(new_hand, new_cards_played, new_num_cards_played)      # s'
        )  # Store the experience in the replay buffer

from bitarray import bitarray
import numpy as np
import torch

from CompleteState import CompleteState
from DQNPlayer import DQNPlayer
from DQNPlayer_HandBitmap import DQNPlayer_HandBitmap
from Deck import DeckHelper
from Player import Player
import QCheaterNetwork
from QNetwork import QNetwork
from State import State
from card_mapper import CardMapper
from QCheaterNetwork import QCheaterNetwork


class DQNCheater(DQNPlayer_HandBitmap):
    """
        Class used for the implementation of the agents in the 4th experiment.
        The Cheater agent extends the DQNPlayer_HandBitmap class by adding a bitarray to track the cards played in the current round.
        (it is the cheater since it can see the cards played by other players in the current round).
    """
    def __init__(self, player_id: int, total_cards: int, cards_per_player: int, 
                 discount_factor: float = 0.99, epsilon_decay: float = 0.995, 
                 replay_buffer_capacity: int = 10000, batch_size: int = 32,
                 SYNC_TARGET_EVERY: int = 10):
        super().__init__(player_id, total_cards, cards_per_player, discount_factor, 
                         epsilon_decay, replay_buffer_capacity, batch_size,
                         SYNC_TARGET_EVERY)
        self.played_cards_in_round = bitarray(total_cards)  # Bitarray to track the cards played in the current round
        hand_bitmap_size = self.cards_per_player  # Each card is represented by a bit in the bitmap

        self.policy_network = QCheaterNetwork(input_dim=(total_cards * 3), num_actions=2**hand_bitmap_size)  # Initialize the policy network with bitmap size
        self.target_network = QCheaterNetwork(input_dim=(total_cards * 3), num_actions=2**hand_bitmap_size)  # Initialize the target network with bitmap size

    def update_state(self, action: bitarray, played_cards_in_round: bitarray, reward: int):
        """Update the player's state based on the cards played by other players in the current round.
            Then it saves the experience (state, action, reward, next_state) in the replay buffer for training the Q-network."""
        
        # Compute the next state after the action has been taken
        new_hand = self._hand & ~action  # Update the player's hand by removing the played cards
        new_cards_played = self.cards_played | played_cards_in_round  # Update the played cards bitarray with all the played cards in the current round
        new_num_cards_played = self.played_cards_in_round  # Count the number of cards played in the current round

        # Save the experience in the replay buffer
        self._experience_push(new_hand, new_cards_played, action, new_num_cards_played, reward)  # Save the experience in the replay buffer

        # State Update
        self._hand = new_hand  # Update the player's hand
        self.cards_played = new_cards_played  # Update the played cards bitarray
        self.played_cards_in_round = played_cards_in_round  # Update the played cards bitarray with all the played cards in the current round

        self.epsilon *= self.epsilon_decay  # Decay the exploration rate
        if len(self.replay_buffer) >= self.batch_size:
            self._training_step(batch_size=self.batch_size, gamma=self.discount_factor, SYNC_TARGET_EVERY=self.sync_target_every)  # Perform a training step for the Q-network
    
    def _experience_push(self, new_hand, new_cards_played, action, new_num_cards_played, reward):
        """
        Push the experience into the replay buffer.
        The action is first compressed to its bitmap representation before being stored.
        """
        compressed_action = self.card_mapper.compress_to_hand_size(action)  # Compress the action to bitmap
        self.replay_buffer.push(
            CompleteState(self._hand, self.cards_played, self.played_cards_in_round), # s
            compressed_action,                                           # a
            reward,                                                      # r
            CompleteState(new_hand, new_cards_played, new_num_cards_played)      # s'
        )  # Store the experience in the replay buffer

    def q_play_cards(self) -> bitarray:
        """Selects the best action based on the Q-network's predictions."""
        q_values = self.policy_network(CompleteState(self._hand, self.cards_played, self.played_cards_in_round))  # Get Q-values for the current state
        best_action_index = int(torch.argmax(q_values).item()) + 1  # Get the index of the best action
        # Convert the best action byte into a bitarray representing the cards to play
        best_action_bitarray = DeckHelper.int_to_bitarray(self.total_cards, best_action_index)  # Convert to bitarray
        return best_action_bitarray
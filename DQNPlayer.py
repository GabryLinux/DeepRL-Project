
import math
from random import random

from bitarray import bitarray
import numpy as np
import gymnasium as gym
import torch
from Deck import DeckHelper
from Player import Player
from gymnasium import spaces

import QNetwork
from ReplayBuffer import ReplayBuffer
from State import State



class DQNPlayer(Player):
    def __init__(
            self, player_id: int, total_cards: int, cards_per_player: int, 
            discount_factor: float = 0.99, epsilon_decay: float = 0.995, replay_buffer_capacity: int = 10000, batch_size: int = 32,
            SYNC_TARGET_EVERY: int = 10,
            disable_network_building: bool = False  # New parameter to control network building
        ):
        super().__init__(player_id)

        # -------- Game Parameters --------
        self.total_cards = total_cards                      # Number of cards in the deck
        self.cards_per_player = cards_per_player  # Number of cards each player starts with

        self.cards_played = bitarray(total_cards)  # Bitarray to track played cards
        self.cards_played.setall(0)  # Initialize all cards as not played

        self._hand = bitarray(total_cards)  # Bitarray to represent the player's hand
        self._hand.setall(0)  # Initialize all cards in hand as not present

        self._action = bitarray(total_cards)  # Bitarray to represent the action (cards to play)
        self._action.setall(0)  # Initialize all actions as not taken

        self.num_cards_played = 0

        # -------- Learning Parameters --------
        # REMEMBER TO DISABLE THE POLICY AND TARGET NETWORK BUILDING WHEN INHERITING FROM THIS CLASS (like DQNPlayer_HandBitmap and DQNCheater)
        
        if not disable_network_building:
            self.policy_network = QNetwork.QNetwork(num_cards=total_cards, num_actions=(2**total_cards)) #QNetwork.QNetwork(num_cards=total_cards, num_actions=(2**self.total_cards - 1))  # Initialize the Q-network
            self.target_network = QNetwork.QNetwork(num_cards=total_cards, num_actions=(2**total_cards)) #QNetwork.QNetwork(num_cards=total_cards, num_actions=(2**self.total_cards - 1))  # Initialize the target network
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)  # Experience replay buffer
        self.batch_size = batch_size  # Batch size for training
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.sync_target_every = SYNC_TARGET_EVERY  # Frequency of syncing the target network


    def play_cards(self) -> bitarray:
        available_cards = DeckHelper.bitarray_to_cards(self.total_cards, self._hand)
        if not available_cards:
            b = bitarray(self.total_cards)
            b.setall(0)  # No cards to play
            _action = b
        if random() < self.epsilon:
            # Exploration Case: Randomly choose a subset of available cards to play
            _action = self._random_play(self._hand)
        else:
            # Exploitation: Use the Q-network to select the best action
            _action = self.q_play_cards()
        self._hand &= ~_action  # Update the player's hand by removing the played cards
        return _action
    
    def q_play_cards(self) -> bitarray:
        """Selects the best action based on the Q-network's predictions."""
        q_values = self.policy_network(State(self._hand, self.cards_played, self.num_cards_played))  # Get Q-values for the current state
        best_action_index = int(torch.argmax(q_values).item()) + 1  # Get the index of the best action
        # Convert the best action byte into a bitarray representing the cards to play
        best_action_bitarray = DeckHelper.int_to_bitarray(self.total_cards, best_action_index)  # Convert to bitarray
        return best_action_bitarray

    def update_state(self, action: bitarray, played_cards_in_round: bitarray, reward: int):
        """Update the player's state based on the cards played by other players in the current round.
            Then it saves the experience (state, action, reward, next_state) in the replay buffer for training the Q-network."""
        
        # Compute the next state after the action has been taken
        new_hand = self._hand & ~action  # Update the player's hand by removing the played cards
        new_cards_played = self.cards_played | played_cards_in_round  # Update the played cards bitarray with all the played cards in the current round
        new_num_cards_played = sum(new_cards_played)  # Count the number of cards played in the current round

        # Save the experience in the replay buffer
        self._experience_push(new_hand, new_cards_played, action, new_num_cards_played, reward)  # Save the experience in the replay buffer

        # State Update
        self._hand = new_hand  # Update the player's hand
        self.cards_played = new_cards_played  # Update the played cards bitarray
        self.num_cards_played = new_num_cards_played  # Update the number of cards placed
        self.cards_played = new_cards_played  # Update the played cards bitarray with all the played cards in the current round

        self.epsilon *= self.epsilon_decay  # Decay the exploration rate
        if len(self.replay_buffer) >= self.batch_size:
            self._training_step(batch_size=self.batch_size, gamma=self.discount_factor, SYNC_TARGET_EVERY=self.sync_target_every)  # Perform a training step for the Q-network

    
    def _experience_push(self, new_hand, new_cards_played, action, new_num_cards_played, reward):
        """Push the experience (state, action, reward, next_state) into the replay buffer."""
        self.replay_buffer.push(
            State(self._hand, self.cards_played, self.num_cards_played), # s
            action,                                                      # a
            reward,                                                      # r
            State(new_hand, new_cards_played, new_num_cards_played)      # s'
        )

    def _training_step(self, batch_size: int, gamma: float, SYNC_TARGET_EVERY: int = 10):
        """Perform a single training step for the Q-network using experiences from the replay buffer."""
        estimate_error, target_error = QNetwork.TrainingUtils.train_dqn_step(
            policy_net=self.policy_network,
            target_net=self.target_network,  # For simplicity, using the same network as target; in practice, use a separate target network
            replay_buffer=self.replay_buffer,
            batch_size=batch_size,
            gamma=gamma,
            SYNC_EVERY=SYNC_TARGET_EVERY
        )
        if estimate_error is not None and target_error is not None:
            self.policy_network.errors.append(estimate_error)   # Store the estimation error for analysis
            self.target_network.errors.append(target_error)     # Store the target error for analysis


    def reset(self):
        """Reset the player's state for a new game."""
        self.cards_played.setall(0)  # Reset the played cards bitarray
        self._hand.setall(0)  # Reset the player's hand
        self._action.setall(0)  # Reset the action bitarray
        self.num_cards_played = 0  # Reset the number of cards played

    def get_errors(self):
        """Get the list of estimation and target errors for analysis."""
        return self.policy_network.errors, self.target_network.errors

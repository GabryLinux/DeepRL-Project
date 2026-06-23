

from collections import OrderedDict
import math
from typing import Any, Optional

from bitarray import bitarray
import numpy as np

from Deck import Deck, DeckHelper
from Player import Player
import gymnasium as gym
from gymnasium import spaces

class GameEnv(gym.Env):
    """
        Implementation of the game environment for the 1st, 2nd and 3rd experiments.
        It extends the gym.Env class by implementing the methods required by the Gym API.
    """
    def __init__(self, deck: Deck):
        """Initialize the game with a deck of cards and an empty list of players."""
        self.deck = deck
        self.players: list[Player] = []
        self.old_hands: OrderedDict[Player, bitarray] = OrderedDict()  # Store the old hands of players before they play
        self.played_cards_in_round: OrderedDict[Player, bitarray] = OrderedDict()
        

    def add_player(self, player: Player):
        """Add a player to the game"""
        self.players.append(player)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the game by resetting all the players and shuffling the deck and dealing cards to players. Returns the initial observation and info."""
        super().reset(seed=seed)
        
        for player in self.players:
            player.reset()
        self.deck.reset_deck()
        
        observation: np.ndarray = np.array([])
        
        info = {}
        
        return observation, info
    

    def step(self, action: OrderedDict[Player, bitarray]) -> tuple[dict, float, bool, bool, dict[str, Any]]:
        """Play a round of the game. \n
            It returns:
                - Dict containing:
                    - list[bitarray] of the cards played by each player in the current round.
                    - OrderedDict[Player, int] of the rewards for each player in the current round.
                - 0 (float) as a placeholder for the reward (not used in this context).
                - True (bool) indicating that the round is complete.
                - True (bool) indicating that the game is not over yet.
                - dict[str, Any] containing:
                    - OrderedDict[Player, int] of the rewards for each player in the current round.
                    - bitarray of revealed cards at the end of the round.
        """
        cards_placed_before: OrderedDict[Player, int] = OrderedDict()
        cards_counter = 0
        for player in self.players:
            cards_placed_before[player] = cards_counter
            cards_counter += sum(1 for card in action[player] if card)
            self.played_cards_in_round[player] = action[player]

        # CARD REVEALING PHASE
        played_cards = DeckHelper.aggregate_bitarrays(list(self.played_cards_in_round.values()))

        # SCORING PHASE
        rewards = self.compute_rewards(self.played_cards_in_round)

        # ILLEGAL MOVE CHECK
        illegal_move_detected = self.illegal_move_check(action, rewards)

        # UPDATING THE NEXT FIRST PLAYER FOR THE NEXT ROUND
        self.set_next_first_player(rewards) 
        
        return {
            "played_cards": played_cards, 
            "rewards": rewards
        }, 0, self.is_game_over(), illegal_move_detected, {}

    def start_game(self):
        """Start the game by dealing cards to players."""
        # If there are no players, we cannot start the game
        if not self.players:
            raise ValueError("No players in the game. Cannot start.")
        
        # Cards to deal to each player. If number of players isn't a divisor of the total number,
        # some cards will remain in the deck after dealing.
        cards_per_player = self.deck.num_cards // len(self.players)
        for player in self.players:
            cards = self.deck.draw_cards(cards_per_player)
            player.receive_cards(DeckHelper.cards_to_bitarray(self.deck.num_cards, cards))
            self.old_hands[player] = player.get_hand().copy()  # Store the old hand of the player before they play

        num_bytes = math.ceil(self.deck.num_cards / 8)  # 7
        max_cards_before = self.deck.num_cards - cards_per_player
        self.observation_space = spaces.Dict(
            {
                # Hand of the player: 1 float per every card in the deck, 0.0 or 1.0
                "hand": spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(self.deck.num_cards,), 
                    dtype=np.uint8
                ),
                
                # Cards history: 1 float per every card in the deck, 0.0 or 1.0
                "played_history": spaces.Box(
                    low=0, 
                    high=1, 
                    shape=(self.deck.num_cards,), 
                    dtype=np.uint8
                ),
                
                # Number of cards placed by the player in the current round: 1 int
                "cards_placed": spaces.Discrete(max_cards_before + 1)
            }
        )
        self.action_space = spaces.Discrete(2 ** self.deck.num_cards) # every possible combination of cards to play


    def is_game_over(self) -> bool:
        """It is game over when no player has any cards left in their hand, or if there is only one player left with cards."""
        active_players = [player for player in self.players if player.has_cards()]
        if len(active_players) <= 1:
            return True
        return False
    
    def illegal_move_check(self, actions: OrderedDict[Player, bitarray], rewards: OrderedDict[Player, int]) -> bool:
        """Check if the played cards are a subset of the player's hand. If any illegal move, related reward will be negative. Returns True if any player made an illegal move, False otherwise."""
        illegal_move_detected = False
        for player in self.players:
            if actions[player] & self.old_hands[player] != actions[player]:
                print(f"Player {player.get_id()} made an illegal move:\n")
                print(f"Player's hand: {self.old_hands[player]}\n")
                print(f"Played cards: {actions[player]}\n")
                rewards[player] = -1  # Penalize illegal move with a negative reward
                illegal_move_detected = True

        # HAND UPDATE PHASE: Update the players' hands after the round, regardless of whether there was an illegal move or not.
        for player in self.players:
            self.old_hands[player] = player.get_hand().copy()  # Store the old hand of the player before they play
            
        return illegal_move_detected

    def joker_players(self, played_cards_in_round: OrderedDict[Player, bitarray]) -> list[Player]:
        """
        It returns a list of players who played the joker in the current round. 
        If any player has played the joker, they will be rewarded with the winning suit's reward.
        """
        joker_players = []
        for player in self.players:
            cards_played = DeckHelper.bitarray_to_cards(self.deck.num_cards, played_cards_in_round[player])
            for card in cards_played:
                if DeckHelper.is_joker(self.deck, card):
                    joker_players.append(player)
        return joker_players

    def winning_suit(self, played_cards_in_round, num_suits=4) -> int:
        """
        Given all the cards played, it computes the total value of each suit.
        
        Returns the numerical index of the suit with the highest total value.
        """
        collected_played_cards_bitarray = DeckHelper.aggregate_bitarrays(list(played_cards_in_round.values()))
        
        cards = DeckHelper.bitarray_to_cards(self.deck.num_cards, collected_played_cards_bitarray)
        scores = [0 for _ in range(num_suits)]
        for card in cards:
            suit = DeckHelper.card_to_suit(self.deck, card)  # Usa metodo esistente
            rank = DeckHelper.card_to_rank(self.deck, card)  # Usa metodo esistente
            scores[suit] += rank

        return max(range(len(scores)), key=scores.__getitem__)
    
    def compute_rewards(self, played_cards_in_round: OrderedDict[Player, bitarray]) -> OrderedDict[Player, int]:
        """
        Compute the rewards for each player based on the winning suit.
        If any player has played the joker, will be rewarded with the winning suit's reward.
        In case of multiple players playing the joker, they will share the reward equally.
        """
        rewards_per_player = self.compute_rewards_for_suit_for_player(played_cards_in_round)
        
        # Initializes the total rewards for each suit across all players
        rewards_per_suit = [0 for _ in range(self.deck.suits)]
        for player in self.players:
            for suit in range(self.deck.suits):
                rewards_per_suit[suit] += rewards_per_player[player][suit]
        
        # Determine the winning suit based on the total rewards for each suit
        winning_suit = max(range(len(rewards_per_suit)), key=lambda i: (rewards_per_suit[i], i))
        print(f"Winning suit: {winning_suit} with reward {rewards_per_suit[winning_suit]}") 

        final_rewards: OrderedDict[Player, int] = OrderedDict()
        for player in self.players:
            # Compute the reward for each player based on the winning suit
            final_rewards[player] = rewards_per_player[player][winning_suit]
        print(f"Rewards per player per suit: {[f'{player.get_id()}: {rewards}' for player, rewards in rewards_per_player.items()]}")  # Debug: Print rewards per player per suit

        # Joker check: If any player has played the joker, they will be rewarded with the winning suit's reward.
        joker_players = self.joker_players(played_cards_in_round)
        if joker_players:
            print(f"Joker players: {[player.get_id() for player in joker_players]}")
            
            total_winning_pool = sum(final_rewards[p] for p in self.players)
            shared_reward = total_winning_pool // len(joker_players)
            
            for player in self.players:
                if player in joker_players:
                    final_rewards[player] = shared_reward # Reward joker players with the shared reward
                else:
                    final_rewards[player] = 0 # Penalize non-joker players by setting their reward to 0 if any player played the joker

        return final_rewards

    
    def compute_rewards_for_suit_for_player(
            self,
            cards_played_for_player: dict[Player, bitarray],
    ) -> OrderedDict[Player, list[int]]:
        """
        Compute the reward for each player based on the cards they played in the current round.
        This method is used to compute the reward for each player based on the cards they played in the current round.
        """
        rewards_per_player = OrderedDict()
        for player in self.players:
            rewards = [0 for _ in range(self.deck.suits)]
            cards = DeckHelper.bitarray_to_cards(self.deck.num_cards, cards_played_for_player[player])
            for card in cards:
                if DeckHelper.is_joker(self.deck, card):
                    continue  # Jokers do not contribute to the reward
                card_suit = DeckHelper.card_to_suit(self.deck, card)
                rank = DeckHelper.card_to_rank(self.deck, card)
                rewards[card_suit] += rank
            rewards_per_player[player] = rewards
        return rewards_per_player
        

    def compute_reward_for_suit(self, suit: int, cards_played: bitarray):
        """
        Compute the reward for a given suit in cards_played.
        """
        cards = DeckHelper.bitarray_to_cards(self.deck.num_cards, cards_played)
        reward = 0
        for card in cards:
            if DeckHelper.is_joker(self.deck, card):
                continue  # Jokers do not contribute to the reward
            card_suit = DeckHelper.card_to_suit(self.deck, card)
            rank = DeckHelper.card_to_rank(self.deck, card)
            if card_suit == suit:
                reward += rank
        return reward  # Placeholder for reward calculation based on the winning suit
    

    def set_next_first_player(self, scores: OrderedDict[Player, int]):
        """Set the next first player for the next round."""
        next_first_player = max(scores, key=scores.get) # type: ignore
        
        # Trova l'indice del giocatore
        indices = [i for i, p in enumerate(self.players) if p.get_id() == next_first_player.get_id()]
        
        if not indices:
            print(f"ERROR: Player {next_first_player} not found!")
            return
        
        # Ruota la lista all'indice trovato
        idx = indices[0]
        self.players = self.players[idx:] + self.players[:idx]
        

from collections import OrderedDict
import unittest

from bitarray import bitarray
from Game import GameEnv as Game
from Player import Player
from Deck import Deck, DeckHelper
from RandomPlayer import RandomPlayer

class GameTest(unittest.TestCase):
    def test_game_initialization(self) -> Game:
        CARDS = 52
        JACKS = 2
        TOTAL_CARDS = CARDS + JACKS
        deck = Deck(num_cards=CARDS)
        game = Game(deck=deck)
        random_player1 = RandomPlayer(player_id=1, total_cards=TOTAL_CARDS)
        random_player2 = RandomPlayer(player_id=2, total_cards=TOTAL_CARDS)
        game.add_player(random_player1)
        game.add_player(random_player2)

        assert len(game.players) == 2
        return game

    def test_game_start(self):
        game = self.test_game_initialization()
        game.start_game()

        for player in game.players:
            assert player.has_cards()  # Players should have cards after the game starts


    def test_agent_play_cards(self):
        game = self.test_game_initialization()
        game.start_game()

        #game.play_round()
        return
    
    def test_game_over(self):
        game = self.test_game_initialization()
        game.start_game()

        while not game.is_game_over():
            actions: OrderedDict[Player, bitarray] = OrderedDict()
            for player in game.players:
                actions[player] = player.play_cards()

            game.step(actions)
        
        assert game.is_game_over()  # The game should be over when no players have cards left

    def test_reward_players(self):
        game = self.test_game_initialization()
        game.start_game()

        actions: OrderedDict[Player, bitarray] = OrderedDict()
        for player in game.players:
            actions[player] = player.play_cards()

        game.step(actions)
        
        played_cards_in_round = game.played_cards_in_round
        winning_suit = game.winning_suit(played_cards_in_round)
        rewards = game.compute_rewards(played_cards_in_round)
        
        if game.joker_players(played_cards_in_round):
            return

        

        